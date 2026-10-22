/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 21:40:00
 * REASON FOR CREATION: Step 3 of GPU_MILESTONE_M2000M.md and milestone G1/G2 of the
 *   GPU plan. The sm_50 kernels are proven -- 253 of 253 tensors decoded with zero
 *   mismatches, and the fused decode+matvec runs 16x faster than the host float path --
 *   but they are driven by standalone C programs. `grep -rn -i cuda wpc-runtime/src`
 *   returned nothing, so the engine that actually answers a question has never touched
 *   the card. This module is the missing link: it lets the Rust engine own the CUDA
 *   context, upload the packed model once, and launch the existing kernel.
 * MECHANICS: The CUDA Driver API is reached through dlopen/dlsym at run time rather than
 *   linked at build time. That is deliberate: the same binary then builds and runs on a
 *   machine with no CUDA at all, and `Gpu::open` simply returns an error instead of the
 *   program failing to link. Only the twelve entry points the forward pass needs are
 *   bound. Kernels are loaded from ahead-of-time PTX (wpc4_gemv_sm50.ptx) because the
 *   relocated toolchain targets sm_50 and the driver JITs it in ~40 ms, once.
 * SYSTEM PART: WPC / GPU offload lane, runtime side.
 * ARCHITECTURE FUNCTION: Layer 4 (AI Runtime) of AIONS_MASTER_BUILD_PLAN.md. Provides
 *   the device-resident weight buffer and the matrix-vector product that dominates
 *   single-token decoding. Residency is the whole point: uploading the model costs
 *   6.672 s while one fused matvec costs 1.043 ms, so weights are uploaded once and
 *   every tensor is addressed as an offset into that one buffer.
 * DEPENDENCIES/LINKS: libcuda.so.1 (WSL: /usr/lib/wsl/lib) via dlopen; the PTX built
 *   from gpu/wpc4-decode/wpc4_gemv_sm50.cu; block layout mirrors QuantBlockV4 in
 *   wpc-format/src/lib.rs (BLOCK_SIZE_V4 128, PACKED_BYTES_V4 64, SIZE 68). Consumed by
 *   WpcLinearV4::matvec in wpc_weights_v4.rs.
 * TECH STACK: Rust 2021, no new crates. Rust because the weights must stay resident,
 *   which forces the CUDA context to live in the same long-lived process as the token
 *   loop -- and that process is this engine. A separate C or Python host would mean
 *   either re-uploading per token or a process boundary per layer, either of which
 *   costs more than the card gains.
 * LOCAL WORKSPACE: C:\temp\aions-multiturn-2026-08-25\wpc-runtime\src\gpu.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/resident-multi-turn
 * ==========================================
 */

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

/// Mirrors QuantBlockV4: f16 zero_point, f16 scale, then 64 packed bytes.
pub const WPC4_BLOCK_BYTES: usize = 68;
/// Threads per block in wpc4_gemv_sm50.cu. The kernel's shared array is sized by this
/// constant on the device side, so the launch passes 0 dynamic shared bytes.
const GEMV_THREADS: u32 = 256;

const CUDA_SUCCESS: c_int = 0;
#[cfg(unix)]
const RTLD_NOW: c_int = 2;

const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;
const ATTR_SM_COUNT: c_int = 16;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;

// Both platforms can reach the driver; only the way of asking differs. Windows talks to
// nvcuda.dll directly, with no virtualisation layer between the process and the card --
// which is the reason to prefer it here, see the note on cuCtxSynchronize below.
#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// Names to try, most specific first. WSL keeps the driver outside the default search
/// path, so the full path is tried before the bare name.
#[cfg(unix)]
const DRIVER_NAMES: &[&str] = &[
    "/usr/lib/wsl/lib/libcuda.so.1",
    "libcuda.so.1",
    "libcuda.so",
];
#[cfg(windows)]
const DRIVER_NAMES: &[&str] = &["nvcuda.dll"];

unsafe fn load_driver(name: &str) -> *mut c_void {
    let c = match CString::new(name) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };
    #[cfg(unix)]
    {
        dlopen(c.as_ptr(), RTLD_NOW)
    }
    #[cfg(windows)]
    {
        LoadLibraryA(c.as_ptr())
    }
}

unsafe fn find_symbol(handle: *mut c_void, name: &CStr) -> *mut c_void {
    #[cfg(unix)]
    {
        dlsym(handle, name.as_ptr())
    }
    #[cfg(windows)]
    {
        GetProcAddress(handle, name.as_ptr())
    }
}

type FnInit = unsafe extern "C" fn(u32) -> CuResult;
type FnDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type FnDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
type FnDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult;
type FnPrimaryCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type FnCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;
type FnModuleLoad = unsafe extern "C" fn(*mut CuModule, *const c_char) -> CuResult;
type FnModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type FnMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type FnMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type FnMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
#[allow(clippy::type_complexity)]
type FnLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type FnCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type FnGetErrorString = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

struct Api {
    device_get_name: FnDeviceGetName,
    device_get_attribute: FnDeviceGetAttribute,
    mem_get_info: FnMemGetInfo,
    module_load: FnModuleLoad,
    module_get_function: FnModuleGetFunction,
    mem_alloc: FnMemAlloc,
    mem_free: FnMemFree,
    memcpy_htod: FnMemcpyHtoD,
    memcpy_dtoh: FnMemcpyDtoH,
    launch_kernel: FnLaunchKernel,
    ctx_synchronize: FnCtxSynchronize,
    get_error_string: FnGetErrorString,
}

/// Look up exactly the symbol named. No guessing.
///
/// This used to try a `_v2` suffix first and fall back to the plain name, on the theory
/// that the Driver API versioned its calls when 64-bit pointers arrived. That is true of
/// the memory calls and false of the rest -- and the failure mode is silent. `cuCtxSynchronize_v2`
/// exists and is not an equivalent of `cuCtxSynchronize`; binding to it made every
/// synchronisation report CUDA error 709, "context is destroyed", on a context that was
/// demonstrably alive, on both Linux and Windows. Hours went into suspecting WSL, the
/// card, the kernel and the context type, when the truth was that the wrong function was
/// being called.
///
/// So each binding below names the exact symbol it wants, `_v2` included where the
/// versioned form is the correct one.
unsafe fn sym(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
    let c = CString::new(name).ok()?;
    let p = find_symbol(handle, &c);
    if p.is_null() {
        return None;
    }
    if std::env::var("AIONS_GPU_TRACE").is_ok() {
        eprintln!("cuda-bind: {name}");
    }
    Some(p)
}

macro_rules! bind {
    ($handle:expr, $name:literal, $ty:ty) => {{
        let p = sym($handle, $name)
            .ok_or_else(|| anyhow::anyhow!("libcuda is missing the symbol {}", $name))?;
        std::mem::transmute::<*mut c_void, $ty>(p)
    }};
}

/// A block of device memory, freed when dropped.
pub struct DeviceBuffer {
    ptr: CuDevicePtr,
    len: usize,
    free: FnMemFree,
}

impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                (self.free)(self.ptr);
            }
        }
    }
}

pub struct Gpu {
    api: Api,
    _ctx: CuContext,
    device: CuDevice,
    _module: CuModule,
    gemv: CuFunction,
}

impl Gpu {
    /// Open device 0 and load the fused decode+matvec kernel.
    ///
    /// `ptx_path` defaults to `AIONS_GPU_PTX`. Returns an error rather than panicking
    /// when there is no driver, no card, or no PTX, so the caller can fall back to the
    /// CPU path on a machine that simply has no GPU.
    pub fn open(ptx_path: Option<&str>) -> anyhow::Result<Gpu> {
        let ptx = match ptx_path {
            Some(p) => p.to_string(),
            None => std::env::var("AIONS_GPU_PTX").map_err(|_| {
                anyhow::anyhow!("no PTX path given and AIONS_GPU_PTX is not set")
            })?,
        };
        if !std::path::Path::new(&ptx).is_file() {
            anyhow::bail!("PTX not found: {ptx}");
        }

        unsafe {
            // Try the WSL location explicitly before the loader's search path: WSL puts
            // the driver in /usr/lib/wsl/lib, which is not always on the default path.
            let mut handle = std::ptr::null_mut();
            for name in DRIVER_NAMES {
                handle = load_driver(name);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                anyhow::bail!("libcuda could not be loaded; is a driver installed?");
            }

            let init = bind!(handle, "cuInit", FnInit);
            let device_get = bind!(handle, "cuDeviceGet", FnDeviceGet);
            // The primary context rather than cuCtxCreate. cuCtxCreate hands back a
            // context owned by this thread's context stack, and on WSL that ownership
            // proved fragile: the driver reported error 709 "context is destroyed" on
            // an ordinary allocation, at a different point on each run. The primary
            // context is reference-counted by the driver, shared with anything else in
            // the process, and is what every mature runtime uses.
            let primary_retain = bind!(handle, "cuDevicePrimaryCtxRetain", FnPrimaryCtxRetain);
            let ctx_set_current = bind!(handle, "cuCtxSetCurrent", FnCtxSetCurrent);
            let api = Api {
                device_get_name: bind!(handle, "cuDeviceGetName", FnDeviceGetName),
                device_get_attribute: bind!(
                    handle,
                    "cuDeviceGetAttribute",
                    FnDeviceGetAttribute
                ),
                // The five memory calls genuinely are the versioned ones: their v1 forms
                // take 32-bit sizes and would truncate a 2 GB model.
                mem_get_info: bind!(handle, "cuMemGetInfo_v2", FnMemGetInfo),
                module_load: bind!(handle, "cuModuleLoad", FnModuleLoad),
                module_get_function: bind!(
                    handle,
                    "cuModuleGetFunction",
                    FnModuleGetFunction
                ),
                mem_alloc: bind!(handle, "cuMemAlloc_v2", FnMemAlloc),
                mem_free: bind!(handle, "cuMemFree_v2", FnMemFree),
                memcpy_htod: bind!(handle, "cuMemcpyHtoD_v2", FnMemcpyHtoD),
                memcpy_dtoh: bind!(handle, "cuMemcpyDtoH_v2", FnMemcpyDtoH),
                launch_kernel: bind!(handle, "cuLaunchKernel", FnLaunchKernel),
                ctx_synchronize: bind!(handle, "cuCtxSynchronize", FnCtxSynchronize),
                get_error_string: bind!(handle, "cuGetErrorString", FnGetErrorString),
            };

            check(&api, init(0), "cuInit")?;
            let mut device: CuDevice = 0;
            check(&api, device_get(&mut device, 0), "cuDeviceGet")?;
            let mut ctx: CuContext = std::ptr::null_mut();
            check(
                &api,
                primary_retain(&mut ctx, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            check(&api, ctx_set_current(ctx), "cuCtxSetCurrent")?;

            let ptx_c = CString::new(ptx.as_str())?;
            let mut module: CuModule = std::ptr::null_mut();
            check(
                &api,
                (api.module_load)(&mut module, ptx_c.as_ptr()),
                "cuModuleLoad",
            )?;

            let name = CString::new("wpc4_gemv")?;
            let mut gemv: CuFunction = std::ptr::null_mut();
            check(
                &api,
                (api.module_get_function)(&mut gemv, module, name.as_ptr()),
                "cuModuleGetFunction(wpc4_gemv)",
            )?;

            Ok(Gpu {
                api,
                _ctx: ctx,
                device,
                _module: module,
                gemv,
            })
        }
    }

    pub fn device_name(&self) -> String {
        let mut buf = vec![0u8; 128];
        unsafe {
            if (self.api.device_get_name)(buf.as_mut_ptr() as *mut c_char, 128, self.device)
                != CUDA_SUCCESS
            {
                return "(unknown)".to_string();
            }
            CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// (compute capability major, minor, streaming multiprocessor count)
    pub fn capability(&self) -> (i32, i32, i32) {
        let attr = |a: c_int| -> i32 {
            let mut v: c_int = 0;
            unsafe {
                if (self.api.device_get_attribute)(&mut v, a, self.device) != CUDA_SUCCESS {
                    return -1;
                }
            }
            v as i32
        };
        (attr(ATTR_CC_MAJOR), attr(ATTR_CC_MINOR), attr(ATTR_SM_COUNT))
    }

    /// Wait for the device and surface any error it is holding.
    ///
    /// Launches are asynchronous, so a fault inside a kernel is not reported by the
    /// launch call. Calling this between steps turns "something died somewhere" into
    /// "this step died".
    pub fn sync(&self) -> anyhow::Result<()> {
        unsafe {
            check(
                &self.api,
                (self.api.ctx_synchronize)(),
                "cuCtxSynchronize",
            )
        }
    }

    /// (free bytes, total bytes) of device memory.
    pub fn mem_info(&self) -> anyhow::Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe {
            check(
                &self.api,
                (self.api.mem_get_info)(&mut free, &mut total),
                "cuMemGetInfo",
            )?;
        }
        Ok((free, total))
    }

    /// Copy bytes to the card and keep them there.
    ///
    /// The packed model is uploaded through this once; every tensor afterwards is an
    /// offset into the returned buffer. Re-uploading per call would cost roughly ten
    /// thousand times more than the multiply it feeds.
    pub fn upload(&self, bytes: &[u8]) -> anyhow::Result<DeviceBuffer> {
        let mut ptr: CuDevicePtr = 0;
        unsafe {
            check(
                &self.api,
                (self.api.mem_alloc)(&mut ptr, bytes.len().max(1)),
                "cuMemAlloc",
            )?;
            let r = (self.api.memcpy_htod)(ptr, bytes.as_ptr() as *const c_void, bytes.len());
            if let Err(e) = check(&self.api, r, "cuMemcpyHtoD(upload)") {
                (self.api.mem_free)(ptr);
                return Err(e);
            }
        }
        Ok(DeviceBuffer {
            ptr,
            len: bytes.len(),
            free: self.api.mem_free,
        })
    }

    /// Allocate device memory without writing to it.
    pub fn alloc(&self, len: usize) -> anyhow::Result<DeviceBuffer> {
        let mut ptr: CuDevicePtr = 0;
        unsafe {
            check(
                &self.api,
                (self.api.mem_alloc)(&mut ptr, len.max(1)),
                "cuMemAlloc",
            )?;
        }
        Ok(DeviceBuffer {
            ptr,
            len,
            free: self.api.mem_free,
        })
    }

    pub fn write_f32(&self, buf: &DeviceBuffer, data: &[f32]) -> anyhow::Result<()> {
        let bytes = std::mem::size_of_val(data);
        anyhow::ensure!(bytes <= buf.len, "write_f32: {bytes} bytes into {} ", buf.len);
        unsafe {
            check(
                &self.api,
                (self.api.memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD",
            )?;
        }
        Ok(())
    }

    pub fn read_f32(&self, buf: &DeviceBuffer, out: &mut [f32]) -> anyhow::Result<()> {
        let bytes = std::mem::size_of_val(out);
        anyhow::ensure!(bytes <= buf.len, "read_f32: {bytes} bytes from {}", buf.len);
        unsafe {
            check(
                &self.api,
                (self.api.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, buf.ptr, bytes),
                "cuMemcpyDtoH",
            )?;
        }
        Ok(())
    }

    /// y = W * x, with W held packed in WPC v4 inside `weights` at `byte_offset`.
    ///
    /// One thread block per output row, which is what the kernel expects. `x` and `y`
    /// are device buffers so a forward pass can keep activations on the card instead of
    /// paying a round trip per layer.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        weights: &DeviceBuffer,
        byte_offset: usize,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        blocks_per_row: u32,
        n_rows: u32,
    ) -> anyhow::Result<()> {
        let needed = blocks_per_row as usize * n_rows as usize * WPC4_BLOCK_BYTES;
        anyhow::ensure!(
            byte_offset + needed <= weights.len,
            "gemv would read past the resident buffer: offset {byte_offset} + {needed} > {}",
            weights.len
        );

        let w_ptr: CuDevicePtr = weights.ptr + byte_offset as u64;
        let x_ptr = x.ptr;
        let y_ptr = y.ptr;
        let bpr = blocks_per_row;
        let rows = n_rows;

        let mut params: [*mut c_void; 5] = [
            &w_ptr as *const CuDevicePtr as *mut c_void,
            &x_ptr as *const CuDevicePtr as *mut c_void,
            &y_ptr as *const CuDevicePtr as *mut c_void,
            &bpr as *const u32 as *mut c_void,
            &rows as *const u32 as *mut c_void,
        ];

        unsafe {
            check(
                &self.api,
                (self.api.launch_kernel)(
                    self.gemv,
                    n_rows,
                    1,
                    1,
                    GEMV_THREADS,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(wpc4_gemv)",
            )?;
        }
        // A launch is asynchronous, so without this the caller could read `y` before the
        // kernel has written it. The earlier version of this file skipped the call
        // because it appeared to fail; that was a wrong symbol binding, not a driver
        // fault -- see the note on `sym`.
        self.sync()
    }
}

fn cuda_error(api: &Api, code: CuResult, what: &str) -> anyhow::Error {
    let mut msg: *const c_char = std::ptr::null();
    let text = unsafe {
        if (api.get_error_string)(code, &mut msg) == CUDA_SUCCESS && !msg.is_null() {
            CStr::from_ptr(msg).to_string_lossy().into_owned()
        } else {
            "(no message)".to_string()
        }
    };
    anyhow::anyhow!("{what} failed with CUDA error {code}: {text}")
}

fn check(api: &Api, code: CuResult, what: &str) -> anyhow::Result<()> {
    // AIONS_GPU_TRACE=1 prints every driver call and its raw result. A wrongly bound
    // symbol returns plausible-looking garbage rather than failing loudly, so when
    // something goes wrong the only trustworthy evidence is the number each call
    // actually returned.
    if std::env::var("AIONS_GPU_TRACE").is_ok() {
        eprintln!("cuda-trace: {what} -> {code}");
    }
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(cuda_error(api, code, what))
    }
}
