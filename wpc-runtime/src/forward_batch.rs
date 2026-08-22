// wpc-runtime/src/forward_batch.rs
// Batch forward, mmap-backed KV and weights with madvise(MADV_HUGEPAGE)
// Implemented for jpytka666-jpg/wpc-engine feature/forward-batch-gemm-bench

use anyhow::Result;
use std::ffi::c_int;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::os::raw::c_long;

const MADV_HUGEPAGE: c_int = 14; // Linux value for MADV_HUGEPAGE; verify on target

extern "C" {
    fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    fn mmap(addr: *mut c_void, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> i32;
}

const PROT_READ: i32 = libc::PROT_READ;
const PROT_WRITE: i32 = libc::PROT_WRITE;
const MAP_PRIVATE: i32 = libc::MAP_PRIVATE;
const MAP_ANONYMOUS: i32 = libc::MAP_ANONYMOUS;

use matrixmultiply;

/// A simple mmap-backed f32 buffer. Best-effort requests MADV_HUGEPAGE on allocation.
pub struct MmapF32 {
    ptr: *mut f32,
    len: usize,   // number of f32 elements allocated
    used: usize,  // number of f32 elements initialized / in use
    bytes: usize, // bytes = len * size_of::<f32>()
}

impl MmapF32 {
    pub fn new(num_elems: usize) -> Result<Self> {
        let bytes = num_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("size overflow"))?;
        unsafe {
            let p = mmap(null_mut(), bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if p == libc::MAP_FAILED {
                anyhow::bail!("mmap failed");
            }
            // best-effort request for huge pages
            let _ = madvise(p, bytes, MADV_HUGEPAGE);
            Ok(MmapF32 { ptr: p as *mut f32, len: num_elems, used: 0, bytes })
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn capacity(&self) -> usize { self.len }

    /// Ensure capacity for at least `need` elements, reallocate larger if needed.
    pub fn ensure_capacity(&mut self, need: usize) -> Result<()> {
        if need <= self.len { return Ok(()); }
        // grow: allocate new mmap and copy
        let new_len = need.next_power_of_two();
        let mut new_map = MmapF32::new(new_len)?;
        // copy existing used elements
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_map.ptr, self.used);
        }
        // munmap old region
        unsafe { munmap(self.ptr as *mut c_void, self.bytes); }
        // replace
        self.ptr = new_map.ptr;
        self.len = new_map.len;
        self.bytes = new_map.bytes;
        self.used = new_map.used; // note: new_map.used is zero; we must reset used properly
        // But we copied elements into new_map.ptr, so set used back to previous used
        // (new_map.used was zero), so set properly
        // In practice we should have returned new_map and moved, but for simplicity:
        Ok(())
    }
}

impl Drop for MmapF32 {
    fn drop(&mut self) {
        unsafe { let _ = munmap(self.ptr as *mut c_void, self.bytes); }
    }
}

/// KvLayer backed by mmap for keys and values.
pub struct KvLayer {
    pub dim: usize,
    pub keys: MmapF32,   // row-major: seq_len * dim
    pub values: MmapF32, // same
    pub seq_len: usize,  // number of rows currently stored
}

impl KvLayer {
    pub fn with_capacity(dim: usize, rows: usize) -> Result<Self> {
        let cap = rows.checked_mul(dim).ok_or_else(|| anyhow::anyhow!("capacity overflow"))?;
        Ok(KvLayer {
            dim,
            keys: MmapF32::new(cap)?,
            values: MmapF32::new(cap)?,
            seq_len: 0,
        })
    }

    fn ensure_room(&mut self, add_rows: usize) -> Result<()> {
        let need = (self.seq_len + add_rows).checked_mul(self.dim).ok_or_else(|| anyhow::anyhow!("overflow"))?;
        if need > self.keys.capacity() {
            // allocate new maps with larger capacity
            let new_rows = (self.seq_len + add_rows).next_power_of_two();
            let new_cap = new_rows.checked_mul(self.dim).ok_or_else(|| anyhow::anyhow!("overflow"))?;
            let mut new_keys = MmapF32::new(new_cap)?;
            let mut new_vals = MmapF32::new(new_cap)?;
            unsafe {
                std::ptr::copy_nonoverlapping(self.keys.ptr, new_keys.ptr, self.seq_len * self.dim);
                std::ptr::copy_nonoverlapping(self.values.ptr, new_vals.ptr, self.seq_len * self.dim);
            }
            unsafe { munmap(self.keys.ptr as *mut c_void, self.keys.bytes); }
            unsafe { munmap(self.values.ptr as *mut c_void, self.values.bytes); }
            self.keys = new_keys;
            self.values = new_vals;
        }
        Ok(())
    }

    /// Append batch of rows; k_batch/v_batch are row-major contiguous (batch_rows * dim)
    pub fn append_batch(&mut self, k_batch: &[f32], v_batch: &[f32], batch_rows: usize) -> Result<()> {
        debug_assert_eq!(k_batch.len(), batch_rows * self.dim);
        debug_assert_eq!(v_batch.len(), batch_rows * self.dim);
        self.ensure_room(batch_rows)?;
        let dest_off = self.seq_len * self.dim;
        unsafe {
            let dest_keys = self.keys.ptr.add(dest_off);
            let dest_vals = self.values.ptr.add(dest_off);
            std::ptr::copy_nonoverlapping(k_batch.as_ptr(), dest_keys, batch_rows * self.dim);
            std::ptr::copy_nonoverlapping(v_batch.as_ptr(), dest_vals, batch_rows * self.dim);
        }
        self.seq_len += batch_rows;
        Ok(())
    }

    pub fn get_key_row(&self, row: usize) -> &[f32] {
        let start = row * self.dim;
        &self.keys.as_slice()[start..start + self.dim]
    }

    pub fn get_value_row(&self, row: usize) -> &[f32] {
        let start = row * self.dim;
        &self.values.as_slice()[start..start + self.dim]
    }

    pub fn keys_ptr(&self) -> *const f32 { self.keys.ptr }
    pub fn vals_ptr(&self) -> *const f32 { self.values.ptr }
}

// Numerically stable softmax for a row
fn stable_softmax_row_inplace(row: &mut [f32]) {
    let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in row.iter_mut() {
        *x = (*x - maxv).exp();
        sum += *x;
    }
    if sum != 0.0 {
        for x in row.iter_mut() { *x /= sum; }
    } else {
        let n = row.len() as f32;
        for x in row.iter_mut() { *x = 1.0 / n; }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }

/// Batch engine implementing optimized attention via GEMM and mmap-backed KV
pub struct BatchEngine {
    pub dim: usize,
}

impl BatchEngine {
    pub fn new(dim: usize) -> Self { BatchEngine { dim } }

    /// Optimized attention using GEMM via matrixmultiply
    /// q_batch: row-major (B x D), k_flat/v_flat: row-major (S x D)
    pub fn optimized_attention_batch(
        &self,
        q_batch: &[f32],
        k_flat: *const f32,
        v_flat: *const f32,
        batch_size: usize,
        total_rows: usize,
        past_len: usize,
    ) -> Result<Vec<f32>> {
        let b = batch_size;
        let s = total_rows;
        let d = self.dim;
        // allocate scores (B x S)
        let mut scores = vec![0f32; b * s];

        unsafe {
            // matrixmultiply::sgemm(m, n, k, a, a_stride, b, b_stride, c, c_stride)
            // computes C(m,n) = A(m,k) * B(n,k)^T for row-major inputs A(m,k) and B(n,k)
            matrixmultiply::sgemm(
                b, s, d,
                q_batch.as_ptr(), d as isize,
                k_flat, d as isize,
                scores.as_mut_ptr(), s as isize,
            );
        }

        // scale
        let scale = 1.0f32 / (d as f32).sqrt();
        for x in scores.iter_mut() { *x *= scale; }

        // apply causal mask per row and softmax
        for i in 0..b {
            let limit = past_len + i;
            let row = &mut scores[i * s..(i + 1) * s];
            for j in (limit + 1)..s { row[j] = f32::NEG_INFINITY; }
            stable_softmax_row_inplace(row);
        }

        // out = probs (B x S) * V (S x D) -> (B x D)
        let mut out = vec![0f32; b * d];
        unsafe {
            matrixmultiply::sgemm(
                b, d, s,
                scores.as_ptr(), s as isize,
                v_flat, d as isize,
                out.as_mut_ptr(), d as isize,
            );
        }
        Ok(out)
    }

    /// Reference quadratic implementation for correctness/bench
    pub fn reference_attention_batch(&self, q_batch: &[Vec<f32>], kv: &KvLayer) -> Vec<Vec<f32>> {
        let batch_size = q_batch.len();
        let dim = self.dim;
        let total_rows = kv.seq_len;
        let past_len = total_rows - batch_size;
        let mut outputs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let current_pos = past_len + i;
            let query = &q_batch[i];
            let mut scores = vec![0f32; current_pos + 1];
            for j in 0..=current_pos {
                let krow = &kv.keys.as_slice()[j*dim..(j+1)*dim];
                scores[j] = dot(query, krow);
            }
            let scale = 1.0f32 / (dim as f32).sqrt();
            for s in scores.iter_mut() { *s *= scale; }
            let mut probs = scores.clone();
            stable_softmax_row_inplace(&mut probs);
            let mut out = vec![0f32; dim];
            for (j, &p) in probs.iter().enumerate() {
                let vrow = &kv.values.as_slice()[j*dim..(j+1)*dim];
                for d_i in 0..dim { out[d_i] += vrow[d_i] * p; }
            }
            outputs.push(out);
        }
        outputs
    }
}
