use anyhow::Result;
use matrixmultiply;

/// Ask the kernel to back this region with huge pages.
///
/// Only Linux understands the request, and even there it is a hint the kernel may
/// decline. It is worth making because the scratch areas here run to tens of millions of
/// floats, and fewer, larger pages mean fewer address translations during a forward pass.
/// Failure is ignored on purpose: a declined hint costs nothing and there is no sensible
/// way for a caller to act on it.
#[cfg(target_os = "linux")]
fn hint_huge_pages(data: &mut [f32]) {
    const MADV_HUGEPAGE: std::ffi::c_int = 14;
    extern "C" {
        fn madvise(
            addr: *mut std::ffi::c_void,
            length: usize,
            advice: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
    if data.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(data);
    unsafe {
        let _ = madvise(data.as_mut_ptr() as *mut std::ffi::c_void, bytes, MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
fn hint_huge_pages(_data: &mut [f32]) {}

/// A growable scratch buffer of f32.
///
/// This used to call mmap and munmap directly, which meant the engine could only be built
/// on Unix: `libc::MAP_PRIVATE` and its neighbours do not exist elsewhere, so the crate
/// failed to compile on Windows and took every crate depending on it down with it.
///
/// The allocation is now an ordinary Vec, which is portable and, at these sizes, ends up
/// served by the same kernel machinery regardless: an allocator hands out tens of
/// megabytes by mapping pages, not by carving up a heap. The huge-page hint the original
/// was really after is kept, applied to the Vec's own memory.
///
/// The name stays. It is used throughout this file, and renaming it would be a different
/// kind of change mixed into a port.
pub struct MmapF32 {
    data: Vec<f32>,
    used: usize,
}

impl MmapF32 {
    pub fn new(num_elems: usize) -> Result<Self> {
        num_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("size overflow"))?;
        let mut data = vec![0.0f32; num_elems];
        hint_huge_pages(&mut data);
        Ok(Self { data, used: 0 })
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    pub fn mark_used(&mut self, used: usize) -> Result<()> {
        if used > self.data.len() {
            anyhow::bail!("used exceeds capacity");
        }
        self.used = used;
        Ok(())
    }

    pub fn ensure_capacity(&mut self, need: usize) -> Result<()> {
        if need <= self.data.len() {
            return Ok(());
        }
        let new_len = need
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("capacity overflow"))?;
        // resize keeps everything already written, which is a superset of the `used`
        // prefix the hand-rolled version copied across by hand.
        self.data.resize(new_len, 0.0);
        hint_huge_pages(&mut self.data);
        Ok(())
    }
}

// No Drop impl: the Vec frees its own memory. The hand-written one called munmap on a
// pointer this struct no longer owns.

pub struct KvLayer {
    pub dim: usize,
    pub keys: MmapF32,
    pub values: MmapF32,
    pub seq_len: usize,
}

impl KvLayer {
    pub fn with_capacity(dim: usize, rows: usize) -> Result<Self> {
        let cap = rows
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("capacity overflow"))?;
        Ok(Self {
            dim,
            keys: MmapF32::new(cap)?,
            values: MmapF32::new(cap)?,
            seq_len: 0,
        })
    }

    fn ensure_room(&mut self, add_rows: usize) -> Result<()> {
        let total_rows = self
            .seq_len
            .checked_add(add_rows)
            .ok_or_else(|| anyhow::anyhow!("row count overflow"))?;
        let need = total_rows
            .checked_mul(self.dim)
            .ok_or_else(|| anyhow::anyhow!("capacity overflow"))?;
        if need <= self.keys.capacity() {
            return Ok(());
        }
        let new_rows = total_rows
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("row capacity overflow"))?;
        let new_cap = new_rows
            .checked_mul(self.dim)
            .ok_or_else(|| anyhow::anyhow!("capacity overflow"))?;
        let copy_elems = self.seq_len * self.dim;
        // Growing in place: resize keeps the prefix, so the manual copy between two fresh
        // allocations is no longer needed. Same result, no raw pointers, and it works on
        // every platform.
        self.keys.ensure_capacity(new_cap)?;
        self.values.ensure_capacity(new_cap)?;
        self.keys.mark_used(copy_elems)?;
        self.values.mark_used(copy_elems)?;
        Ok(())
    }

    pub fn append_batch(&mut self, k_batch: &[f32], v_batch: &[f32], batch_rows: usize) -> Result<()> {
        let elems = batch_rows
            .checked_mul(self.dim)
            .ok_or_else(|| anyhow::anyhow!("batch size overflow"))?;
        if k_batch.len() != elems || v_batch.len() != elems {
            anyhow::bail!("batch length does not match batch_rows * dim");
        }
        self.ensure_room(batch_rows)?;
        let dest_off = self.seq_len * self.dim;
        // Slice copy rather than raw pointers: same instruction sequence after
        // optimisation, but the bounds are checked and it compiles everywhere.
        self.keys.as_mut_slice()[dest_off..dest_off + elems].copy_from_slice(k_batch);
        self.values.as_mut_slice()[dest_off..dest_off + elems].copy_from_slice(v_batch);
        self.seq_len += batch_rows;
        self.keys.used = dest_off + elems;
        self.values.used = dest_off + elems;
        Ok(())
    }

    pub fn get_key_row(&self, row: usize) -> Result<&[f32]> {
        if row >= self.seq_len {
            anyhow::bail!("key row out of bounds");
        }
        let start = row * self.dim;
        Ok(&self.keys.as_slice()[start..start + self.dim])
    }

    pub fn get_value_row(&self, row: usize) -> Result<&[f32]> {
        if row >= self.seq_len {
            anyhow::bail!("value row out of bounds");
        }
        let start = row * self.dim;
        Ok(&self.values.as_slice()[start..start + self.dim])
    }

    pub fn keys_ptr(&self) -> *const f32 {
        self.keys.as_slice().as_ptr()
    }

    pub fn vals_ptr(&self) -> *const f32 {
        self.values.as_slice().as_ptr()
    }
}

fn stable_softmax_row_inplace(row: &mut [f32]) {
    let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in row.iter_mut() {
        *x = (*x - maxv).exp();
        sum += *x;
    }
    if sum > 0.0 && sum.is_finite() {
        for x in row.iter_mut() {
            *x /= sum;
        }
    } else {
        let n = row.len() as f32;
        for x in row.iter_mut() {
            *x = 1.0 / n;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

unsafe fn sgemm_row_major(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    matrixmultiply::sgemm(m, k, n, 1.0, a, rsa, csa, b, rsb, csb, 0.0, c, rsc, csc);
}

pub struct BatchEngine {
    pub dim: usize,
}

impl BatchEngine {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// # Safety
    /// `k_flat` and `v_flat` must each point to at least `total_rows * dim`
    /// readable `f32` values. The pointed-to memory must remain valid for the
    /// duration of this call and may not alias `q_batch` or the returned output.
    pub unsafe fn optimized_attention_batch(
        &self,
        q_batch: &[f32],
        k_flat: *const f32,
        v_flat: *const f32,
        batch_size: usize,
        total_rows: usize,
        past_len: usize,
    ) -> Result<Vec<f32>> {
        if k_flat.is_null() || v_flat.is_null() {
            anyhow::bail!("KV pointers must not be null");
        }
        let b = batch_size;
        let s = total_rows;
        let d = self.dim;
        let expected_q = b
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("query size overflow"))?;
        if q_batch.len() != expected_q {
            anyhow::bail!("q_batch length does not match batch_size * dim");
        }
        let visible_end = past_len
            .checked_add(b)
            .ok_or_else(|| anyhow::anyhow!("position overflow"))?;
        if visible_end > s {
            anyhow::bail!("past_len + batch_size exceeds total_rows");
        }
        if b == 0 || s == 0 || d == 0 {
            return Ok(vec![0.0; b * d]);
        }

        let mut scores = vec![0.0f32; b * s];
        sgemm_row_major(
            b,
            d,
            s,
            q_batch.as_ptr(),
            d as isize,
            1,
            k_flat,
            1,
            d as isize,
            scores.as_mut_ptr(),
            s as isize,
            1,
        );

        let scale = 1.0f32 / (d as f32).sqrt();
        for x in scores.iter_mut() {
            *x *= scale;
        }

        for i in 0..b {
            let limit = past_len + i;
            let row = &mut scores[i * s..(i + 1) * s];
            for x in &mut row[(limit + 1)..] {
                *x = f32::NEG_INFINITY;
            }
            stable_softmax_row_inplace(row);
        }

        let mut out = vec![0.0f32; b * d];
        sgemm_row_major(
            b,
            s,
            d,
            scores.as_ptr(),
            s as isize,
            1,
            v_flat,
            d as isize,
            1,
            out.as_mut_ptr(),
            d as isize,
            1,
        );
        Ok(out)
    }

    pub fn optimized_attention_from_kv(
        &self,
        q_batch: &[f32],
        kv: &KvLayer,
        batch_size: usize,
        past_len: usize,
    ) -> Result<Vec<f32>> {
        if kv.dim != self.dim {
            anyhow::bail!("KV dimension does not match engine dimension");
        }
        if past_len
            .checked_add(batch_size)
            .ok_or_else(|| anyhow::anyhow!("position overflow"))?
            != kv.seq_len
        {
            anyhow::bail!("past_len + batch_size must equal KV sequence length");
        }
        unsafe {
            self.optimized_attention_batch(
                q_batch,
                kv.keys_ptr(),
                kv.vals_ptr(),
                batch_size,
                kv.seq_len,
                past_len,
            )
        }
    }

    pub fn reference_attention_batch(&self, q_batch: &[Vec<f32>], kv: &KvLayer) -> Result<Vec<Vec<f32>>> {
        let batch_size = q_batch.len();
        if batch_size > kv.seq_len {
            anyhow::bail!("batch larger than KV sequence");
        }
        let past_len = kv.seq_len - batch_size;
        let dim = self.dim;
        let mut outputs = Vec::with_capacity(batch_size);

        for (i, query) in q_batch.iter().enumerate() {
            if query.len() != dim {
                anyhow::bail!("query dimension mismatch");
            }
            let current_pos = past_len + i;
            let mut scores = Vec::with_capacity(current_pos + 1);
            for j in 0..=current_pos {
                scores.push(dot(kv.get_key_row(j)?, query));
            }
            let scale = 1.0f32 / (dim as f32).sqrt();
            for x in scores.iter_mut() {
                *x *= scale;
            }
            stable_softmax_row_inplace(&mut scores);

            let mut out = vec![0.0f32; dim];
            for (j, p) in scores.iter().copied().enumerate() {
                let vrow = kv.get_value_row(j)?;
                for d_i in 0..dim {
                    out[d_i] += vrow[d_i] * p;
                }
            }
            outputs.push(out);
        }
        Ok(outputs)
    }
}
