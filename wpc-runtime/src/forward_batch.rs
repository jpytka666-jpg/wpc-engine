use anyhow::Result;
use matrixmultiply;
use std::ffi::{c_int, c_void};
use std::ptr::null_mut;

const MADV_HUGEPAGE: c_int = 14;

extern "C" {
    fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> i32;
}

const PROT_READ: i32 = libc::PROT_READ;
const PROT_WRITE: i32 = libc::PROT_WRITE;
const MAP_PRIVATE: i32 = libc::MAP_PRIVATE;
const MAP_ANONYMOUS: i32 = libc::MAP_ANONYMOUS;

pub struct MmapF32 {
    ptr: *mut f32,
    len: usize,
    used: usize,
    bytes: usize,
}

impl MmapF32 {
    pub fn new(num_elems: usize) -> Result<Self> {
        let alloc_elems = num_elems.max(1);
        let bytes = alloc_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("size overflow"))?;

        unsafe {
            let p = mmap(
                null_mut(),
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                anyhow::bail!("mmap failed");
            }
            let _ = madvise(p, bytes, MADV_HUGEPAGE);
            Ok(Self {
                ptr: p as *mut f32,
                len: num_elems,
                used: 0,
                bytes,
            })
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn capacity(&self) -> usize {
        self.len
    }

    /// Mark the first `used` elements as initialized so growth can preserve them.
    pub fn mark_used(&mut self, used: usize) -> Result<()> {
        if used > self.len {
            anyhow::bail!("used exceeds capacity");
        }
        self.used = used;
        Ok(())
    }

    pub fn ensure_capacity(&mut self, need: usize) -> Result<()> {
        if need <= self.len {
            return Ok(());
        }

        let new_len = need
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("capacity overflow"))?;
        let old_used = self.used;
        let new_map = MmapF32::new(new_len)?;

        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_map.ptr, old_used);
            let old_ptr = self.ptr as *mut c_void;
            let old_bytes = self.bytes;
            self.ptr = new_map.ptr;
            self.len = new_map.len;
            self.bytes = new_map.bytes;
            self.used = old_used;
            let _ = munmap(old_ptr, old_bytes);
        }

        std::mem::forget(new_map);
        Ok(())
    }
}

impl Drop for MmapF32 {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr as *mut c_void, self.bytes);
        }
    }
}

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
        let mut new_keys = MmapF32::new(new_cap)?;
        let mut new_vals = MmapF32::new(new_cap)?;
        let copy_elems = self.seq_len * self.dim;
        unsafe {
            std::ptr::copy_nonoverlapping(self.keys.ptr, new_keys.ptr, copy_elems);
            std::ptr::copy_nonoverlapping(self.values.ptr, new_vals.ptr, copy_elems);
        }
        new_keys.used = copy_elems;
        new_vals.used = copy_elems;
        self.keys = new_keys;
        self.values = new_vals;
        Ok(())
    }

    pub fn append_batch(
        &mut self,
        k_batch: &[f32],
        v_batch: &[f32],
        batch_rows: usize,
    ) -> Result<()> {
        let elems = batch_rows
            .checked_mul(self.dim)
            .ok_or_else(|| anyhow::anyhow!("batch size overflow"))?;
        if k_batch.len() != elems || v_batch.len() != elems {
            anyhow::bail!("batch length does not match batch_rows * dim");
        }
        self.ensure_room(batch_rows)?;
        let dest_off = self.seq_len * self.dim;
        unsafe {
            std::ptr::copy_nonoverlapping(k_batch.as_ptr(), self.keys.ptr.add(dest_off), elems);
            std::ptr::copy_nonoverlapping(v_batch.as_ptr(), self.values.ptr.add(dest_off), elems);
        }
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
        self.keys.ptr
    }

    pub fn vals_ptr(&self) -> *const f32 {
        self.values.ptr
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

#[allow(clippy::too_many_arguments)]
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

    pub fn optimized_attention_batch(
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
        unsafe {
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
        }

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
        unsafe {
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
        }
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
        self.optimized_attention_batch(
            q_batch,
            kv.keys_ptr(),
            kv.vals_ptr(),
            batch_size,
            kv.seq_len,
            past_len,
        )
    }

    pub fn reference_attention_batch(
        &self,
        q_batch: &[Vec<f32>],
        kv: &KvLayer,
    ) -> Result<Vec<Vec<f32>>> {
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
