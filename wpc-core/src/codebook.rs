use half::f16;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub const BLOCK_DIM: usize = 16;

#[derive(Clone)]
pub struct PatternDict {
    pub centroids: Vec<[f32; BLOCK_DIM]>, // 256
}

impl PatternDict {
    pub fn train(data: &[[f32; BLOCK_DIM]], k: usize, iters: usize) -> Self {
        let centroids = kmeans(data, k, iters, 42);
        Self { centroids }
    }

    pub fn nearest(&self, v: &[f32; BLOCK_DIM]) -> (u8, [f32; BLOCK_DIM]) {
        let mut best_dist = f32::MAX;
        let mut best_id = 0;
        
        for (i, c) in self.centroids.iter().enumerate() {
            let dist = sq_dist(v, c);
            if dist < best_dist {
                best_dist = dist;
                best_id = i;
            }
        }
        (best_id as u8, self.centroids[best_id])
    }
}

#[derive(Clone)]
pub struct ResidualDict {
    pub centroids_f16: Vec<[f16; BLOCK_DIM]>, // 65536
    pub centroids_f32: Vec<[f32; BLOCK_DIM]>,
}

impl ResidualDict {
    pub fn train(data: &[[f32; BLOCK_DIM]], k: usize, iters: usize) -> Self {
        // Subsample to max 8192 items to speed up k-means
        let max_samples = 8192;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let sampled: Vec<[f32; BLOCK_DIM]> = if data.len() > max_samples {
            let mut v = data.to_vec();
            v.shuffle(&mut rng);
            v.truncate(max_samples);
            v
        } else {
            data.to_vec()
        };

        let centroids = kmeans(&sampled, k, iters, 42);

        let mut centroids_f16 = Vec::with_capacity(centroids.len());
        let mut centroids_f32 = Vec::with_capacity(centroids.len());
        for c in centroids {
            let mut c16 = [f16::ZERO; BLOCK_DIM];
            let mut c32 = [0.0; BLOCK_DIM];
            for i in 0..BLOCK_DIM {
                c16[i] = f16::from_f32(c[i]);
                c32[i] = c16[i].to_f32(); // store exact representable value
            }
            centroids_f16.push(c16);
            centroids_f32.push(c32);
        }

        Self { centroids_f16, centroids_f32 }
    }

    pub fn nearest(&self, v: &[f32; BLOCK_DIM]) -> (u16, [f32; BLOCK_DIM]) {
        let mut best_dist = f32::MAX;
        let mut best_id = 0;
        let mut best_vec = [0.0; BLOCK_DIM];
        
        for (i, c) in self.centroids_f32.iter().enumerate() {
            let dist = sq_dist(v, c);
            if dist < best_dist {
                best_dist = dist;
                best_id = i;
                best_vec = *c;
            }
        }
        (best_id as u16, best_vec)
    }
}

// Dummy for first pass
pub fn dummy_residual_dict() -> ResidualDict {
    let centroids_f16 = vec![[f16::ZERO; BLOCK_DIM]; 1]; // Only 1 to save lookup time
    let centroids_f32 = vec![[0.0; BLOCK_DIM]; 1];
    ResidualDict { centroids_f16, centroids_f32 }
}

fn sq_dist(a: &[f32; BLOCK_DIM], b: &[f32; BLOCK_DIM]) -> f32 {
    let mut sum = 0.0;
    for i in 0..BLOCK_DIM {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

use rayon::prelude::*;

fn kmeans(data: &[[f32; BLOCK_DIM]], k: usize, max_iters: usize, seed: u64) -> Vec<[f32; BLOCK_DIM]> {
    let n = data.len();
    if n == 0 {
        return vec![[0.0; BLOCK_DIM]; k];
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut centroids = Vec::with_capacity(k);
    
    // Fast random initialization
    if n <= k {
        centroids.extend_from_slice(data);
        while centroids.len() < k {
            centroids.push(data[rng.gen_range(0..n)]);
        }
    } else {
        // Random subset
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        for &idx in indices.iter().take(k) {
            centroids.push(data[idx]);
        }
    }

    // Lloyd iterations
    let mut assignments = vec![0; n];
    for _ in 0..max_iters {
        // Parallel assignment step
        let new_assignments: Vec<usize> = data.par_iter().map(|p| {
            let (mut best_d, mut best_c) = (f32::MAX, 0);
            for (j, c) in centroids.iter().enumerate() {
                let d = sq_dist(p, c);
                if d < best_d { best_d = d; best_c = j; }
            }
            best_c
        }).collect();
        
        let mut changed = false;
        for i in 0..n {
            if assignments[i] != new_assignments[i] {
                assignments[i] = new_assignments[i];
                changed = true;
            }
        }
        if !changed { break; }
        
        // Update
        let mut counts = vec![0; k];
        let mut sums = vec![[0.0; BLOCK_DIM]; k];
        for (i, p) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..BLOCK_DIM { sums[c][j] += p[j]; }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..BLOCK_DIM {
                    centroids[j][d] = sums[j][d] / counts[j] as f32;
                }
            } else {
                // Perturb
                centroids[j] = data[rng.gen_range(0..n)];
            }
        }
    }
    
    centroids
}
