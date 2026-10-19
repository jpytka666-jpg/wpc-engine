// darkstar-header-v1
// AUTHOR: Klaudia (Claude Code, Consigliere) for Marcin Szul (owner)
// AI MODEL / AGENT: Claude Opus 5.5 (claude-opus-5-5) in Claude Code on ZBook
// TIMESTAMP: 2026-10-18T19:00+01:00 (created)
// REASON: owner "run Noworodek's training on M60A" — CUDA kernels for the Noworodek CBMS decoder (same maths as wpc-engine train-cbms / transformer_backprop.rs)
// MECHANICS: compiled at runtime by NVRTC for compute_52 (Tesla M60); one window = S tokens, rows are window-major (b*S+i); f32 everywhere
// SYSTEM PART: AIONS Noworodek / Mikernel Codeone — GPU trainer
// ARCHITECTURE FUNCTION: elementwise / per-row / per-window kernels; all matrix products go through cuBLAS in main.rs
// DEPENDENCIES / LINKS: src/main.rs (noworodek-gpu); reference CPU code github jpytka666-jpg/wpc-engine noworodek-cbms-training
// TECH STACK: CUDA C (NVRTC), Rust cudarc
// LOCAL WORKSPACE: ~/noworodek/gpu-trainer
// GIT / GITHUB: PENDING
// REVISIONS / EVIDENCE: PENDING
extern "C" __global__ void gather_rows(const float* emb, const unsigned int* ids, float* out, int rows, int H) {
  int r = blockIdx.x, d = threadIdx.x; if (r < rows && d < H) out[r * H + d] = emb[(size_t)ids[r] * H + d];
}
// RMSNorm forward, one block per row, blockDim = H (128). inv[r] saved for backward.
extern "C" __global__ void rms_fwd(const float* x, const float* w, float* y, float* inv, int rows, int H, float eps) {
  __shared__ float s[256]; int r = blockIdx.x, d = threadIdx.x; float v = x[r * H + d]; s[d] = v * v; __syncthreads();
  for (int o = H / 2; o > 0; o >>= 1) { if (d < o) s[d] += s[d + o]; __syncthreads(); }
  float iv = rsqrtf(s[0] / H + eps); if (d == 0) inv[r] = iv; y[r * H + d] = v * iv * w[d];
}
// RMSNorm backward (reference norm_bwd): dx = w*inv*(dy - x*coeff), coeff = inv^2 * sum(dy*w*x)/H; dw += dy*x*inv (atomic)
extern "C" __global__ void rms_bwd(const float* x, const float* w, const float* inv, const float* dy, float* dx, float* dw, int rows, int H) {
  __shared__ float s[256]; int r = blockIdx.x, d = threadIdx.x; float xv = x[r * H + d], g = dy[r * H + d], iv = inv[r];
  s[d] = g * w[d] * xv; __syncthreads(); for (int o = H / 2; o > 0; o >>= 1) { if (d < o) s[d] += s[d + o]; __syncthreads(); }
  float coeff = iv * iv * s[0] / H; dx[r * H + d] = w[d] * iv * (g - xv * coeff); atomicAdd(&dw[d], g * xv * iv);
}
// causal single-head attention for one window per block; P saved [win][S][S]
extern "C" __global__ void attn_fwd(const float* q, const float* k, const float* v, float* out, float* P, int S, int H) {
  int w = blockIdx.x, t = threadIdx.x; const float* Q = q + w * S * H; const float* K = k + w * S * H; const float* V = v + w * S * H;
  __shared__ float sc[16 * 16]; float scale = sqrtf((float)H);
  if (t < S * S) { int i = t / S, j = t % S; float dot = 0.f; if (j <= i) for (int d = 0; d < H; d++) dot += Q[i * H + d] * K[j * H + d]; sc[t] = dot / scale; }
  __syncthreads();
  if (t < S) { int i = t; float m = -1e30f; for (int j = 0; j <= i; j++) m = fmaxf(m, sc[i * S + j]); float z = 0.f;
    for (int j = 0; j <= i; j++) { float e = expf(sc[i * S + j] - m); sc[i * S + j] = e; z += e; }
    for (int j = 0; j < S; j++) sc[i * S + j] = j <= i ? sc[i * S + j] / fmaxf(z, 1.17549435e-38f) : 0.f; }
  __syncthreads();
  if (t < S * S) P[w * S * S + t] = sc[t];
  for (int d = t; d < H; d += blockDim.x) for (int i = 0; i < S; i++) { float o = 0.f; for (int j = 0; j <= i; j++) o += sc[i * S + j] * V[j * H + d]; out[w * S * H + i * H + d] = o; }
}
extern "C" __global__ void attn_bwd(const float* q, const float* k, const float* v, const float* P, const float* dy, float* dq, float* dk, float* dv, int S, int H) {
  int w = blockIdx.x, t = threadIdx.x; size_t o = (size_t)w * S * H; const float* p = P + w * S * S; float scale = sqrtf((float)H);
  __shared__ float dp[16 * 16]; __shared__ float ds[16 * 16];
  if (t < S * S) { int i = t / S, j = t % S; float a = 0.f; if (j <= i) for (int d = 0; d < H; d++) a += dy[o + i * H + d] * v[o + j * H + d]; dp[t] = a; }
  __syncthreads();
  if (t < S) { int i = t; float dot = 0.f; for (int j = 0; j <= i; j++) dot += p[i * S + j] * dp[i * S + j];
    for (int j = 0; j < S; j++) ds[i * S + j] = j <= i ? p[i * S + j] * (dp[i * S + j] - dot) : 0.f; }
  __syncthreads();
  for (int d = t; d < H; d += blockDim.x) for (int i = 0; i < S; i++) {
    float aq = 0.f, ak = 0.f, av = 0.f;
    for (int j = 0; j < S; j++) { aq += ds[i * S + j] * k[o + j * H + d]; ak += ds[j * S + i] * q[o + j * H + d]; av += p[j * S + i] * dy[o + j * H + d]; }
    dq[o + i * H + d] = aq / scale; dk[o + i * H + d] = ak / scale; dv[o + i * H + d] = av; }
}
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
extern "C" __global__ void add3(const float* a, const float* b, const float* c, float* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] + b[i] + c[i]; }
extern "C" __global__ void silu_mul(const float* g, const float* u, float* sg, float* m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float x = g[i]; float s = x / (1.f + expf(-x)); sg[i] = s; m[i] = s * u[i]; } }
extern "C" __global__ void silu_mul_bwd(const float* dm, const float* g, const float* u, const float* sg, float* dg, float* du, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float x = g[i]; float s = 1.f / (1.f + expf(-x)); dg[i] = dm[i] * u[i] * (s + x * s * (1.f - s)); du[i] = dm[i] * sg[i]; } }
// rows S-1 of every window → compact [B,H]
extern "C" __global__ void take_last(const float* y, float* out, int S, int H) { int b = blockIdx.x, d = threadIdx.x; out[b * H + d] = y[((size_t)b * S + S - 1) * H + d]; }
extern "C" __global__ void put_last(const float* src, float* dy, int S, int H) { int b = blockIdx.x, d = threadIdx.x; dy[((size_t)b * S + S - 1) * H + d] = src[b * H + d]; }
// softmax cross-entropy on one logits row per block; loss[b]; if grad != 0: logits row := (softmax - onehot) * gscale
extern "C" __global__ void ce_rows(float* logits, const unsigned int* tgt, float* loss, int V, float gscale, int grad) {
  __shared__ float s[256]; int b = blockIdx.x, t = threadIdx.x; float* row = logits + (size_t)b * V;
  float m = -1e30f; for (int j = t; j < V; j += blockDim.x) m = fmaxf(m, row[j]); s[t] = m; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (t < o) s[t] = fmaxf(s[t], s[t + o]); __syncthreads(); } m = s[0]; __syncthreads();
  float z = 0.f; for (int j = t; j < V; j += blockDim.x) z += expf(row[j] - m); s[t] = z; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (t < o) s[t] += s[t + o]; __syncthreads(); } z = fmaxf(s[0], 1.17549435e-38f);
  unsigned int y = tgt[b]; if (t == 0) loss[b] = -logf(fmaxf(expf(row[y] - m) / z, 1e-12f));
  __syncthreads();
  if (grad) for (int j = t; j < V; j += blockDim.x) row[j] = (expf(row[j] - m) / z - (j == (int)y ? 1.f : 0.f)) * gscale;
}
// SGD: w -= lr * g
extern "C" __global__ void sgd(float* w, const float* g, float lr, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) w[i] -= lr * g[i]; }
// sparse embedding update: emb[ids[r]] -= lr * dx[r]
extern "C" __global__ void emb_sgd(float* emb, const unsigned int* ids, const float* dx, float lr, int rows, int H) {
  int r = blockIdx.x, d = threadIdx.x; if (r < rows && d < H) atomicAdd(&emb[(size_t)ids[r] * H + d], -lr * dx[r * H + d]); }
extern "C" __global__ void zero(float* x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = 0.f; }
