// Tensor-core MMA issue rates on this box: the forms Sol-Attn's exact branch
// could use for its PV matmul.
//
// NOT a check and not part of any check. It needs `nvcc`, which nothing else
// in bench/ does, and it asserts nothing -- it prints five numbers. It is here
// because `docs/open_experiments.md` #17 quotes its output, and a number in a
// doc with no reproduction path is the failure this repo keeps naming.
//
//     nvcc -O3 -arch=sm_89 -o /tmp/mma_rate bench/mma_rate.cu && /tmp/mma_rate
//
// Set -arch to your own card. Needs a mostly idle GPU: it saturates every SM,
// so a concurrent render both slows this and is slowed by it.
//
// Why it exists. `sol_layout.cuh:81` justifies the all-INT8 exact branch with
// "sm_120 is issue-rate bound and f32-accumulate forms issue at half rate".
// That is upstream's claim about sm_120, and the cost of a 16-bit PV on OUR
// card followed from it by inference. Two things are measured instead:
// the int8-to-16-bit MAC-rate ratio, and whether the half-rate f32-accumulate
// claim holds on sm_89.
//
// Method. Every accumulator set is independent (ACC of them per thread) so the
// loop is issue-bound rather than latency-bound on the accumulator chain;
// operands stay register-resident so no memory path is involved. Reported in
// MACs/s, which is the only unit in which an int8 m16n8k32 (4096 MACs per
// warp-instruction) and a 16-bit m16n8k16 (2048) are comparable -- comparing
// instruction counts would flatter the 16-bit forms by exactly 2x.
//
// A cuBLAS GEMM is NOT a substitute here, and was tried first: at N=8192 on a
// 4090 `torch._int_mm` reached 141.8 TOPS against int8's ~660 TOPS peak while
// bf16 `matmul` sat at 163.9 TFLOPS, essentially its peak. That measures
// cuBLAS's int8 GEMM quality, not the tensor core, and would have inverted the
// conclusion.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

#define ACC 8
#define ITER 4096

#define BENCH_KERNEL(NAME, MMA_ASM, NA, NB, ND)                              \
__global__ void NAME(uint32_t* sink, uint32_t seed) {                        \
    uint32_t a[ACC][NA], b[ACC][NB];                                         \
    uint32_t d[ACC][ND];                                                     \
    _Pragma("unroll")                                                        \
    for (int s = 0; s < ACC; ++s) {                                          \
        _Pragma("unroll")                                                    \
        for (int i = 0; i < NA; ++i) a[s][i] = seed + s * 7 + i;             \
        _Pragma("unroll")                                                    \
        for (int i = 0; i < NB; ++i) b[s][i] = seed + s * 13 + i;            \
        _Pragma("unroll")                                                    \
        for (int i = 0; i < ND; ++i) d[s][i] = 0;                            \
    }                                                                        \
    for (int it = 0; it < ITER; ++it) {                                      \
        _Pragma("unroll")                                                    \
        for (int s = 0; s < ACC; ++s) { MMA_ASM }                            \
    }                                                                        \
    /* Consume d, but on a condition that never fires: the stores would       \
       otherwise be the thing being timed. */                                \
    uint32_t acc = 0;                                                        \
    for (int s = 0; s < ACC; ++s)                                            \
        for (int i = 0; i < ND; ++i) acc ^= d[s][i];                         \
    if (acc == 0xdeadbeefu) sink[0] = acc;                                   \
}

// int8 m16n8k32 -> s32: what the exact branch runs today for QK and for PV.
BENCH_KERNEL(k_s8, {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[s][0]), "+r"(d[s][1]), "+r"(d[s][2]), "+r"(d[s][3])
                 : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]),
                   "r"(b[s][0]), "r"(b[s][1]));
}, 4, 2, 4)

// u8 x s8 m16n8k32 -> s32: the PV form specifically (unsigned P, signed V).
BENCH_KERNEL(k_u8s8, {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[s][0]), "+r"(d[s][1]), "+r"(d[s][2]), "+r"(d[s][3])
                 : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]),
                   "r"(b[s][0]), "r"(b[s][1]));
}, 4, 2, 4)

// bf16 m16n8k16 -> f32: the drop-in 16-bit PV. sol_layout.cuh already wraps
// this as mma_bf16 and the per-row route kernel already runs it.
BENCH_KERNEL(k_bf16f32, {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(*(float*)&d[s][0]), "+f"(*(float*)&d[s][1]),
                   "+f"(*(float*)&d[s][2]), "+f"(*(float*)&d[s][3])
                 : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]),
                   "r"(b[s][0]), "r"(b[s][1]));
}, 4, 2, 4)

// fp16 m16n8k16 -> f32: the form sage's "fp16 (most accurate)" PV issues.
BENCH_KERNEL(k_f16f32, {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(*(float*)&d[s][0]), "+f"(*(float*)&d[s][1]),
                   "+f"(*(float*)&d[s][2]), "+f"(*(float*)&d[s][3])
                 : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]),
                   "r"(b[s][0]), "r"(b[s][1]));
}, 4, 2, 4)

// fp16 m16n8k16 -> f16: half-precision accumulate, 2 result registers. Cheap,
// but accumulating a whole routed key list in f16 is its own question -- this
// measures the instruction, not whether it is usable.
BENCH_KERNEL(k_f16f16, {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[s][0]), "+r"(d[s][1])
                 : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]),
                   "r"(b[s][0]), "r"(b[s][1]));
}, 4, 2, 2)

struct Case { const char* name; void (*fn)(uint32_t*, uint32_t); double macs; };

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int blocks = prop.multiProcessorCount * 4;
    const int threads = 128;                       // 4 warps
    const double warps = (double)blocks * threads / 32.0;

    uint32_t* sink;
    cudaMalloc(&sink, 4);

    // MACs per warp-instruction: m * n * k.
    Case cases[] = {
        {"s8   m16n8k32 -> s32", k_s8,      16.0 * 8 * 32},
        {"u8s8 m16n8k32 -> s32", k_u8s8,    16.0 * 8 * 32},
        {"bf16 m16n8k16 -> f32", k_bf16f32, 16.0 * 8 * 16},
        {"f16  m16n8k16 -> f32", k_f16f32,  16.0 * 8 * 16},
        {"f16  m16n8k16 -> f16", k_f16f16,  16.0 * 8 * 16},
    };

    printf("%s (sm_%d%d), %d SMs, %d blocks x %d thr\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           blocks, threads);
    printf("%-24s %10s %14s %12s\n", "form", "ms", "TMAC/s", "vs int8");

    double base = 0;
    for (int c = 0; c < 5; ++c) {
        cases[c].fn<<<blocks, threads>>>(sink, 0x01010101u);   // warmup
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int r = 0; r < 10; ++r) cases[c].fn<<<blocks, threads>>>(sink, 0x01010101u);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
        ms /= 10.0f;
        const double tmacs = warps * ITER * ACC * cases[c].macs / (ms * 1e-3) / 1e12;
        if (c == 0) base = tmacs;
        printf("%-24s %10.3f %14.1f %11.2fx\n", cases[c].name, ms, tmacs, tmacs / base);
    }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("\nCUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}
