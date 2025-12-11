# GPU-Optimized Sparse Matrix–Sparse Matrix Multiplication (SpMSpM)

## 📄 [**Final Report**](https://drive.google.com/file/d/1DvDdbefz91X2rnek4_Tlkc19BMdgRaYB/view?usp=sharing) - Full analysis of kernel optimizations, performance benchmarks, and GPU architecture insights for SpMSpM.

## Project Overview

This project explores the optimization of Sparse Matrix–Sparse Matrix Multiplication (SpMSpM) on GPUs using CUDA. We implement and compare several GPU kernels incorporating warp-level parallelism, shared-memory hashing, and dynamic scheduling to achieve significant speedups over CPU and basic GPU baselines across multiple NVIDIA GPU architectures.

**Authors**: Huy Nguyen, Tanush Siotia, Jou Barzdukas 

**Advisor**: Prof. Adwait Jog

**Department of Computer Science, University of Virginia**

## Key Contributions

- **Performance Gains**: Achieved up to **30× speedup** over CPU baseline and **10× speedup** over basic GPU implementation on H100 GPUs.

- **Progressive Optimization**: Five kernels demonstrating incremental optimization strategies from basic GPU parallelism to advanced warp-level techniques.

- **Cross-Architecture Analysis**: Benchmarked across Quadro RTX 4000, Quadro RTX 6000, and NVIDIA H100 GPUs.

- **Design Tradeoffs**:
  - Warp-level parallelism improves memory throughput and parallelism.
  - Hash-based accumulation replaces O(k) linear search with O(1) average lookup.
  - Dynamic scheduling balances workload across warps for irregular matrices.

## Kernel Implementations

| Kernel | Description |
|--------|-------------|
| **Kernel 0** | Baseline GPU implementation — one thread per row |
| **Kernel 1** | Warp-level parallelism + shared memory accumulation |
| **Kernel 2** | Shared-memory hash table with O(1) lookup (Knuth hash) |
| **Kernel 3** | Work-adaptive hash tables with per-row sizing |
| **Kernel 4** | Fully optimized: dynamic scheduling, thread coarsening, `__ldg()` cache, warp primitives |

## Results Summary

Performance results on a 10,000×10,000 matrix with 32 nnz/row:

| GPU | CPU | Basic GPU | Kernel 1 | Kernel 2 | Kernel 3 | Kernel 4 |
|-----|-----|-----------|----------|----------|----------|----------|
| RTX 4000 | 102.51 ms | 58.32 ms | 90.89 ms | 19.21 ms | 34.24 ms | **8.77 ms** |
| RTX 6000 | 116.81 ms | 29.56 ms | 32.86 ms | 8.09 ms | 13.87 ms | **4.34 ms** |
| H100 | 85.85 ms | 9.03 ms | 5.75 ms | 3.53 ms | 3.90 ms | **2.79 ms** |

Kernel 4 achieves the best performance across all architectures, combining hash-based accumulation with dynamic row scheduling and warp-cooperative output.

## Tools & Hardware

- **Language**: CUDA C++
- **GPUs Tested**:
  - Quadro RTX 4000 (Turing)
  - Quadro RTX 6000 (Turing)
  - NVIDIA H100 96GB (Hopper)

## Build

```bash
make
```

This builds:
- `spmspm` — the main SpMSpM executable
- `data/gen_coo` — matrix generator tool

## Usage

### Generate test matrices

```bash
cd data
./gen_coo <N> <nnz_per_row> <output_file>
```

Example:
```bash
./gen_coo 10240 32 matrix0.txt
```

### Run SpMSpM

```bash
# Run specific kernel versions
./spmspm -f data/matrix0.txt -0      # Basic GPU (Kernel 0)
./spmspm -f data/matrix0.txt -1      # Kernel 1
./spmspm -f data/matrix0.txt -2      # Kernel 2
./spmspm -f data/matrix0.txt -3      # Kernel 3
./spmspm -f data/matrix0.txt -4      # Kernel 4

# Run multiple kernels
./spmspm -f data/matrix0.txt -0 -1 -2 -3 -4

# Enable exact verification (instead of quick verify)
./spmspm -f data/matrix0.txt -4 -v
```

## Clean

```bash
make clean
```

## References

1. Wang, Yizhuo, et al. "Optimizing General Sparse Matrix-Matrix Multiplication on the GPU." *ACM Transactions on Architecture and Code Optimization*, vol. 22, no. 4, Nov. 2025.

2. Dalton, Steven, Luke Olson, and Nathan Bell. "Optimizing Sparse Matrix—Matrix Multiplication for the GPU." *ACM Transactions on Mathematical Software*, vol. 41, no. 4, Oct. 2015.

## License

This project is licensed under the [MIT License](LICENSE).

---

*This work provides foundational insights into GPU-accelerated sparse matrix multiplication, with applications in scientific computing and machine learning.*
