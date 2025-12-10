#include "common.h"
#include "timer.h"
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Tunables / configuration
// -----------------------------------------------------------------------------
// We still target: 10k x 10k, ~32 nnz per row, RTX 4000 (Turing, 48KB shared / SM).
// One warp per block, 1024-entry hash per warp, like your fastest version.

#define BLOCK_SIZE       64
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)  // = 2
#define HASH_SIZE        2048                      // Must be power of 2
#define FULL_MASK        0xFFFFFFFFu
#define EMPTY_COL        0xFFFFFFFFu

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ unsigned int hashFunc4(unsigned int col) {
    // Knuth multiplicative hash, mask into [0, HASH_SIZE-1]
    return (col * 2654435761u) & (HASH_SIZE - 1);
}

// Warp-wide sum reduction using shuffles
__device__ __forceinline__ unsigned int warpReduceSum(unsigned int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// -----------------------------------------------------------------------------
// Kernel: dynamic row scheduling + hash-based accumulation
// -----------------------------------------------------------------------------
/**
 * Each warp repeatedly grabs the next available row index from a global counter
 * (nextRow_d) and computes that row of C = A * B.
 *
 * - One warp per block.
 * - Per-warp shared-memory hash table (hashCols, hashValues).
 * - Output format: COO (rowIdxs, colIdxs, values).
 */
__global__ void spmspm_kernel4(const CSRMatrix* __restrict__ csrMatrix1_d,
                               const CSRMatrix* __restrict__ csrMatrix2_d,
                               COOMatrix* __restrict__ cooMatrix_d,
                               unsigned int* __restrict__ nextRow_d,
                               unsigned int numRows) {

    const unsigned int laneId      = threadIdx.x & (WARP_SIZE - 1);   // 0..31
    const unsigned int localWarpId = threadIdx.x / WARP_SIZE;         // 0 for BLOCK_SIZE=32

    // Per-warp shared hash table
    __shared__ unsigned int hashCols[WARPS_PER_BLOCK][HASH_SIZE];
    __shared__ float        hashValues[WARPS_PER_BLOCK][HASH_SIZE];

    while (true) {
        // ---------------------------------------------------------------------
        // Get next row to process (dynamic scheduling)
        // ---------------------------------------------------------------------
        unsigned int row;
        if (laneId == 0) {
            row = atomicAdd(nextRow_d, 1u);
        }
        row = __shfl_sync(FULL_MASK, row, 0);

        if (row >= numRows) {
            // No more rows; this warp is done
            return;
        }

        // ---------------------------------------------------------------------
        // Initialize hash table cooperatively for this row
        // ---------------------------------------------------------------------
        for (unsigned int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
            hashCols[localWarpId][i]   = EMPTY_COL;
            hashValues[localWarpId][i] = 0.0f;
        }
        __syncwarp();

        // ---------------------------------------------------------------------
        // Row of A
        // ---------------------------------------------------------------------
        unsigned int rowStart1 = 0;
        unsigned int rowEnd1   = 0;
        if (laneId == 0) {
            rowStart1 = csrMatrix1_d->rowPtrs[row];
            rowEnd1   = csrMatrix1_d->rowPtrs[row + 1];
        }
        rowStart1 = __shfl_sync(FULL_MASK, rowStart1, 0);
        rowEnd1   = __shfl_sync(FULL_MASK, rowEnd1,   0);

        // ---------------------------------------------------------------------
        // For each nonzero in row of A, multiply with corresponding row of B
        // ---------------------------------------------------------------------
        for (unsigned int i = rowStart1; i < rowEnd1; ++i) {
            unsigned int col1   = 0;
            float        value1 = 0.0f;

            // Lane 0 loads A's entry, then broadcasts to warp
            if (laneId == 0) {
                col1   = __ldg(&csrMatrix1_d->colIdxs[i]);
                value1 = __ldg(&csrMatrix1_d->values[i]);
            }
            col1   = __shfl_sync(FULL_MASK, col1,   0);
            value1 = __shfl_sync(FULL_MASK, value1, 0);

            // Load row of B that corresponds to column 'col1' of A
            unsigned int row2      = 0;
            unsigned int rowStart2 = 0;
            unsigned int rowEnd2   = 0;
            if (laneId == 0) {
                row2      = col1;
                rowStart2 = csrMatrix2_d->rowPtrs[row2];
                rowEnd2   = csrMatrix2_d->rowPtrs[row2 + 1];
            }
            rowStart2 = __shfl_sync(FULL_MASK, rowStart2, 0);
            rowEnd2   = __shfl_sync(FULL_MASK, rowEnd2,   0);

            // Warp cooperatively processes B's row: strided across the row
            // for coalesced access.
            for (unsigned int j = rowStart2 + laneId; j < rowEnd2; j += WARP_SIZE) {
                unsigned int col2   = __ldg(&csrMatrix2_d->colIdxs[j]);
                float        value2 = __ldg(&csrMatrix2_d->values[j]);
                float        product = value1 * value2;

                // Hash-based insert with linear probing in shared memory
                unsigned int hashIdx = hashFunc4(col2);
                unsigned int probes  = 0;

                while (probes < HASH_SIZE) {
                    unsigned int existing = hashCols[localWarpId][hashIdx];

                    if (existing == col2) {
                        // Same column: accumulate
                        atomicAdd(&hashValues[localWarpId][hashIdx], product);
                        break;
                    } else if (existing == EMPTY_COL) {
                        // Try to claim this slot
                        unsigned int old = atomicCAS(&hashCols[localWarpId][hashIdx],
                                                     EMPTY_COL, col2);
                        if (old == EMPTY_COL || old == col2) {
                            atomicAdd(&hashValues[localWarpId][hashIdx], product);
                            break;
                        }
                    }
                    hashIdx = (hashIdx + 1) & (HASH_SIZE - 1);
                    ++probes;
                }
            }
            // NOTE: no __syncwarp() needed here; shared memory updates are protected
            // by atomics, and there is no cross-iteration dependency.
        }

        __syncwarp(); // Ensure all inserts for this row are done before we count/write

        // ---------------------------------------------------------------------
        // Count nonzeros in hash table (warp-wide)
        // ---------------------------------------------------------------------
        unsigned int localCount = 0;
        for (unsigned int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
            if (hashCols[localWarpId][i] != EMPTY_COL) {
                ++localCount;
            }
        }
        unsigned int rowNnz = warpReduceSum(localCount);

        // All lanes now know rowNnz
        if (rowNnz == 0) {
            // This row produces no entries; go grab the next row
            continue;
        }

        // ---------------------------------------------------------------------
        // Reserve space in global COO arrays
        // ---------------------------------------------------------------------
        unsigned int startIdx = 0;
        if (laneId == 0) {
            startIdx = atomicAdd(&cooMatrix_d->numNonzeros, rowNnz);
        }
        startIdx = __shfl_sync(FULL_MASK, startIdx, 0);

        // ---------------------------------------------------------------------
        // Warp-cooperative write-out of hash table to COO (coalesced-ish)
        // ---------------------------------------------------------------------
        unsigned int prefixBase = 0;
        for (unsigned int base = 0; base < HASH_SIZE; base += WARP_SIZE) {
            unsigned int idx = base + laneId;
            bool hasEntry = (idx < HASH_SIZE) &&
                            (hashCols[localWarpId][idx] != EMPTY_COL);

            // Mask of lanes with a valid entry in this segment
            unsigned int mask = __ballot_sync(FULL_MASK, hasEntry);
            unsigned int segCount = __popc(mask);

            // Rank of this lane among set bits in 'mask'
            unsigned int laneMask = mask & ((1u << laneId) - 1u);
            unsigned int rank     = __popc(laneMask);

            if (hasEntry) {
                unsigned int outPos = startIdx + prefixBase + rank;
                cooMatrix_d->rowIdxs[outPos] = row;
                cooMatrix_d->colIdxs[outPos] = hashCols[localWarpId][idx];
                cooMatrix_d->values[outPos]  = hashValues[localWarpId][idx];
            }

            // Advance the base offset by number of entries in this segment
            prefixBase += segCount;
        }

        // After finishing this row, the while(true) loop will grab another row.
    }
}

// -----------------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------------
/**
 * Dynamic-scheduling SpMSpM on GPU (hash-based, COO output).
 *
 * csrMatrix1, csrMatrix2: host pointers (for sizes)
 * csrMatrix1_d, csrMatrix2_d: device pointers
 * cooMatrix_d: device pointer
 */
void spmspm_gpu4(CSRMatrix* csrMatrix1,
                 CSRMatrix* csrMatrix2,
                 CSRMatrix* csrMatrix1_d,
                 CSRMatrix* csrMatrix2_d,
                 COOMatrix* cooMatrix_d) {

    unsigned int numRows = csrMatrix1->numRows;

    // Global row counter for dynamic scheduling
    unsigned int* d_nextRow = nullptr;
    cudaMalloc((void**)&d_nextRow, sizeof(unsigned int));
    cudaMemset(d_nextRow, 0, sizeof(unsigned int));

    // Decide how many warps to launch.
    // Simple choice: one warp per row (like before), but now with dynamic scheduling.
    // You can experiment with fewer or more warps here.
    unsigned int numWarpsToLaunch = numRows;
    unsigned int warpsPerBlock = WARPS_PER_BLOCK; // = 2 with BLOCK_SIZE=64
    unsigned int numBlocks =
        (numWarpsToLaunch + warpsPerBlock - 1) / warpsPerBlock;

    dim3 blockDim(BLOCK_SIZE, 1, 1);  // 64 threads = 1 warp per block
    dim3 gridDim(numBlocks, 1, 1);

    spmspm_kernel4<<<gridDim, blockDim>>>(
        csrMatrix1_d,
        csrMatrix2_d,
        cooMatrix_d,
        d_nextRow,
        numRows
    );

    cudaDeviceSynchronize();

    cudaFree(d_nextRow);
}

