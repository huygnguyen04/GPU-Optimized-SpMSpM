#include "common.h"
#include "timer.h"

// Kernel 2: Hash-Based Lookup Optimization
// Built on Kernel 1
// 
// OPTIMIZATION: HASH TABLE for O(1) average lookup instead of O(n) linear search
// - kernel1: Linear search through shared memory array
// - kernel2: Hash table with linear probing in shared memory

// =============================================================================
// GPU-SPECIFIC PARAMETERS - Uncomment the section for your target GPU
// =============================================================================

// --- RTX 6000 / RTX 4090 / V100 / T4 (48KB shared memory per block) ---
// Shared memory: 4 warps * 1024 entries * 8 bytes = 32KB < 48KB
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)  // = 4
#define HASH_SIZE 1024  // Power of 2

__device__ __forceinline__ unsigned int hashFunc2(unsigned int col) {
    return (col * 2654435761u) & (HASH_SIZE - 1);
}

__global__ void spmspm_kernel2(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d,
                               unsigned int* outputColsPool, float* outputValuesPool) {
    unsigned int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE;
    unsigned int localWarpId = threadIdx.x / WARP_SIZE;
    unsigned int row = warpId;
    
    // Shared memory hash table per warp
    __shared__ unsigned int hashCols[WARPS_PER_BLOCK][HASH_SIZE];
    __shared__ float hashValues[WARPS_PER_BLOCK][HASH_SIZE];
    
    if (row >= csrMatrix1_d->numRows) return;
    
    // Initialize hash table cooperatively
    for (unsigned int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
        hashCols[localWarpId][i] = 0xFFFFFFFF;
        hashValues[localWarpId][i] = 0.0f;
    }
    __syncwarp();
    
    unsigned int rowStart1 = csrMatrix1_d->rowPtrs[row];
    unsigned int rowEnd1 = csrMatrix1_d->rowPtrs[row + 1];
    
    for (unsigned int i = rowStart1; i < rowEnd1; ++i) {
        unsigned int col1 = csrMatrix1_d->colIdxs[i];
        float value1 = csrMatrix1_d->values[i];
        
        unsigned int row2 = col1;
        unsigned int rowStart2 = csrMatrix2_d->rowPtrs[row2];
        unsigned int rowEnd2 = csrMatrix2_d->rowPtrs[row2 + 1];
        
        // Warp cooperatively processes B's row
        for (unsigned int j = rowStart2 + laneId; j < rowEnd2; j += WARP_SIZE) {
            unsigned int col2 = csrMatrix2_d->colIdxs[j];
            float value2 = csrMatrix2_d->values[j];
            float product = value1 * value2;
            
            // Hash-based insert with linear probing
            unsigned int hashIdx = hashFunc2(col2);
            unsigned int probes = 0;
            
            while (probes < HASH_SIZE) {
                unsigned int existing = hashCols[localWarpId][hashIdx];
                
                if (existing == col2) {
                    atomicAdd(&hashValues[localWarpId][hashIdx], product);
                    break;
                }
                else if (existing == 0xFFFFFFFF) {
                    unsigned int old = atomicCAS(&hashCols[localWarpId][hashIdx], 0xFFFFFFFF, col2);
                    if (old == 0xFFFFFFFF || old == col2) {
                        atomicAdd(&hashValues[localWarpId][hashIdx], product);
                        break;
                    }
                }
                
                hashIdx = (hashIdx + 1) & (HASH_SIZE - 1);
                probes++;
            }
        }
        __syncwarp();
    }
    
    __syncwarp();
    
    // Count and write results (lane 0 only)
    if (laneId == 0) {
        unsigned int count = 0;
        for (unsigned int i = 0; i < HASH_SIZE; ++i) {
            if (hashCols[localWarpId][i] != 0xFFFFFFFF) count++;
        }
        
        if (count > 0) {
            unsigned int startIdx = atomicAdd(&cooMatrix_d->numNonzeros, count);
            unsigned int writeIdx = 0;
            
            for (unsigned int i = 0; i < HASH_SIZE; ++i) {
                if (hashCols[localWarpId][i] != 0xFFFFFFFF) {
                    cooMatrix_d->rowIdxs[startIdx + writeIdx] = row;
                    cooMatrix_d->colIdxs[startIdx + writeIdx] = hashCols[localWarpId][i];
                    cooMatrix_d->values[startIdx + writeIdx] = hashValues[localWarpId][i];
                    writeIdx++;
                }
            }
        }
    }
}

void spmspm_gpu2(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, 
    CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    
    unsigned int numRows = csrMatrix1->numRows;
    unsigned int numWarps = numRows;
    unsigned int numThreads = numWarps * WARP_SIZE;
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numBlocks = (numThreads + blockSize - 1) / blockSize;
    
    unsigned int* outputColsPool;
    float* outputValuesPool;
    cudaMalloc((void**)&outputColsPool, sizeof(unsigned int));
    cudaMalloc((void**)&outputValuesPool, sizeof(float));
    
    spmspm_kernel2<<<numBlocks, blockSize>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d,
                                              outputColsPool, outputValuesPool);
    
    cudaDeviceSynchronize();
    cudaFree(outputColsPool);
    cudaFree(outputValuesPool);
}