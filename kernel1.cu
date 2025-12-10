
#include "common.h"
#include "timer.h"

// Kernel 1: Warp-Level Parallelism + Shared Memory
// Built directly on Kernel 0

// OPTIMIZATION we used: 
// - kernel0: 1 thread per row, global memory pools
// - kernel1: 32 threads (1 warp) per row and shared memory


// --- RTX 6000 / RTX 4090 / V100 / T4 (48KB shared memory per block) ---
// Shared memory: 4 warps * 1400 entries * 8 bytes = 44.8KB < 48KB
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)  // = 4
#define MAX_NNZ_PER_ROW 1400

// --- A100 (164KB shared memory per block) ---
// Shared memory: 8 warps * 2500 entries * 8 bytes = 160KB < 164KB
// #define BLOCK_SIZE 256
// #define WARP_SIZE 32
// #define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)  // = 8
// #define MAX_NNZ_PER_ROW 2500

// --- H100 (228KB shared memory per block) ---
// Shared memory: 8 warps * 3500 entries * 8 bytes = 224KB < 228KB
// #define BLOCK_SIZE 256
// #define WARP_SIZE 32
// #define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)  // = 8
// #define MAX_NNZ_PER_ROW 3500

// =============================================================================

__global__ void spmspm_kernel1(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d,
                               unsigned int* outputColsPool, float* outputValuesPool) {
    unsigned int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE;
    unsigned int localWarpId = threadIdx.x / WARP_SIZE;
    unsigned int row = warpId;
    
    // Shared memory accumulators
    __shared__ float sharedValues[WARPS_PER_BLOCK][MAX_NNZ_PER_ROW];
    __shared__ unsigned int sharedCols[WARPS_PER_BLOCK][MAX_NNZ_PER_ROW];
    __shared__ unsigned int sharedCount[WARPS_PER_BLOCK];
    
    if (row >= csrMatrix1_d->numRows) return;
    
    // Initialize shared memory
    if (laneId == 0) {
        sharedCount[localWarpId] = 0;
    }
    for (unsigned int k = laneId; k < MAX_NNZ_PER_ROW; k += WARP_SIZE) {
        sharedValues[localWarpId][k] = 0.0f;
        sharedCols[localWarpId][k] = 0xFFFFFFFF;
    }
    __syncwarp();
    
    unsigned int rowStart1 = csrMatrix1_d->rowPtrs[row];
    unsigned int rowEnd1 = csrMatrix1_d->rowPtrs[row + 1];
    
    for (unsigned int i1 = rowStart1; i1 < rowEnd1; ++i1) {
        unsigned int col1 = csrMatrix1_d->colIdxs[i1];
        float value1 = csrMatrix1_d->values[i1];
        unsigned int row2 = col1;
        
        unsigned int rowStart2 = csrMatrix2_d->rowPtrs[row2];
        unsigned int rowEnd2 = csrMatrix2_d->rowPtrs[row2 + 1];
        
        // 32 threads process B's row in parallel
        for (unsigned int i2 = rowStart2 + laneId; i2 < rowEnd2; i2 += WARP_SIZE) {
            unsigned int col2 = csrMatrix2_d->colIdxs[i2];
            float value2 = csrMatrix2_d->values[i2];
            float product = value1 * value2;
            
            // Search for existing column
            bool found = false;
            unsigned int count = sharedCount[localWarpId];
            
            for (unsigned int k = 0; k < count && !found; ++k) {
                if (sharedCols[localWarpId][k] == col2) {
                    atomicAdd(&sharedValues[localWarpId][k], product);
                    found = true;
                }
            }
            
            if (!found) {
                unsigned int newIdx = atomicAdd(&sharedCount[localWarpId], 1);
                if (newIdx < MAX_NNZ_PER_ROW) {
                    sharedCols[localWarpId][newIdx] = col2;
                    atomicAdd(&sharedValues[localWarpId][newIdx], product);
                }
            }
        }
        __syncwarp();
    }
    
    __syncwarp();
    
    // Lane 0 writes results
    if (laneId == 0) {
        unsigned int count = sharedCount[localWarpId];
        if (count > MAX_NNZ_PER_ROW) count = MAX_NNZ_PER_ROW;
        
        if (count > 0) {
            unsigned int startIdx = atomicAdd(&cooMatrix_d->numNonzeros, count);
            
            for (unsigned int k = 0; k < count; ++k) {
                cooMatrix_d->rowIdxs[startIdx + k] = row;
                cooMatrix_d->colIdxs[startIdx + k] = sharedCols[localWarpId][k];
                cooMatrix_d->values[startIdx + k] = sharedValues[localWarpId][k];
            }
        }
    }
}

void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, 
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
    
    spmspm_kernel1<<<numBlocks, blockSize>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d,
                                              outputColsPool, outputValuesPool);
    
    cudaDeviceSynchronize();
    cudaFree(outputColsPool);
    cudaFree(outputValuesPool);
}
