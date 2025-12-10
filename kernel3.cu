#include "common.h"
#include "timer.h"
#include <vector>
#include <algorithm>

// Kernel 3: Row-Work Modeling + Row Bucketing (Dalton-style optimization)
// Built on Kernel 2

// OPTIMIZATION: Row bucketing based on work estimation
// - kernel2: Same hash size for all rows (1024)
// - kernel3: Different hash sizes for small/large rows

// FIXES over previous version:
// 1. Single memory allocation for all row IDs (avoid multiple malloc/free)
// 2. Use __ldg() for read-only cache optimization
// 3. Skip empty rows (Fi = 0)
// 4. Better threshold tuning
// 5. Warp shuffle for efficient counting
//
// Based on Dalton et al. 2015 (TOMS) ESC paper


// --- RTX 6000 / RTX 4090 / V100 / T4 (48KB shared memory per block) ---
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// Only 2 buckets: small and large (simpler, less overhead)
#define HASH_SIZE_SMALL 512    // For rows with F_i <= 256
#define HASH_SIZE_LARGE 1024   // For rows with F_i > 256
#define THRESHOLD_SMALL 256

// =============================================================================

__device__ __forceinline__ unsigned int hashFunc3(unsigned int col, unsigned int hashSize) {
    return (col * 2654435761u) & (hashSize - 1);
}

// Template kernel with optimizations
template<int HASH_SIZE>
__global__ void spmspm_hash_kernel3(
    const CSRMatrix* __restrict__ csrMatrix1_d,
    const CSRMatrix* __restrict__ csrMatrix2_d,
    COOMatrix* __restrict__ cooMatrix_d,
    const unsigned int* __restrict__ rowIds,
    unsigned int numRowsToProcess,
    unsigned int outputCapacity
) {
    unsigned int warpGlobalId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE;
    unsigned int localWarpId = threadIdx.x / WARP_SIZE;
    
    if (warpGlobalId >= numRowsToProcess) return;
    
    unsigned int row = rowIds[warpGlobalId];
    
    __shared__ unsigned int hashCols[WARPS_PER_BLOCK][HASH_SIZE];
    __shared__ float hashValues[WARPS_PER_BLOCK][HASH_SIZE];
    
    // Cooperative hash table initialization
    for (unsigned int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
        hashCols[localWarpId][i] = 0xFFFFFFFF;
        hashValues[localWarpId][i] = 0.0f;
    }
    __syncwarp();
    
    // Use __ldg() for read-only cache optimization
    unsigned int rowStart1 = __ldg(&csrMatrix1_d->rowPtrs[row]);
    unsigned int rowEnd1 = __ldg(&csrMatrix1_d->rowPtrs[row + 1]);
    
    for (unsigned int i = rowStart1; i < rowEnd1; ++i) {
        unsigned int col1 = __ldg(&csrMatrix1_d->colIdxs[i]);
        float value1 = __ldg(&csrMatrix1_d->values[i]);
        
        unsigned int row2 = col1;
        unsigned int rowStart2 = __ldg(&csrMatrix2_d->rowPtrs[row2]);
        unsigned int rowEnd2 = __ldg(&csrMatrix2_d->rowPtrs[row2 + 1]);
        
        // Warp cooperatively processes row of matrix B
        for (unsigned int j = rowStart2 + laneId; j < rowEnd2; j += WARP_SIZE) {
            unsigned int col2 = __ldg(&csrMatrix2_d->colIdxs[j]);
            float value2 = __ldg(&csrMatrix2_d->values[j]);
            float product = value1 * value2;
            
            unsigned int hashIdx = hashFunc3(col2, HASH_SIZE);
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
    
    // Warp-cooperative counting using shuffle reduction
    unsigned int localCount = 0;
    for (unsigned int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
        if (hashCols[localWarpId][i] != 0xFFFFFFFF) {
            localCount++;
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        localCount += __shfl_down_sync(0xFFFFFFFF, localCount, offset);
    }
    
    // Allocate and write output with bounds checking
    __shared__ unsigned int outputBase[WARPS_PER_BLOCK];
    if (laneId == 0 && localCount > 0) {
        outputBase[localWarpId] = atomicAdd(&cooMatrix_d->numNonzeros, localCount);
    }
    __syncwarp();
    
    if (laneId == 0 && localCount > 0) {
        unsigned int baseIdx = outputBase[localWarpId];
        unsigned int writeIdx = 0;
        
        // Bounds check to prevent buffer overflow
        if (baseIdx + localCount <= outputCapacity) {
            for (unsigned int i = 0; i < HASH_SIZE; ++i) {
                if (hashCols[localWarpId][i] != 0xFFFFFFFF) {
                    cooMatrix_d->rowIdxs[baseIdx + writeIdx] = row;
                    cooMatrix_d->colIdxs[baseIdx + writeIdx] = hashCols[localWarpId][i];
                    cooMatrix_d->values[baseIdx + writeIdx] = hashValues[localWarpId][i];
                    writeIdx++;
                }
            }
        }
    }
}

void spmspm_gpu3(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, 
    CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    
    // Get output capacity from GPU COO matrix
    COOMatrix cooMatrixInfo;
    cudaMemcpy(&cooMatrixInfo, cooMatrix_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    unsigned int outputCapacity = cooMatrixInfo.capacity;
    
    // Step 1: Compute row work estimates and bucket rows (CPU-side)
    std::vector<unsigned int> smallRows, largeRows;
    smallRows.reserve(csrMatrix1->numRows);
    largeRows.reserve(csrMatrix1->numRows);
    
    for (unsigned int i = 0; i < csrMatrix1->numRows; ++i) {
        unsigned int rowStart = csrMatrix1->rowPtrs[i];
        unsigned int rowEnd = csrMatrix1->rowPtrs[i + 1];
        
        // Skip empty rows in matrix A
        if (rowStart == rowEnd) continue;
        
        // Compute F_i (flop count for row i)
        unsigned int Fi = 0;
        for (unsigned int p = rowStart; p < rowEnd; ++p) {
            unsigned int colA = csrMatrix1->colIdxs[p];
            Fi += (csrMatrix2->rowPtrs[colA + 1] - csrMatrix2->rowPtrs[colA]);
        }
        
        // Skip rows with no work
        if (Fi == 0) continue;
        
        if (Fi <= THRESHOLD_SMALL)
            smallRows.push_back(i);
        else
            largeRows.push_back(i);
    }
    
    dim3 block(BLOCK_SIZE);
    
    // Step 2: Allocate device memory ONCE for both buckets
    unsigned int totalRows = smallRows.size() + largeRows.size();
    if (totalRows == 0) return;
    
    unsigned int* d_allRows;
    cudaMalloc(&d_allRows, totalRows * sizeof(unsigned int));
    
    // Copy small rows to device (at offset 0)
    if (!smallRows.empty()) {
        cudaMemcpy(d_allRows, smallRows.data(), 
                   smallRows.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    
    // Copy large rows to device (at offset smallRows.size())
    if (!largeRows.empty()) {
        cudaMemcpy(d_allRows + smallRows.size(), largeRows.data(),
                   largeRows.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    
    // Step 3: Launch kernels for each bucket
    if (!smallRows.empty()) {
        unsigned int numWarps = smallRows.size();
        unsigned int numThreads = numWarps * WARP_SIZE;
        dim3 grid((numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        spmspm_hash_kernel3<HASH_SIZE_SMALL><<<grid, block>>>(
            csrMatrix1_d, csrMatrix2_d, cooMatrix_d, 
            d_allRows,  // Points to small rows
            smallRows.size(),
            outputCapacity);
    }
    
    if (!largeRows.empty()) {
        unsigned int numWarps = largeRows.size();
        unsigned int numThreads = numWarps * WARP_SIZE;
        dim3 grid((numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        spmspm_hash_kernel3<HASH_SIZE_LARGE><<<grid, block>>>(
            csrMatrix1_d, csrMatrix2_d, cooMatrix_d,
            d_allRows + smallRows.size(),  // Points to large rows
            largeRows.size(),
            outputCapacity);
    }
    
    cudaDeviceSynchronize();
    
    // Step 4: Single cleanup
    cudaFree(d_allRows);
}