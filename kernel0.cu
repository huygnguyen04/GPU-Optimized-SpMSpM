#include "common.h"
#include "timer.h"

// Kernel 0: Basic GPU parallel implementation
// Each thread processes one row of the output matrix
// Uses externally allocated output pools (following CPU pattern)
// No advanced optimizations - baseline implementation

// --- RTX 6000 / RTX 4090 / V100 / T4 (48KB shared memory per block) ---
#define BLOCK_SIZE 256

// For bigger GPU, we can try using larger block size 

// --- A100 (164KB shared memory per block) ---
// #define BLOCK_SIZE 512

// --- H100 (228KB shared memory per block) ---
// #define BLOCK_SIZE 512

// =============================================================================

__global__ void spmspm_kernel0(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d,
                               unsigned int* outputColsPool, float* outputValuesPool, unsigned int numCols) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < csrMatrix1_d->numRows) {
        // Each thread gets its own portion of the pool (like CPU's outputValues and outputCols)
        float* outputValues = &outputValuesPool[row * numCols];
        unsigned int* outputCols = &outputColsPool[row * numCols];
        unsigned int numOutputCols = 0;
        
        // Initialize output values to 0 (dense accumulator indexed by column)
        for (unsigned int c = 0; c < numCols; ++c) {
            outputValues[c] = 0.0f;
        }
        
        // Iterate over non-zeros in row of matrix A
        for (unsigned int i1 = csrMatrix1_d->rowPtrs[row]; i1 < csrMatrix1_d->rowPtrs[row + 1]; ++i1) {
            unsigned int col1 = csrMatrix1_d->colIdxs[i1];
            float value1 = csrMatrix1_d->values[i1];
            
            unsigned int row2 = col1;
            
            // Iterate over non-zeros in row2 of matrix B
            for (unsigned int i2 = csrMatrix2_d->rowPtrs[row2]; i2 < csrMatrix2_d->rowPtrs[row2 + 1]; ++i2) {
                unsigned int col2 = csrMatrix2_d->colIdxs[i2];
                float value2 = csrMatrix2_d->values[i2];
                float oldVal = outputValues[col2];
                outputValues[col2] += value1 * value2;
                if (oldVal == 0.0f) {
                    outputCols[numOutputCols++] = col2;
                }
            }
        }
        
        // Write accumulated values to COO output
        if (numOutputCols > 0) {
            unsigned int startIdx = atomicAdd(&cooMatrix_d->numNonzeros, numOutputCols);
            for (unsigned int i = 0; i < numOutputCols; ++i) {
                unsigned int col = outputCols[i];
                cooMatrix_d->rowIdxs[startIdx + i] = row;
                cooMatrix_d->colIdxs[startIdx + i] = col;
                cooMatrix_d->values[startIdx + i] = outputValues[col];
            }
        }
    }
}

void spmspm_gpu0(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, 
    CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    
    unsigned int numRows = csrMatrix1->numRows;
    unsigned int numCols = csrMatrix2->numCols;
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numBlocks = (numRows + blockSize - 1) / blockSize;
    
    // Allocate output pools on GPU (following instructor's suggestion)
    unsigned int* outputColsPool;
    float* outputValuesPool;
    cudaMalloc((void**)&outputColsPool, numRows * numCols * sizeof(unsigned int));
    cudaMalloc((void**)&outputValuesPool, numRows * numCols * sizeof(float));
    
    // Launch kernel with pools
    spmspm_kernel0<<<numBlocks, blockSize>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d,
                                              outputColsPool, outputValuesPool, numCols);
    
    // Free pools
    cudaDeviceSynchronize();
    cudaFree(outputColsPool);
    cudaFree(outputValuesPool);
}