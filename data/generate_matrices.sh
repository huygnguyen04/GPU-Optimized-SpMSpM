#!/bin/bash

echo "Generating test matrices for kernels 0-4..."
echo ""

echo "=== STANDARD TEST MATRICES ==="

# Default (same as matrix0.txt)
./gen_coo 10240 32 matrix0.txt
echo "  matrix0.txt: 10240x10240, 32 nnz/row (327K total) - DEFAULT"

# Small for quick testing
./gen_coo 5000 32 matrix_small.txt
echo "  matrix_small.txt: 5000x5000, 32 nnz/row (160K total)"

# Medium
./gen_coo 8000 32 matrix_medium.txt
echo "  matrix_medium.txt: 8000x8000, 32 nnz/row (256K total)"

# Large
./gen_coo 15000 32 matrix_large.txt
echo "  matrix_large.txt: 15000x15000, 32 nnz/row (480K total)"


# More matrices for hash_size tuning
./gen_coo 10000 16 matrix_sparse.txt
echo "  matrix_sparse.txt: 10000x10000, 16 nnz/row (160K total)"

./gen_coo 10000 22 matrix_mid.txt
echo "  matrix_mid.txt: 10000x10000, 22 nnz/row (220K total)"

./gen_coo 10000 45 matrix_dense.txt
echo "  matrix_dense.txt: 10000x10000, 45 nnz/row (450K total)"