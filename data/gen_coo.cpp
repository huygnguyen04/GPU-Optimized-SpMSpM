#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " N nnz_per_row output_file\n";
        std::cerr << "  N           : matrix dimension (square N x N)\n";
        std::cerr << "  nnz_per_row : number of nonzeros per row (<= N)\n";
        std::cerr << "  output_file : path to output file\n";
        return 1;
    }

    unsigned int N = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
    unsigned int nnz_per_row = static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));
    const char* out_path = argv[3];

    if (N == 0) {
        std::cerr << "Error: N must be > 0.\n";
        return 1;
    }
    if (nnz_per_row == 0 || nnz_per_row > N) {
        std::cerr << "Error: nnz_per_row must be in [1, N].\n";
        return 1;
    }

    unsigned int total_nnz = N * nnz_per_row;

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "Error: could not open output file '" << out_path << "'.\n";
        return 1;
    }

    // Header: numRows, numNonzeros
    out << N << "\n";
    out << total_nnz << "\n";

    // Deterministic RNG for reproducibility
    std::mt19937 rng(12345);

    // Temporary array of column indices 0..N-1
    std::vector<unsigned int> cols(N);
    std::iota(cols.begin(), cols.end(), 0);

    // For each row, pick nnz_per_row distinct columns
    for (unsigned int row = 0; row < N; ++row) {
        std::shuffle(cols.begin(), cols.end(), rng);
        for (unsigned int k = 0; k < nnz_per_row; ++k) {
            unsigned int col = cols[k];
            out << row << " " << col << "\n";
        }
    }

    out.close();
    std::cout << "Wrote " << N << "x" << N
              << " matrix with " << total_nnz << " nonzeros to "
              << out_path << "\n";

    return 0;
}
