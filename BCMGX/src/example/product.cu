#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <cassert>

struct CSR {
    int nrows = 0, ncols = 0;
    std::vector<int> rowptr;
    std::vector<int> col;
    std::vector<double> val;
};

struct MMHeader {
    bool symmetric = false;
};

MMHeader parse_header(std::ifstream& in) {
    std::string line;
    MMHeader h;

    if (!std::getline(in, line))
        throw std::runtime_error("File vuoto");

    if (line.find("MatrixMarket") == std::string::npos)
        throw std::runtime_error("Header MatrixMarket non valido");

    if (line.find("symmetric") != std::string::npos)
        h.symmetric = true;

    return h;
}

CSR read_mtx(const std::string& filename) {
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Impossibile aprire " + filename);

    MMHeader header = parse_header(in);

    std::string line;
    int nrows = 0, ncols = 0, nnz = 0;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::stringstream ss(line);
        ss >> nrows >> ncols >> nnz;
        break;
    }

    if (nrows <= 0 || ncols <= 0 || nnz < 0)
        throw std::runtime_error("Dimensioni non valide");

    std::vector<std::unordered_map<int, double>> rows(nrows);

    int r, c;
    double v;
    while (in >> r >> c >> v) {
        if (r < 1 || r > nrows || c < 1 || c > ncols)
            throw std::runtime_error("Indice fuori range");

        r--; c--;
        rows[r][c] += v;

        if (header.symmetric && r != c) {
            rows[c][r] += v;
        }
    }

    CSR A;
    A.nrows = nrows;
    A.ncols = ncols;
    A.rowptr.resize(nrows + 1, 0);

    for (int i = 0; i < nrows; ++i)
        A.rowptr[i + 1] = A.rowptr[i] + rows[i].size();

    int total_nnz = A.rowptr.back();
    A.col.resize(total_nnz);
    A.val.resize(total_nnz);

    for (int i = 0; i < nrows; ++i) {
        int pos = A.rowptr[i];

        std::vector<std::pair<int, double>> sorted;
        sorted.reserve(rows[i].size());
        for (std::unordered_map<int, double>::const_iterator it = rows[i].begin();
             it != rows[i].end(); ++it) {
            sorted.push_back(*it);
        }

        std::sort(sorted.begin(), sorted.end());

        for (size_t k = 0; k < sorted.size(); ++k) {
            A.col[pos] = sorted[k].first;
            A.val[pos] = sorted[k].second;
            pos++;
        }
    }

    assert(A.rowptr[0] == 0);
    assert(A.rowptr.back() == (int)A.col.size());

    return A;
}

CSR multiply(const CSR& A, const CSR& B) {
    if (A.ncols != B.nrows)
        throw std::runtime_error("Dimensioni incompatibili");

    CSR C;
    C.nrows = A.nrows;
    C.ncols = B.ncols;
    C.rowptr.resize(C.nrows + 1, 0);

    std::vector<std::vector<int>> Brow(B.nrows);
    for (int i = 0; i < B.nrows; ++i) {
        for (int k = B.rowptr[i]; k < B.rowptr[i + 1]; ++k) {
            Brow[i].push_back(k);
        }
    }

    std::vector<std::unordered_map<int, double>> Crow(C.nrows);

    for (int i = 0; i < A.nrows; ++i) {
        for (int ka = A.rowptr[i]; ka < A.rowptr[i + 1]; ++ka) {
            int k = A.col[ka];
            double va = A.val[ka];

            for (size_t idx = 0; idx < Brow[k].size(); ++idx) {
                int kb = Brow[k][idx];
                int j = B.col[kb];
                Crow[i][j] += va * B.val[kb];
            }
        }
    }

    for (int i = 0; i < C.nrows; ++i)
        C.rowptr[i + 1] = C.rowptr[i] + Crow[i].size();

    int nnz = C.rowptr.back();
    C.col.resize(nnz);
    C.val.resize(nnz);

    for (int i = 0; i < C.nrows; ++i) {
        int pos = C.rowptr[i];

        std::vector<std::pair<int, double>> sorted;
        sorted.reserve(Crow[i].size());
        for (std::unordered_map<int, double>::const_iterator it = Crow[i].begin();
             it != Crow[i].end(); ++it) {
            sorted.push_back(*it);
        }

        std::sort(sorted.begin(), sorted.end());

        for (size_t k = 0; k < sorted.size(); ++k) {
            C.col[pos] = sorted[k].first;
            C.val[pos] = sorted[k].second;
            pos++;
        }
    }

    for (int i = 0; i < C.nrows; ++i) {
        for (int k = C.rowptr[i]; k + 1 < C.rowptr[i + 1]; ++k) {
            assert(C.col[k] < C.col[k + 1]);
        }
    }

    return C;
}

void write_mtx(const std::string& filename, const CSR& A) {
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Impossibile scrivere " + filename);

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << A.nrows << " " << A.ncols << " " << A.val.size() << "\n";

    for (int i = 0; i < A.nrows; ++i) {
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
            out << i + 1 << " "
                << A.col[k] + 1 << " "
                << A.val[k] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " A.mtx B.mtx C.mtx\n";
        return 1;
    }

    try {
        CSR A = read_mtx(argv[1]);
        CSR B = read_mtx(argv[2]);
        CSR C = multiply(A, B);
        write_mtx(argv[3], C);
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
