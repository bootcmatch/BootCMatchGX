template <typename T>
vector<int>* approx_match_cpu_suitor(CSR* W_)
{
    CSR* W = CSRm::copyToHost(W_);
    int* row = W->row;
    int* col = W->col;
    T* val = W->val;
    int n = W->n;

    // prepare
    vector<int>* suitor = Vector::init<int>(n, true, false);

    T* ws = MALLOC(T, n);
    for (int i = 0; i < n; i++) {
        suitor->val[i] = -1;
        ws[i] = -1;
    }

    // algorithm
    for (int i = 0; i < n; i++) {
        int u = i;
        int current = u;
        bool done = false;

        while (!done) {
            int partner = suitor->val[current];
            T heaviest = ws[current];
            for (int j = row[current]; j < row[current + 1]; j++) {
                int v = col[j];
                if (SUITOR_GT(val[j], heaviest) && SUITOR_GT(val[j], ws[v])) {
                    partner = v;
                    heaviest = val[j];
                }
            }

            done = true;

            if (heaviest != -1) {
                int y = suitor->val[partner];
                suitor->val[partner] = current;
                ws[partner] = heaviest;
                if (y != -1) {
                    current = y;
                    done = false;
                }
            }
        }
    }
    FREE(ws);

    CSRm::free(W);

    return Vector::copyToDevice(suitor);
}

template <typename T>
vector<int>* approx_match_cpu_suitor_LOCAL(CSR* W_)
{
    _MPI_ENV;

    CSR* W = CSRm::copyToHost(W_);
    int* row = W->row;
    int* col = W->col;
    T* val = W->val;
    int n = W->n;

    // prepare
    vector<int>* suitor = Vector::init<int>(n, true, false);

    T* ws = MALLOC(T, n);
    for (int i = 0; i < n; i++) {
        suitor->val[i] = -1;
        ws[i] = -1.;
    }

    int W_start = W->row_shift;
    int W_stop = W->row_shift + W->n;

    // algorithm
    for (int i = 0; i < n; i++) {
        int u = i;
        int current = u;
        bool done = false;

        while (!done) {
            int partner = suitor->val[current];
            T heaviest = ws[current];
            for (int j = row[current]; j < row[current + 1]; j++) {
                int v = col[j];
                v = v - W->row_shift;

                if (v < 0 || v >= n) {
                    continue;
                }

                if (val[j] > heaviest && val[j] > ws[v]) {
                    partner = v;
                    heaviest = val[j];
                }
            }

            done = true;

            if (heaviest != -1) {
                int y = suitor->val[partner];
                suitor->val[partner] = current;
                ws[partner] = heaviest;
                if (y != -1) {
                    current = y;
                    done = false;
                }
            }
        }
    }
    FREE(ws);
    CSRm::free(W);
    return Vector::copyToDevice(suitor);
}