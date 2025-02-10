

void CSRm::compare_nnz(CSR* A, CSR* B, int type)
{
    FILE* fp = stdout;
    CSR *A_ = NULL, *B_ = NULL;
    bool A_dev_flag, B_dev_flag;
    itype nnz_border = 0;

    if (A->n != B->n || A->m != B->m) {
        fprintf(fp, "A->n != B->n || A->m != B->m\n");
        return;
    }
    if ((type != 4) && (A->nnz != B->nnz)) {
        fprintf(fp, "(type != 4) && (A->nnz != B->nnz)\n");
        fprintf(fp, "A->nnz = %d,  B->nnz = %d\n", A->nnz, B->nnz);
        if (A->nnz > B->nnz) {
            nnz_border = B->nnz;
        } else {
            nnz_border = A->nnz;
        }
    }

    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
        A_dev_flag = true;
    } else {
        A_ = A;
        A_dev_flag = false;
    }

    if (B->on_the_device) {
        B_ = CSRm::copyToHost(B);
        B_dev_flag = true;
    } else {
        B_ = B;
        B_dev_flag = false;
    }

    switch (type) {
    case 0:
        fprintf(fp, "A ROW: %d (%d)\n\t", A_->n, A_->full_n);
        for (int i = 0; i < A_->n + 1; i++) {
            if (A_->row[i] == B_->row[i]) {
                fprintf(fp, "%3d ", A_->row[i]);
            } else {
                fprintf(fp, "\033[0;31m%3d\033[0m ", A_->row[i]);
            }
        }
        fprintf(fp, "\n\nB ROW: %d (%d)\n\t", B_->n, B_->full_n);
        for (int i = 0; i < B_->n + 1; i++) {
            if (A_->row[i] == B_->row[i]) {
                fprintf(fp, "%3d ", B_->row[i]);
            } else {
                fprintf(fp, "\033[0;31m%3d\033[0m ", B_->row[i]);
            }
        }
        break;
    case 1:
        fprintf(fp, "A COL:\n\t");
        for (int i = 0; i < nnz_border; i++) {
            if (A_->col[i] == B_->col[i]) {
                fprintf(fp, "%3d ", A_->col[i]);
            } else {
                fprintf(fp, "\033[0;31m%3d\033[0m ", A_->col[i]);
            }
        }
        fprintf(fp, "\n\nB COL:\n\t");
        for (int i = 0; i < nnz_border; i++) {
            if (A_->col[i] == B_->col[i]) {
                fprintf(fp, "%3d ", B_->col[i]);
            } else {
                fprintf(fp, "\033[0;31m%3d\033[0m ", B_->col[i]);
            }
        }
        break;
    case 2:
        fprintf(fp, "A VAL:\n\t");
        for (int i = 0; i < nnz_border; i++) {
            if (A_->val[i] == B_->val[i]) {
                fprintf(fp, "%6.2lf ", A_->val[i]);
            } else {
                fprintf(fp, "\033[0;31m%6.2lf\033[0m ", A_->val[i]);
            }
        }
        fprintf(fp, "\n\nB VAL:\n\t");
        for (int i = 0; i < nnz_border; i++) {
            if (A_->val[i] == B_->val[i]) {
                fprintf(fp, "%6.2lf ", B_->val[i]);
            } else {
                fprintf(fp, "\033[0;31m%6.2lf\033[0m ", B_->val[i]);
            }
        }
        break;
    case 3:
        fprintf(fp, "Nnz differences in MATRIX_Form:\n\t[ ... ]\n");
        break;
    case 4:
        fprintf(fp, "Nnz differences in boolMATRIX_Form:\n");
        for (int i = 0; i < A_->n; i++) {
            fprintf(fp, "\t");
            for (int j = 0; j < A_->m; j++) {
                if (j % 32 == 0) {
                    fprintf(fp, "| ");
                }
                int flag = 0, tempA = A_->row[i], tempB = B_->row[i];
                for (tempA = A_->row[i]; flag == 0 && (i != (A_->n) - 1 ? tempA < (A_->row[i + 1]) : tempA < A_->nnz); tempA++) {
                    if (A_->col[tempA] == j) {
                        flag = 1;
                    }
                }
                for (tempB = B_->row[i]; (i != (B_->n) - 1 ? tempB < (B_->row[i + 1]) : tempB < B_->nnz); tempB++) {
                    if (B_->col[tempB] == j) {
                        if (flag == 0) {
                            flag = 1;
                        } else {
                            if (A_->val[tempA - 1] != B_->val[tempB]) {
                                flag = 2;
                            } else {
                                flag = -1;
                            }
                        }
                        break;
                    }
                }
                if (flag == 0 || flag == -1) {
                    if (flag == 0) {
                        fprintf(fp, "O ");
                    } else {
                        fprintf(fp, "\033[0;32mX\033[0m ");
                    }
                } else {
                    if (flag == 1) {
                        fprintf(fp, "\033[0;33mX\033[0m ");
                    } else {
                        fprintf(fp, "\033[0;35mX\033[0m ");
                    }
                }
            }
            fprintf(fp, "\n");
        }
        break;
    }
    fprintf(fp, "\n\n");

    if (A_dev_flag) {
        CSRm::free(A_);
    }
    if (B_dev_flag) {
        CSRm::free(B_);
    }
    return;
}

CSR* CSRm::clone(CSR* A_)
{
    CSR *Out = NULL, *A = NULL;
    bool dev_flag;
    if (A_->on_the_device) {
        A = CSRm::copyToHost(A_);
        dev_flag = true;
    } else {
        A = A_;
        dev_flag = false;
    }
    Out = CSRm::init(A->n, A->m, A->nnz, true, false, false, A->full_n, A->row_shift);
    Out->row = (itype*)malloc(sizeof(itype) * ((A->n) + 1));
    Out->col = (itype*)malloc(sizeof(itype) * (A->nnz));
    Out->val = (vtype*)malloc(sizeof(vtype) * (A->nnz));

    memcpy((void*)Out->row, (const void*)A->row, sizeof(itype) * ((A->n) + 1));
    memcpy((void*)Out->col, (const void*)A->col, sizeof(itype) * (A->nnz));
    memcpy((void*)Out->val, (const void*)A->val, sizeof(vtype) * (A->nnz));

    if (dev_flag) {
        CSRm::free(A);
        A = Out;
        Out = CSRm::copyToDevice(Out);
        CSRm::free(A);
    }
    return (Out);
}

bool CSRm::equals(CSR* A, CSR* B)
{
    CSR *A_ = NULL, *B_ = NULL;
    bool A_dev_flag, B_dev_flag, r = true;

    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
        A_dev_flag = true;
    } else {
        A_ = A;
        A_dev_flag = false;
    }

    if (B->on_the_device) {
        B_ = CSRm::copyToHost(B);
        B_dev_flag = true;
    } else {
        B_ = B;
        B_dev_flag = false;
    }

    if ((A_->n != B_->n)) {
        r = false;
        //         printf("(A_->n != B_->n)\n");
        //         printf("memcmp(A_->row, B_->row, sizeof(itype)*((A_->n)+1))\n");
    } else {
        if (memcmp(A_->row, B_->row, sizeof(itype) * ((A_->n) + 1))) {
            r = false;
            //             printf("memcmp(A_->row, B_->row, sizeof(itype)*((A_->n)+1))\n");
        }
    }
    if ((A_->m != B_->m)) {
        r = false;
        //         printf("(A_->m != B_->m)\n");
    }
    if ((A_->nnz != B_->nnz)) {
        r = false;
        //         printf("(A_->nnz != B_->nnz)\n");
        //         printf("memcmp(A_->val, B_->val, sizeof(vtype)*A_->nnz)\n");
        //         printf("memcmp(A_->col, B_->col, sizeof(itype)*A_->nnz)\n");
    } else {
        if (memcmp(A_->val, B_->val, sizeof(vtype) * A_->nnz)) {
            r = false;
            //             printf("memcmp(A_->val, B_->val, sizeof(vtype)*A_->nnz)\n");
        }
        if (memcmp(A_->col, B_->col, sizeof(itype) * A_->nnz)) {
            r = false;
            //             printf("memcmp(A_->col, B_->col, sizeof(itype)*A_->nnz)\n");
        }
    }

    if (A_dev_flag) {
        CSRm::free(A_);
    }
    if (B_dev_flag) {
        CSRm::free(B_);
    }
    return (r);
}

void CSRm::freeStruct(CSR* A)
{
    std::free(A);
}



void CSRm::partialAlloc(CSR* A, bool init_row, bool init_col, bool init_val)
{

    assert(A->on_the_device);

    cudaError_t err;
    if (init_val) {
        cudaMalloc_CNT
            err
            = cudaMalloc((void**)&A->val, A->nnz * sizeof(vtype));
        CHECK_DEVICE(err);
    }
    if (init_col) {
        cudaMalloc_CNT
            err
            = cudaMalloc((void**)&A->col, A->nnz * sizeof(itype));
        CHECK_DEVICE(err);
    }
    if (init_row) {
        cudaMalloc_CNT
            err
            = cudaMalloc((void**)&A->row, (A->n + 1) * sizeof(itype));
        CHECK_DEVICE(err);
    }
}

bool CSRm::halo_equals(halo_info* a, halo_info* b)
{
    _MPI_ENV;

    if (a->to_receive_n != b->to_receive_n) {
        return (false);
    }
    //     if (Vector::equals<itype>(a->to_receive, b->to_receive) != true)
    //         return(false);
    //     if (Vector::equals<itype>(a->to_receive_d, b->to_receive_d) != true)
    //         return(false);
    if ((b->to_receive_n > 0) && (memcmp((void*)a->to_receive->val, (const void*)b->to_receive->val, sizeof(itype) * a->to_receive->n) != 0)) {
        return (false);
    }

    if (memcmp((void*)b->to_receive_counts, (const void*)a->to_receive_counts, sizeof(int) * nprocs) != 0) {
        return (false);
    }
    if (memcmp((void*)b->to_receive_spls, (const void*)a->to_receive_spls, sizeof(int) * nprocs) != 0) {
        return (false);
    }
    if ((b->to_receive_n > 0) && (memcmp((void*)b->what_to_receive, (const void*)a->what_to_receive, sizeof(vtype) * b->to_receive_n) != 0)) {
        return (false);
    }

    if (a->to_send_n != b->to_send_n) {
        return (false);
    }
    //     if (Vector::equals<itype>(a->to_send, b->to_send) != true)
    //         return(false);
    //     if (Vector::equals<itype>(a->to_send_d, b->to_send_d) != true)
    //         return(false);
    if ((b->to_receive_n > 0) && (memcmp((void*)a->to_send->val, (const void*)b->to_send->val, sizeof(itype) * a->to_send->n) != 0)) {
        return (false);
    }

    if (memcmp((void*)b->to_send_counts, (const void*)a->to_send_counts, sizeof(int) * nprocs) != 0) {
        return (false);
    }
    if (memcmp((void*)b->to_send_spls, (const void*)a->to_send_spls, sizeof(int) * nprocs) != 0) {
        return (false);
    }
    if ((b->to_receive_n > 0) && (memcmp((void*)b->what_to_send, (const void*)a->what_to_send, sizeof(vtype) * b->to_send_n) != 0)) {
        return (false);
    }

    return (true);
}

bool CSRm::chk_uniprol(CSR* A)
{
    CSR* A_ = NULL;
    bool dev_flag, r = true;

    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
        dev_flag = true;
    } else {
        A_ = A;
        dev_flag = false;
    }

    for (int i = 1; r && (i < A->n); i++) {
        if (A_->row[i] != A_->row[i - 1] + 1) {
            r = false;
        }
    }

    if (dev_flag) {
        CSRm::free(A_);
    }
    return (r);
}

halo_info CSRm::clone_halo(halo_info* a)
{
    _MPI_ENV;

    halo_info b;

    b.to_receive_n = a->to_receive_n;
    if (a->to_receive_n > 0) {
        b.to_receive = Vector::clone<gstype>(a->to_receive);
        b.to_receive_d = Vector::clone<gstype>(a->to_receive_d);
    }

    b.to_receive_counts = (int*)malloc(sizeof(int) * nprocs);
    b.to_receive_spls = (int*)malloc(sizeof(int) * nprocs);
    b.what_to_receive = (vtype*)malloc(sizeof(vtype) * nprocs);
    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc(&(b.what_to_receive_d), sizeof(vtype) * b.to_receive_n));

    if (a->to_receive_n != 0) {
        memcpy((void*)b.to_receive_counts, (const void*)a->to_receive_counts, sizeof(int) * nprocs);
        memcpy((void*)b.to_receive_spls, (const void*)a->to_receive_spls, sizeof(int) * nprocs);
        memcpy((void*)b.what_to_receive, (const void*)a->what_to_receive, sizeof(vtype) * b.to_receive_n);
        //         if (a->what_to_receive_d != NULL)
        //             CHECK_DEVICE( cudaMemcpy(b.what_to_receive_d, a->what_to_receive_d, sizeof(vtype) * b.to_receive_n, cudaMemcpyDeviceToDevice) );
    }

    b.to_send_n = a->to_send_n;
    printf("a->to_send_n = %d, a->to_send->n = %d\n", a->to_send_n, a->to_send->n);
    if (a->to_send_n > 0) {
        b.to_send = Vector::clone<itype>(a->to_send);
        b.to_send_d = Vector::clone<itype>(a->to_send_d);
    }

    b.to_send_counts = (int*)malloc(sizeof(int) * nprocs);
    b.to_send_spls = (int*)malloc(sizeof(int) * nprocs);
    b.what_to_send = (vtype*)malloc(sizeof(vtype) * nprocs);
    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc(&(b.what_to_send_d), sizeof(vtype) * b.to_send_n));

    if (a->to_send_n != 0) {
        memcpy((void*)b.to_send_counts, (const void*)a->to_send_counts, sizeof(int) * nprocs);
        memcpy((void*)b.to_send_spls, (const void*)a->to_send_spls, sizeof(int) * nprocs);
        memcpy((void*)b.what_to_send, (const void*)a->what_to_send, sizeof(vtype) * b.to_send_n);
        //         if (a->what_to_receive_d != NULL)
        //             CHECK_DEVICE( cudaMemcpy(b.what_to_send_d, a->what_to_send_d, sizeof(vtype) * b.to_send_n, cudaMemcpyDeviceToDevice) );
    }

    return (b);
}

void CSRm::checkMatching(vector<itype>* v_)
{
    _MPI_ENV;
    vector<itype>* V = NULL;
    if (v_->on_the_device) {
        V = Vector::copyToHost(v_);
    } else {
        V = v_;
    }

    for (int i = 0; i < V->n; i++) {
        int v = i;
        int u = V->val[i];

        if (u == -1) {
            continue;
        }

        if (V->val[u] != v) {
            printf("\n%d]ERROR-MATCHING: %d %d %d\n", myid, i, v, V->val[u]);
            exit(1);
        }
    }

    if (v_->on_the_device) {
        Vector::free(V);
    }
}

vtype CSRm::vectorANorm(cublasHandle_t cublas_h, CSR* A, vector<vtype>* x)
{
    _MPI_ENV;

    if (nprocs > 1) {
        assert(A->n != x->n);
    }

    vector<vtype>* temp = CSRVector_product_MPI(A, x, 1);

    vector<vtype>* x_shift = Vector::init<vtype>(A->n, false, true);

    x_shift->val = x->val + A->row_shift;
    vtype local_norm = Vector::dot(cublas_h, temp, x_shift), norm;

    if (nprocs > 1) {
        CHECK_MPI(MPI_Allreduce(
            &local_norm,
            &norm,
            1, // sizeof(vtype),
            MPI_DOUBLE, // MPI_BYTE,
            MPI_SUM,
            MPI_COMM_WORLD));
        local_norm = norm;
    }

    norm = sqrt(local_norm);

    Vector::free(temp);

    return norm;
}




#include <cusparse.h>

const char* cusparseGetStatusString(cusparseStatus_t error)
{
    switch (error) {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "<unknown>";
}

#define CHECK_CUSPARSE(X)                                                                                   \
    {                                                                                                       \
        cusparseStatus_t status = X;                                                                        \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                            \
            const char* err_str = cusparseGetStatusString(status);                                          \
            fprintf(stderr, "[ERROR CUSPARSE] :\n\t%s; LINE: %d; FILE: %s\n", err_str, __LINE__, __FILE__); \
            exit(1);                                                                                        \
        }                                                                                                   \
    }

CSR* CSRm::Transpose_monoproc_cusparse(CSR* A)
{
    assert(A->on_the_device);

    static cusparseHandle_t cusparse_h;
    static int cusparse_h_initialized = 0;

    if (!cusparse_h_initialized) {
        CHECK_CUSPARSE(cusparseCreate(&cusparse_h));
        cusparse_h_initialized = 1;
    }

    cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxbase = CUSPARSE_INDEX_BASE_ZERO;

    // --------------------------- Custom CudaMalloc ---------------------------------
    CSR* AT = CSRm::init((stype)A->m, (gstype)A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
    // CSR *AT = CSRm::init(n_rows, A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
    //  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // CSR *AT = CSRm::init(A->m, A->n, A->nnz, false, true, A->is_symmetric, A->m, 0);
    // AT->val = CustomCudaMalloc::alloc_vtype(AT->nnz);
    // AT->col = CustomCudaMalloc::alloc_itype(AT->nnz);
    // AT->row = CustomCudaMalloc::alloc_itype((AT->n) +1);
    // AT->custom_alloced = true;
    //  -------------------------------------------------------------------------------

    size_t buff_T_size = 0;

    cusparseStatus_t err = cusparseCsr2cscEx2_bufferSize(
        cusparse_h,
        A->n,
        A->m,
        A->nnz,
        A->val,
        A->row,
        A->col,
        AT->val,
        AT->row,
        AT->col,
        CUDA_R_64F,
        copyValues,
        idxbase,
        CUSPARSE_CSR2CSC_ALG1,
        &buff_T_size);

    CHECK_CUSPARSE(err);
    assert(buff_T_size);

    void* buff_T = NULL;
    CHECK_DEVICE(cudaMalloc(&buff_T, buff_T_size));

    err = cusparseCsr2cscEx2(
        cusparse_h,
        A->n,
        A->m,
        A->nnz,
        A->val,
        A->row,
        A->col,
        AT->val,
        AT->row,
        AT->col,
        CUDA_R_64F,
        copyValues,
        idxbase,
        CUSPARSE_CSR2CSC_ALG1,
        buff_T);
    CHECK_CUSPARSE(err);

    CHECK_DEVICE(cudaFree(buff_T));

    //CHECK_CUSPARSE(cusparseDestroy(cusparse_h));

    return AT;
}

CSR* CSRm::Transpose_cusparse(CSR* A, stype n_rows, bool used_by_solver)
{
    static cusparseHandle_t cusparse_h;
    static int cusparse_h_initialized = 0;

    if (!cusparse_h_initialized) {
        CHECK_CUSPARSE(cusparseCreate(&cusparse_h));
        cusparse_h_initialized = 1;
    }

    assert(A->on_the_device);

    cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxbase = CUSPARSE_INDEX_BASE_ZERO;

    // --------------------------- Custom CudaMalloc ---------------------------------
    // CSR *AT = CSRm::init(A->m, A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
    // CSR *AT = CSRm::init(n_rows, A->full_n, A->nnz, true, true, A->is_symmetric, A->m, 0);
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    CSR* AT = CSRm::init(n_rows, A->full_n, A->nnz, false, true, A->is_symmetric, A->m, 0);
    AT->val = CustomCudaMalloc::alloc_vtype(AT->nnz, (used_by_solver ? 0 : 1));
    AT->col = CustomCudaMalloc::alloc_itype(AT->nnz, (used_by_solver ? 0 : 1));
    AT->row = CustomCudaMalloc::alloc_itype((AT->n) + 1, (used_by_solver ? 0 : 1));
    AT->custom_alloced = true;
    // -------------------------------------------------------------------------------

    size_t buff_T_size = 0;

    cusparseStatus_t err = cusparseCsr2cscEx2_bufferSize(
        cusparse_h,
        A->n,
        A->m,
        A->nnz,
        A->val,
        A->row,
        A->col,
        AT->val,
        AT->row,
        AT->col,
        CUDA_R_64F,
        copyValues,
        idxbase,
        CUSPARSE_CSR2CSC_ALG1,
        &buff_T_size);

    CHECK_CUSPARSE(err);
    assert(buff_T_size);

    void* buff_T = NULL;
    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc(&buff_T, buff_T_size));

    err = cusparseCsr2cscEx2(
        cusparse_h,
        A->n,
        A->m,
        A->nnz,
        A->val,
        A->row,
        A->col,
        AT->val,
        AT->row,
        AT->col,
        CUDA_R_64F,
        copyValues,
        idxbase,
        CUSPARSE_CSR2CSC_ALG1,
        buff_T);
    CHECK_CUSPARSE(err);

    CHECK_DEVICE(cudaFree(buff_T));

    return AT;
}

CSR* CSRm::Transpose_monoproc(CSR* A)
{
    assert(A->on_the_device);
    // --------------------------- Custom CudaMalloc ---------------------------------
    CSR* AT = CSRm::init(A->m, A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
    // CSR *AT = CSRm::init(n_rows, A->full_n, A->nnz, true, true, A->is_symmetric, A->m, 0);
    //  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // CSR *AT = CSRm::init(n_rows, A->full_n, A->nnz, false, true, A->is_symmetric, A->m, 0);
    // AT->val = CustomCudaMalloc::alloc_vtype(AT->nnz, (used_by_solver ? 0: 1));
    // AT->col = CustomCudaMalloc::alloc_itype(AT->nnz, (used_by_solver ? 0: 1));
    // AT->row = CustomCudaMalloc::alloc_itype((AT->n) +1, (used_by_solver ? 0: 1));
    // AT->custom_alloced = true;
    //  -------------------------------------------------------------------------------

    itype *temp_row, *temp_col, *temp_buff;
    CHECK_DEVICE(cudaMalloc(&temp_row, ((AT->n) + 1) * sizeof(itype)));
    CHECK_DEVICE(cudaMalloc(&temp_col, (AT->nnz) * sizeof(itype)));
    CHECK_DEVICE(cudaMalloc(&temp_buff, (AT->nnz) * sizeof(itype)));
    CHECK_DEVICE(cudaMemset(temp_row, 0, ((AT->n) + 1) * sizeof(itype)));

    GridBlock gb;
    gb = gb1d(A->n, BLOCKSIZE);
    // STEP 1 --- Count the number of nnz values per column (the new row_idx vector)
    _prepare_column_ptr<<<gb.g, gb.b>>>(A->n, A->row, A->col, temp_row);

    // STEP 2 --- Write row indices of A, they are the column indices of AT
    _write_row_indices<<<gb.g, gb.b>>>(A->n, A->row, temp_col);

    void* d_temp_storage = NULL;
    size_t temp_stor_bytes_1 = 0, temp_stor_bytes_2 = 0, temp_stor_bytes_3 = 0;
    // Determine temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_stor_bytes_1, temp_row, AT->row, (AT->n) + 1);
    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_stor_bytes_2, A->col, temp_buff, A->val, AT->val, A->nnz);
    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_stor_bytes_3, A->col, temp_buff, temp_col, AT->col, A->nnz);

    CHECK_DEVICE(cudaMalloc(&d_temp_storage, MAX(temp_stor_bytes_1, MAX(temp_stor_bytes_2, temp_stor_bytes_3))));

    // STEP 3 --- create AT->row
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_stor_bytes_1, temp_row, AT->row, (AT->n) + 1);

    // A STABLE sorting algorithm MUST be used.
    // STEP 4 --- create AT->val
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_stor_bytes_2, A->col, temp_buff, A->val, AT->val, A->nnz);
    // STEP 5 --- create AT->col
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_stor_bytes_3, A->col, temp_buff, temp_col, AT->col, A->nnz);

    CHECK_DEVICE(cudaFree(d_temp_storage));
    CHECK_DEVICE(cudaFree(temp_col));
    CHECK_DEVICE(cudaFree(temp_buff));
    CHECK_DEVICE(cudaFree(temp_row));

    return AT;
}