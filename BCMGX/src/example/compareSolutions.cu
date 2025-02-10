#include "datastruct/vector.h"
#include "op/basic.h"
#include "utility/arrays.h"
#include "utility/handles.h"

#define USAGE "Usage:\n\t%s <FILE_NAME_1.txt> <FILE_NAME_2.txt> [THRESHOLD]\n"

double norm(handles* h, vector<double>* v)
{
    // wrap raw pointer with a device_ptr
    // thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(v->val);
    // return std::sqrt(thrust::inner_product(dev_ptr, dev_ptr + v->n, dev_ptr, 0));
    return Vector::norm(h->cublas_h, v);
}

/**
 * Compute norm(sol1 - sol2) / norm(sol1) and
 * eventually verify it is under a given threshold
 */
int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    handles* h = Handles::init();

    vector<double>* sol1 = Vector::load<double>(argv[1], true);
    vector<double>* sol2 = Vector::load<double>(argv[2], true);

    // debugArray("sol2[%d] = %lf\n", sol2->val, sol2->n, true, stderr);
    // debugArray("sol1[%d] = %lf\n", sol1->val, sol1->n, true, stderr);

    // Compute norm(sol1)
    double denominator = norm(h, sol1);
    if (denominator == 0.) {
        printf("sol1 must be != <0>\n");
        exit(1);
    }

    // Compute sol1 - sol2 and put the result into sol1
    // Vector::axpy<double>(h->cublas_h, sol2, sol1, -1);
    my_axpby(sol2->val, sol2->n, sol1->val, -1., 1.);

    // debugArray("(sol1 - sol2)[%d]  = %lf\n", sol1->val, sol1->n, true, stderr);

    // Compute norm(sol1 - sol2)
    double numerator = norm(h, sol1);

    double result = numerator / denominator;

    // printf("norm(sol1 - sol2): %lg\n", numerator);
    // printf("norm(sol1): %lg\n", denominator);
    printf("Relative error: %lg\n", result);

    if (argc > 3) {
        double threshold = atof(argv[3]);
        if (result > threshold) {
            printf("Relative error exceeds threshold (%lg)\n", threshold);
            exit(1);
        }
    }

    Vector::free(sol1);
    Vector::free(sol2);

    Handles::free(h);

    return 0;
}
