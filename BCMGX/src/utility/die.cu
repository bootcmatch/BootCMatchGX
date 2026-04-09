#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

void print_stacktrace(int nsymbols, FILE* out)
{
    void* array[nsymbols];
    int size = backtrace(array, nsymbols);
    char** symbols = backtrace_symbols(array, size);

    printf("Stack trace:\n");
    for (int i = 0; i < size; i++) {
        fprintf(out, "%s\n", symbols[i]); // contiene gli indirizzi
    }

    free(symbols);
}
