#include "logf.h"

#include <stdarg.h>

/**
 * @brief Logs a formatted message to both standard output and an optional file.
 * 
 * This function logs a formatted message to both the standard output (`stdout`) and, if a valid
 * file pointer is provided, to the specified file. The function allows for variable arguments in
 * the format similar to `printf`.
 * 
 * @param[in] fp The file pointer where the log should be written. If `NULL`, only standard output
 *               will be used. 
 * @param[in] fmt A format string that specifies how to format the remaining arguments. This string
 *                follows the same rules as `printf` format strings.
 * @param[in] ... Additional arguments that are used to fill in the placeholders in the `fmt` string.
 * 
 * @note If the `fp` parameter is `NULL`, the log will be printed only to the standard output and not
 *       to any file.
 * 
 * @note The `va_list` is processed twice, first for printing to `stdout` and second for printing to
 *       the file. It is important to ensure that the file pointer (`fp`) is valid when logging to a file.
 * 
 * @example
 * FILE* logFile = fopen("log.txt", "w");
 * logf(logFile, "Logging number: %d\n", 42);  // Logs to both stdout and the file "log.txt"
 * logf(NULL, "This message is only printed to stdout.\n"); // Logs only to stdout
 */
void logf(FILE* fp, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if (fp) {
        va_start(args, fmt);
        vfprintf(fp, fmt, args);
        va_end(args);
    }
}
