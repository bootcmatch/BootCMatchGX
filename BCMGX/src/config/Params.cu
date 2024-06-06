#include "config/Params.h"
#include "utility/mpi.h"
#include "utility/string.h"
#include <iostream>
#include <stdio.h>
#include <string.h>

#define DELIM "%"
#define BUFSIZE 1024
#define GLOB_MEM_ALLOC_SIZE 2000000

char linebuffer[BUFSIZE + 1];

// =============================================================================

SolverType string_to_solver_type(const std::string& str)
{
    if (str == "CGHS") {
        return SolverType::CGHS;
    } else if (str == "FCG") {
        return SolverType::FCG;
    } else {
        return SolverType::INVALID;
    }
}

std::string solver_type_to_string(const SolverType& val)
{
    switch (val) {
    case SolverType::CGHS:
        return "CGHS";
    case SolverType::FCG:
        return "FCG";
    case SolverType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class SolverType\n");
        exit(1);
    }
}

// =============================================================================

BootstrapCompositionType string_to_bootstrap_composition_type(const std::string& str)
{
    if (str == "MULTIPLICATIVE") {
        return BootstrapCompositionType::MULTIPLICATIVE;
    } else {
        return BootstrapCompositionType::INVALID;
    }
}

std::string bootstrap_composition_type_to_string(const BootstrapCompositionType& val)
{
    switch (val) {
    case BootstrapCompositionType::MULTIPLICATIVE:
        return "MULTIPLICATIVE";
    case BootstrapCompositionType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class BootstrapCompositionType\n");
        exit(1);
    }
}

// =============================================================================

MatchType string_to_match_type(const std::string& str)
{
    if (str == "SUITOR") {
        return MatchType::SUITOR;
    } else {
        return MatchType::INVALID;
    }
}

std::string match_type_to_string(const MatchType& val)
{
    switch (val) {
    case MatchType::SUITOR:
        return "SUITOR";
    case MatchType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class MatchType\n");
        exit(1);
    }
}

// =============================================================================

CycleType string_to_cycle_type(const std::string& str)
{
    if (str == "V_CYCLE") {
        return CycleType::V_CYCLE;
    } else {
        return CycleType::INVALID;
    }
}

std::string cycle_type_to_string(const CycleType& val)
{
    switch (val) {
    case CycleType::V_CYCLE:
        return "V_CYCLE";
    case CycleType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class CycleType\n");
        exit(1);
    }
}

// =============================================================================

CoarseSolverType string_to_coarse_solver_type(const std::string& str)
{
    if (str == "L1_JACOBI") {
        return CoarseSolverType::L1_JACOBI;
    } else {
        return CoarseSolverType::INVALID;
    }
}

std::string coarse_solver_type_to_string(const CoarseSolverType& val)
{
    switch (val) {
    case CoarseSolverType::L1_JACOBI:
        return "L1_JACOBI";
    case CoarseSolverType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class CoarseSolverType\n");
        exit(1);
    }
}

// =============================================================================

RelaxType string_to_relax_type(const std::string& str)
{
    if (str == "L1_JACOBI") {
        return RelaxType::L1_JACOBI;
    } else {
        return RelaxType::INVALID;
    }
}

std::string relax_type_to_string(const RelaxType& val)
{
    switch (val) {
    case RelaxType::L1_JACOBI:
        return "L1_JACOBI";
    case RelaxType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class RelaxType\n");
        exit(1);
    }
}

// =============================================================================

PreconditionerType string_to_preconditioner_type(const std::string& str)
{
    if (str == "NONE") {
        return PreconditionerType::NONE;
    } else if (str == "L1_JACOBI") {
        return PreconditionerType::L1_JACOBI;
    } else if (str == "BCMG") {
        return PreconditionerType::BCMG;
    } else {
        return PreconditionerType::INVALID;
    }
}

std::string preconditioner_type_to_string(const PreconditionerType& val)
{
    switch (val) {
    case PreconditionerType::NONE:
        return "NONE";
    case PreconditionerType::L1_JACOBI:
        return "L1_JACOBI";
    case PreconditionerType::BCMG:
        return "BCMG";
    case PreconditionerType::INVALID:
        return "INVALID";
    default:
        printf("Unhandled value for enum class PreconditionerType\n");
        exit(1);
    }
}

// =============================================================================

int get_int_param(FILE* fp, const char* param)
{
    int temp;
    char* token;
    void* out = fgets(linebuffer, BUFSIZE, fp);
    if (out == NULL) {
        printf("ERROR: reading param %s from conf file\n", param);
        exit(1);
    }
    token = strtok(linebuffer, DELIM);
    sscanf(token, "%d", &temp);
    return (temp);
}

double get_double_param(FILE* fp, const char* param)
{
    double temp;
    char* token;
    void* out = fgets(linebuffer, BUFSIZE, fp);
    if (out == NULL) {
        printf("ERROR: reading param %s from conf file\n", param);
        exit(1);
    }
    token = strtok(linebuffer, DELIM);
    sscanf(token, "%lf", &temp);
    return (temp);
}

std::string get_string_param(FILE* fp, const char* param)
{
    void* out = fgets(linebuffer, BUFSIZE, fp);
    if (out == NULL) {
        printf("ERROR: reading param %s from conf file\n", param);
        exit(1);
    }
    std::string token = strtok(linebuffer, DELIM);
    trim(token);
    return token;
}

namespace Params {
params initFromFile(const char* path)
{
    _MPI_ENV;

    params inparms;

    FILE* fp = fopen(path, "r");

    if (ISMASTER) {
        std::cout << "Reading setting file " << path << "\n";
    }
    if (fp == NULL) {
        fprintf(stdout, "Error opening setting file %s, errno = %d: %s\n", path, errno, strerror(errno));
        exit(-1);
    } else {
        if (ISMASTER) {
            std::cout << "Setting file found!"
                      << "\n";
        }
    }

    inparms.rhsfile = get_string_param(fp, "rhsfile");
    if (inparms.rhsfile == "NONE") {
        inparms.rhsfile = "";
    }
    inparms.solfile = get_string_param(fp, "solfile");
    if (inparms.solfile == "NONE") {
        inparms.solfile = "";
    }
    inparms.solver_type = string_to_solver_type(get_string_param(fp, "solver_type"));
    inparms.bootstrap_composition_type = string_to_bootstrap_composition_type(get_string_param(fp, "bootstrap_composition_type"));
    inparms.max_hrc = get_int_param(fp, "max_hrc");
    inparms.conv_ratio = get_double_param(fp, "conv_ratio");
    inparms.matchtype = string_to_match_type(get_string_param(fp, "matchtype"));
    inparms.aggrsweeps = get_int_param(fp, "aggrsweeps") + 1;
    inparms.aggrtype = get_int_param(fp, "aggrtype");
    inparms.max_levels = get_int_param(fp, "max_levels");
    inparms.cycle_type = string_to_cycle_type(get_string_param(fp, "cycle_type"));
    inparms.coarse_solver = string_to_coarse_solver_type(get_string_param(fp, "coarse_solver"));
    inparms.relax_type = string_to_relax_type(get_string_param(fp, "relax_type"));
    inparms.relaxnumber_coarse = get_int_param(fp, "inparms");
    inparms.prerelax_sweeps = get_int_param(fp, "prerelax_sweeps");
    inparms.postrelax_sweeps = get_int_param(fp, "postrelax_sweeps");
    inparms.itnlim = get_int_param(fp, "itnlim");
    inparms.rtol = get_double_param(fp, "rtol");
    inparms.mem_alloc_size = GLOB_MEM_ALLOC_SIZE;
    inparms.error = 0;

    inparms.stop_criterion = get_int_param(fp, "stop_criterion");
    inparms.sprec = string_to_preconditioner_type(get_string_param(fp, "sprec"));
    inparms.l1jacsweeps = get_int_param(fp, "l1jacsweeps");
    inparms.dispnorm = get_int_param(fp, "dispnorm");

    if (inparms.max_hrc != 1) {
        std::cout << "[ERROR] bootstrap not yet supported\n";
        inparms.error = 1;
    }
    if (inparms.aggrtype != 0) {
        std::cout << "[ERROR] aggr_type value not supported\n";
        inparms.error = 1;
    }

    fclose(fp);
    return inparms;
}

}
