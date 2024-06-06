#pragma once

#include <string>

enum class SolverType {
    CGHS,
    FCG,
    INVALID
};

SolverType string_to_solver_type(const std::string& str);
std::string solver_type_to_string(const SolverType& val);

enum class BootstrapCompositionType {
    MULTIPLICATIVE,
    //SYMMETRIZED_MULTIPLICATIVE,
    //ADDITIVE,
    INVALID
};

BootstrapCompositionType string_to_bootstrap_composition_type(const std::string& str);
std::string bootstrap_composition_type_to_string(const BootstrapCompositionType& val);

enum class MatchType {
    SUITOR,
    INVALID
};

MatchType string_to_match_type(const std::string& str);
std::string match_type_to_string(const MatchType& val);

enum class CycleType {
    V_CYCLE,
    INVALID
};

CycleType string_to_cycle_type(const std::string& str);
std::string cycle_type_to_string(const CycleType& val);

enum class CoarseSolverType {
    L1_JACOBI,
    INVALID
};

CoarseSolverType string_to_coarse_solver_type(const std::string& str);
std::string coarse_solver_type_to_string(const CoarseSolverType& val);

enum class RelaxType {
    L1_JACOBI,
    INVALID
};

RelaxType string_to_relax_type(const std::string& str);
std::string relax_type_to_string(const RelaxType& val);

enum class PreconditionerType {
    NONE,
    L1_JACOBI,
    BCMG,
    INVALID
};

PreconditionerType string_to_preconditioner_type(const std::string& str);
std::string preconditioner_type_to_string(const PreconditionerType& val);

struct params {
    std::string rhsfile;
    std::string solfile;
    SolverType solver_type;

    // bcmg
    BootstrapCompositionType bootstrap_composition_type;
    int max_hrc;
    double conv_ratio;
    MatchType matchtype;
    int aggrsweeps;
    int aggrtype;
    int max_levels;
    CycleType cycle_type;
    CoarseSolverType coarse_solver;
    RelaxType relax_type;
    int prerelax_sweeps;
    int postrelax_sweeps;
    int itnlim;
    double rtol;
    int relaxnumber_coarse;
    int mem_alloc_size;
    int error;
    int stop_criterion;
    PreconditionerType sprec;
    int l1jacsweeps;
    int dispnorm;
};

namespace Params {
params initFromFile(const char* path);
}
