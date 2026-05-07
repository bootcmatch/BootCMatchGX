#include "config/Params.h"
#include "utility/mpi.h"
#include "utility/string.h"
#include <iostream>
#include <regex>
#include <stdio.h>
#include <string.h>

#define DELIM "%"
#define BUFSIZE 1024

char linebuffer[BUFSIZE + 1];

// =============================================================================

SolverType string_to_solver_type(const std::string& str)
{
    if (str == "CGHS") {
        return SolverType::CGHS;
    } else if (str == "FCG") {
        return SolverType::FCG;
    } else if (str == "CGS") {
        return SolverType::CGS;
    } else if (str == "PIPELINED_CGS") {
        return SolverType::PIPELINED_CGS;
    } else if (str == "CGS_CUBLAS") {
        return SolverType::CGS_CUBLAS;
    } else if (str == "LSGS") {
        return SolverType::LSGS;
    } else if (str == "CGSN") {
        return SolverType::CGSN;
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
    case SolverType::CGS:
        return "CGS";
    case SolverType::PIPELINED_CGS:
        return "PIPELINED_CGS";
    case SolverType::CGS_CUBLAS:
        return "CGS_CUBLAS";
    case SolverType::LSGS:
        return "LSGS";
    case SolverType::CGSN:
        return "CGSN";
    case SolverType::INVALID:
        return "INVALID";
    default:
        DIE("Unhandled value for enum class SolverType\n");
    }
}

// =============================================================================

BootstrapCompositionType string_to_bootstrap_composition_type(const std::string& str)
{
    if (str == "MULTIPLICATIVE") {
        return BootstrapCompositionType::MULTIPLICATIVE;
    } else if (str == "SYMMETRIZED_MULTIPLICATIVE") {
        return BootstrapCompositionType::SYMMETRIZED_MULTIPLICATIVE;
    } else if (str == "ADDITIVE") {
        return BootstrapCompositionType::ADDITIVE;
    } else {
        return BootstrapCompositionType::INVALID;
    }
}

std::string bootstrap_composition_type_to_string(const BootstrapCompositionType& val)
{
    switch (val) {
    case BootstrapCompositionType::MULTIPLICATIVE:
        return "MULTIPLICATIVE";
    case BootstrapCompositionType::SYMMETRIZED_MULTIPLICATIVE:
        return "SYMMETRIZED_MULTIPLICATIVE";
    case BootstrapCompositionType::ADDITIVE:
        return "ADDITIVE";
    case BootstrapCompositionType::INVALID:
        return "INVALID";
    default:
        DIE("Unhandled value for enum class BootstrapCompositionType\n");
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
        DIE("Unhandled value for enum class MatchType\n");
    }
}

// =============================================================================

CycleType string_to_cycle_type(const std::string& str)
{
    if (str == "V_CYCLE") {
        return CycleType::V_CYCLE;
    } else if (str == "H_CYCLE") {
        return CycleType::H_CYCLE;
    } else if (str == "W_CYCLE") {
        return CycleType::W_CYCLE;
    } else if (str == "VARIABLE_V_CYCLE") {
        return CycleType::VARIABLE_V_CYCLE;
    } else {
        return CycleType::INVALID;
    }
}

std::string cycle_type_to_string(const CycleType& val)
{
    switch (val) {
    case CycleType::V_CYCLE:
        return "V_CYCLE";
    case CycleType::H_CYCLE:
        return "H_CYCLE";
    case CycleType::W_CYCLE:
        return "W_CYCLE";
    case CycleType::VARIABLE_V_CYCLE:
        return "VARIABLE_V_CYCLE";
    case CycleType::INVALID:
        return "INVALID";
    default:
        DIE("Unhandled value for enum class CycleType\n");
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
        DIE("Unhandled value for enum class CoarseSolverType\n");
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
        DIE("Unhandled value for enum class RelaxType\n");
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
    } else if (str == "AFSAI") {
        return PreconditionerType::AFSAI;
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
    case PreconditionerType::AFSAI:
        return "AFSAI";
    case PreconditionerType::INVALID:
        return "INVALID";
    default:
        DIE("Unhandled value for enum class PreconditionerType\n");
    }
}

// =============================================================================

void parse_properties_file(const char* path, const std::function<void(const std::string&, const std::string&)>& handler)
{
    FILE* fp = fopen(path, "r");

    if (fp == NULL) {
        DIE("Error opening setting file %s, errno = %d: %s\n", path, errno, strerror(errno));
    }

    int line = 0;
    while (fgets(linebuffer, BUFSIZE, fp) != NULL) {
        line++;

        char* found = strchr(linebuffer, '#');
        int len = found == NULL ? strlen(linebuffer) : (found - linebuffer);
        if (len > 0 && linebuffer[len - 1] == '\n') {
            len--;
        }

        bool is_empty = true;
        int equal_index = -1;
        for (int i = 0; i < len; i++) {
            if (linebuffer[i] != ' ') {
                is_empty = false;
            }

            if (linebuffer[i] == '=') {
                equal_index = i;
            }
        }

        if (is_empty) {
            continue;
        }

        if (equal_index < 0) {
            DIE("Error reading setting file %s: could not find '=' at line %d\n", path, line);
        }

        std::string key(linebuffer, 0, equal_index);
        std::string value(linebuffer + equal_index + 1);

        trim(key);
        trim(value);

        handler(key, value);
    }

    if (ferror(fp)) {
        DIE("Error reading setting file %s, errno = %d: %s\n", path, errno, strerror(errno));
    }

    fclose(fp);
}

namespace Params {
InputParameters initFromPropertiesFile(const char* path)
{
    _MPI_ENV;

    if (ISMASTER) {
        std::cout << "Reading setting file " << path << "\n";
    }

    InputParameters inparms;

    auto handler = [&](const std::string& key, const std::string& value) {
        if (ISMASTER) {
            printf("%20s = %-20s\n", key.c_str(), value.c_str());
        }

        if (key == "rhsfile") {
            inparms.rhsfile = value;
        } else if (key == "solfile") {
            inparms.solfile = value;
        } else if (key == "solver") {
            inparms.solver_types.push_back(string_to_solver_type(value));
        } else if (key == "itnlim") {
            inparms.itnlim = std::stoi(value);
        } else if (key == "rtol") {
            inparms.rtol = std::stod(value);
        } else if (key == "dispnorm") {
            inparms.dispnorm = std::stoi(value);
        } else if (key == "preconditioner") {
            inparms.sprec = string_to_preconditioner_type(value);
        } else if (key == "bootstrap_type") {
            inparms.bootstrap_composition_type = string_to_bootstrap_composition_type(value);
        } else if (key == "max_hrc") {
            inparms.max_hrc = std::stoi(value);
        } else if (key == "conv_ratio") {
            inparms.conv_ratio = std::stod(value);
        } else if (key == "matchtype") {
            inparms.matchtype = string_to_match_type(value);
        } else if (key == "aggrsweeps") {
            inparms.aggrsweeps = std::stoi(value);
        } else if (key == "aggr_type") {
            inparms.aggrtype = std::stoi(value);
        } else if (key == "max_levels") {
            inparms.max_levels = std::stoi(value);
        } else if (key == "cycle_type") {
            inparms.cycle_type = string_to_cycle_type(value);
        } else if (key == "coarse_solver") {
            inparms.coarse_solver = string_to_coarse_solver_type(value);
        } else if (key == "relax_type") {
            inparms.relax_type = string_to_relax_type(value);
        } else if (key == "relaxnumber_coarse") {
            inparms.relaxnumber_coarse = std::stoi(value);
        } else if (key == "smoothed_prolungator") {
            inparms.smoothed_prolungator = std::stoi(value);
        } /*else if (key == "coarsesolver_type") {

        }*/
        else if (key == "prerelax_sweeps") {
            inparms.prerelax_sweeps = std::stoi(value);
        } else if (key == "postrelax_sweeps") {
            inparms.postrelax_sweeps = std::stoi(value);
        } else if (key == "l1jacsweeps") {
            inparms.l1jacsweeps = std::stoi(value);
        } else if (key == "sstep") {
            inparms.ssteps.push_back(std::stoi(value));
        } else if (key == "stop_criterion") {
            inparms.stop_criterion = std::stoi(value);
        } else if (key == "ru_res") {
            inparms.ru_res = std::stoi(value);
        } else if (key == "rec_res_int") {
            inparms.rec_res_int = std::stoi(value);
        }

        else if (key == "use_chebyshev") {
            inparms.use_chebyshev = std::stoi(value);
        } else if (key == "gs_iterations") {
            inparms.gs_iterations = std::stoi(value);
        } else if (key == "gs_tol") {
            inparms.gs_tol = std::stod(value);
        } else if (key == "power_method_tol") {
            inparms.power_method_tol = std::stod(value);
        } else if (key == "power_method_itnlim") {
            inparms.power_method_itnlim = std::stoi(value);
        } else if (key == "use_fgs") {
            inparms.use_fgs = std::stoi(value);
        }
    };

    parse_properties_file(path, handler);

    if (inparms.solver_types.empty()) {
        inparms.solver_types.push_back(SolverType::CGHS);
    }
    if (inparms.ssteps.empty()) {
        inparms.ssteps.push_back(1);
    }

    return inparms;
}

void dump(const InputParameters& self, const CurrentParameters& cp, FILE* out)
{
    fprintf(out, "\trhsfile: %s\n", self.rhsfile.c_str());
    fprintf(out, "\tsolfile: %s\n", self.solfile.c_str());
    fprintf(out, "\tsolver_type: %s\n", solver_type_to_string(cp.solver_type).c_str());

    // bcmg
    fprintf(out, "\tbootstrap_composition_type: %s\n", bootstrap_composition_type_to_string(self.bootstrap_composition_type).c_str());
    fprintf(out, "\tmax_hrc: %d\n", self.max_hrc);
    fprintf(out, "\tconv_ratio: %f\n", self.conv_ratio);
    fprintf(out, "\tmatchtype %s\n", match_type_to_string(self.matchtype).c_str());
    fprintf(out, "\taggrsweeps: %d\n", self.aggrsweeps);
    fprintf(out, "\taggrtype: %d\n", self.aggrtype);
    fprintf(out, "\tmax_levels: %d\n", self.max_levels);
    fprintf(out, "\tcycle_type: %s\n", cycle_type_to_string(self.cycle_type).c_str());
    fprintf(out, "\tcoarse_solver_type: %s\n", coarse_solver_type_to_string(self.coarse_solver).c_str());
    fprintf(out, "\trelax_type: %s\n", relax_type_to_string(self.relax_type).c_str());
    fprintf(out, "\tprerelax_sweeps: %d\n", self.prerelax_sweeps);
    fprintf(out, "\tpostrelax_sweeps: %d\n", self.postrelax_sweeps);
    fprintf(out, "\titnlim: %d\n", self.itnlim);
    fprintf(out, "\trtol: %f\n", self.rtol);
    fprintf(out, "\trelaxnumber_coarse: %d\n", self.relaxnumber_coarse);
    fprintf(out, "\tmem_alloc_size: %d\n", self.mem_alloc_size);
    fprintf(out, "\terror: %d\n", self.error);
    fprintf(out, "\tsstep: %d\n", cp.sstep);
    fprintf(out, "\tstop_criterion: %d\n", self.stop_criterion);
    fprintf(out, "\tru_res: %d\n", self.ru_res);
    fprintf(out, "\tpreconditioner_type: %s\n", preconditioner_type_to_string(self.sprec).c_str());
    fprintf(out, "\tl1jacsweeps: %d\n", self.l1jacsweeps);
    fprintf(out, "\trec_res_int: %d\n", self.rec_res_int);
    fprintf(out, "\tdispnorm: %d\n", self.dispnorm);
}

}

namespace FemParams {
FemParameters initFromPropertiesFile(const char* path)
{
    _MPI_ENV;

    if (ISMASTER) {
        std::cout << "Reading fem config file " << path << "\n";
    }

    FemParameters inparms;

    auto handler = [&](const std::string& key, const std::string& value) {
        if (ISMASTER) {
            printf("%20s = %-20s\n", key.c_str(), value.c_str());
        }

        if (key == "nx") {
            inparms.nx = std::stoi(value);
        } else if (key == "ny") {
            inparms.ny = std::stoi(value);
        } else if (key == "P") {
            inparms.P = std::stoi(value);
        } else if (key == "Q") {
            inparms.Q = std::stoi(value);
        } else if (key == "E") {
            inparms.E = std::stod(value);
        } else if (key == "nu") {
            inparms.nu = std::stod(value);
        }
    };

    parse_properties_file(path, handler);

    return inparms;
}
}