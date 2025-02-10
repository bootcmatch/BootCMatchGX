/**
 * @file
 * @brief This file contains various enumerations, utility functions, and configuration structures used to define
 *        parameters and solvers for solving systems of equations.
 * 
 * The file defines solvers, preconditioners, AMG-related types, and various configuration parameters for managing 
 * and solving equations. Additionally, it provides utility functions for converting between enums and strings.
 */

#pragma once

#include <string>

#define GLOB_MEM_ALLOC_SIZE 2000000 /**< Default memory allocation size */

/**
 * @enum SolverType
 * @brief Enum class to represent the available solvers.
 */
enum class SolverType {
    CGHS, /**< Conjugate Gradient with Hybrid Solver */
    FCG, /**< Flexible Conjugate Gradient */
    CGS, /**< Conjugate Gradient Squared */
    PIPELINED_CGS, /**< Pipeline-based Conjugate Gradient Squared */
    CGS_CUBLAS, /**< Conjugate Gradient Squared using cuBLAS */
    INVALID /**< Represents an invalid solver type */
};

/**
 * @brief Utility function to convert a string to SolverType.
 * 
 * This function maps a string value to the corresponding SolverType enum value.
 * 
 * @param str The string representation of a solver type.
 * @return Corresponding SolverType enum value.
 */
SolverType string_to_solver_type(const std::string& str);

/**
 * @brief Utility function to convert SolverType to a string.
 * 
 * This function converts a SolverType enum value to its corresponding string representation.
 * 
 * @param val The SolverType enum value.
 * @return Corresponding string representation.
 */
std::string solver_type_to_string(const SolverType& val);

/**
 * @enum BootstrapCompositionType
 * @brief Enum class to represent available AMG bootstrap composition modes.
 */
enum class BootstrapCompositionType {
    MULTIPLICATIVE, /**< Multiplicative Bootstrap */
    SYMMETRIZED_MULTIPLICATIVE, /**< Symmetrized Multiplicative Bootstrap */
    ADDITIVE, /**< Additive Bootstrap */
    INVALID /**< Represents an invalid bootstrap composition type */
};

/**
 * @brief Utility function to convert a string to BootstrapCompositionType.
 * 
 * This function maps a string value to the corresponding BootstrapCompositionType enum value.
 * 
 * @param str The string representation of a bootstrap composition type.
 * @return Corresponding BootstrapCompositionType enum value.
 */
BootstrapCompositionType string_to_bootstrap_composition_type(const std::string& str);

/**
 * @brief Utility function to convert BootstrapCompositionType to a string.
 * 
 * This function converts a BootstrapCompositionType enum value to its corresponding string representation.
 * 
 * @param val The BootstrapCompositionType enum value.
 * @return Corresponding string representation.
 */
std::string bootstrap_composition_type_to_string(const BootstrapCompositionType& val);

/**
 * @enum MatchType
 * @brief Enum class to represent available AMG matching algorithms.
 */
enum class MatchType {
    SUITOR, /**< Suitor Matching Algorithm */
    INVALID /**< Represents an invalid match type */
};

/**
 * @brief Utility function to convert a string to MatchType.
 * 
 * This function maps a string value to the corresponding MatchType enum value.
 * 
 * @param str The string representation of a match type.
 * @return Corresponding MatchType enum value.
 */
MatchType string_to_match_type(const std::string& str);

/**
 * @brief Utility function to convert MatchType to a string.
 * 
 * This function converts a MatchType enum value to its corresponding string representation.
 * 
 * @param val The MatchType enum value.
 * @return Corresponding string representation.
 */
std::string match_type_to_string(const MatchType& val);

/**
 * @enum CycleType
 * @brief Enum class to represent supported AMG cycle types.
 */
enum class CycleType {
    V_CYCLE, /**< V-cycle for AMG */
    H_CYCLE, /**< H-cycle for AMG */
    W_CYCLE, /**< W-cycle for AMG */
    VARIABLE_V_CYCLE, /**< Variable V-cycle for AMG */
    INVALID /**< Represents an invalid cycle type */
};

/**
 * @brief Utility function to convert a string to CycleType.
 * 
 * This function maps a string value to the corresponding CycleType enum value.
 * 
 * @param str The string representation of a cycle type.
 * @return Corresponding CycleType enum value.
 */
CycleType string_to_cycle_type(const std::string& str);

/**
 * @brief Utility function to convert CycleType to a string.
 * 
 * This function converts a CycleType enum value to its corresponding string representation.
 * 
 * @param val The CycleType enum value.
 * @return Corresponding string representation.
 */
std::string cycle_type_to_string(const CycleType& val);

/**
 * @enum CoarseSolverType
 * @brief Enum class to represent algorithms for coarsest linear system resolution in AMG.
 */
enum class CoarseSolverType {
    L1_JACOBI,
    INVALID
};

/**
 * @brief Utility function to convert a string to CoarseSolverType.
 * 
 * This function maps a string value to the corresponding CoarseSolverType enum value.
 * 
 * @param str The string representation of a coarse solver type.
 * @return Corresponding CoarseSolverType enum value.
 */
CoarseSolverType string_to_coarse_solver_type(const std::string& str);

/**
 * @brief Utility function to convert CoarseSolverType to a string.
 * 
 * This function converts a CoarseSolverType enum value to its corresponding string representation.
 * 
 * @param val The CoarseSolverType enum value.
 * @return Corresponding string representation.
 */
std::string coarse_solver_type_to_string(const CoarseSolverType& val);

/**
 * @enum RelaxType
 * @brief Enum class to represent available smoothing algorithms for AMG.
 */
enum class RelaxType {
    L1_JACOBI, /**< L1 Jacobi relaxation */
    INVALID /**< Represents an invalid relaxation type */
};

/**
 * @brief Utility function to convert a string to RelaxType.
 * 
 * This function maps a string value to the corresponding RelaxType enum value.
 * 
 * @param str The string representation of a relaxation type.
 * @return Corresponding RelaxType enum value.
 */
RelaxType string_to_relax_type(const std::string& str);

/**
 * @brief Utility function to convert RelaxType to a string.
 * 
 * This function converts a RelaxType enum value to its corresponding string representation.
 * 
 * @param val The RelaxType enum value.
 * @return Corresponding string representation.
 */
std::string relax_type_to_string(const RelaxType& val);

/**
 * @enum PreconditionerType
 * @brief Enum class to represent available preconditioners.
 */
enum class PreconditionerType {
    NONE, /**< No preconditioner */
    L1_JACOBI, /**< L1 Jacobi preconditioner */
    BCMG, /**< BCMG preconditioner */
    AFSAI, /**< AFSAI preconditioner */
    INVALID /**< Represents an invalid preconditioner */
};

/**
 * @brief Utility function to convert a string to PreconditionerType.
 * 
 * This function maps a string value to the corresponding PreconditionerType enum value.
 * 
 * @param str The string representation of a preconditioner type.
 * @return Corresponding PreconditionerType enum value.
 */
PreconditionerType string_to_preconditioner_type(const std::string& str);

/**
 * @brief Utility function to convert PreconditionerType to a string.
 * 
 * This function converts a PreconditionerType enum value to its corresponding string representation.
 * 
 * @param val The PreconditionerType enum value.
 * @return Corresponding string representation.
 */
std::string preconditioner_type_to_string(const PreconditionerType& val);

/**
 * @struct params
 * @brief A structure to hold solver and preconditioner configuration parameters.
 * 
 * This structure holds various configuration settings such as memory allocation sizes, solver types, AMG cycle settings,
 * and other related parameters for configuring and executing solvers and preconditioners.
 */
struct params {
    int mem_alloc_size = GLOB_MEM_ALLOC_SIZE; /**< Memory allocation size */
    int error = 0; /**< Error code */

    // =========================================================================
    // General
    // =========================================================================

    std::string rhsfile = "NONE"; /**< Right-hand side file */
    std::string solfile = "NONE"; /**< Solution file */
    SolverType solver_type = SolverType::CGHS; /**< Solver type */
    int itnlim = 2000; /**< Iteration limit */
    double rtol = 1.e-6; /**< Relative tolerance */
    int dispnorm = 1; /**< Display norm */

    PreconditionerType sprec = PreconditionerType::NONE; /**< Preconditioner type */


    // =========================================================================
    // BCMG Preconditioner
    // =========================================================================

    BootstrapCompositionType bootstrap_composition_type = BootstrapCompositionType::MULTIPLICATIVE; /**< Bootstrap composition type */
    int max_hrc = 1; /**< Maximum hierarchical coarse grid levels */
    double conv_ratio = 0.8; /**< Convergence ratio */
    MatchType matchtype = MatchType::SUITOR; /**< Matching type */
    int aggrsweeps = 2; /**< Aggregation sweeps */
    int aggrtype = 0; /**< Aggregation type */
    int max_levels = 39; /**< Maximum number of levels */
    CycleType cycle_type = CycleType::V_CYCLE; /**< AMG cycle type */
    CoarseSolverType coarse_solver = CoarseSolverType::L1_JACOBI; /**< Coarse solver type */
    RelaxType relax_type = RelaxType::L1_JACOBI; /**< Relaxation type */
    int relaxnumber_coarse = 20; /**< Number of relaxation sweeps on the coarse grid */

    // int coarsesolver_type = 0;

    int prerelax_sweeps = 4;
    int postrelax_sweeps = 4;

    // =========================================================================
    // L1-JACOBI Preconditioner
    // =========================================================================

    int l1jacsweeps = 4; /**< L1-Jacobi sweeps */

    // =========================================================================
    // CGS Solver
    // =========================================================================

    int sstep = 1; /**< Solver step */
    int stop_criterion = 1; /**< Stopping criterion */
    int ru_res = 1; /**< Residual tolerance */
    int rec_res_int = 0; /**< Recovery residual interval */
};

/**
 * @namespace Params
 * @brief Namespace containing utility functions to parse configuration files to @link params @endlink.
 */
namespace Params {
    /**
     * @brief Reads parameters from a properties file.
     * 
     * This function reads and initializes parameters from a `.properties` file.
     * 
     * @param path Path to the properties file.
     * @return A params object initialized with values from the file.
     */
    params initFromPropertiesFile(const char* path);

    /**
     * @brief Reads parameters from a deprecated file format.
     * 
     * This function is deprecated. Use `initFromPropertiesFile` instead.
     * 
     * @deprecated
     * @param path Path to the deprecated configuration file.
     * @return A params object initialized with values from the file.
     */
    params initFromFile(const char* path);

    /**
     * @brief Dumps the current parameters to an output stream.
     * 
     * This function prints the current configuration parameters to a file or console.
     * 
     * @param self The params object to dump.
     * @param out The output stream to which the parameters will be dumped.
     */
    void dump(const params& self, FILE* out);
}
