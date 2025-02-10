/**
 * @file
 */
#pragma once

#include <string>

/**
 * @brief Trims leading whitespace from a string (in place).
 * 
 * This function removes all whitespace characters from the beginning of the string.
 * It modifies the original string directly.
 * 
 * @param s The string to be trimmed.
 */
void ltrim(std::string& s);

/**
 * @brief Trims trailing whitespace from a string (in place).
 * 
 * This function removes all whitespace characters from the end of the string.
 * It modifies the original string directly.
 * 
 * @param s The string to be trimmed.
 */
void rtrim(std::string& s);

/**
 * @brief Trims both leading and trailing whitespace from a string (in place).
 * 
 * This function removes all whitespace characters from both the beginning 
 * and the end of the string. It modifies the original string directly.
 * 
 * @param s The string to be trimmed.
 */
void trim(std::string& s);

/**
 * @brief Trims leading whitespace from a string (copying).
 * 
 * This function removes all whitespace characters from the beginning of the string
 * and returns a new string with the leading whitespace removed.
 * The original string remains unchanged.
 * 
 * @param s The string to be trimmed.
 * @return A new string with leading whitespace removed.
 */
std::string ltrim_copy(std::string s);

/**
 * @brief Trims trailing whitespace from a string (copying).
 * 
 * This function removes all whitespace characters from the end of the string
 * and returns a new string with the trailing whitespace removed.
 * The original string remains unchanged.
 * 
 * @param s The string to be trimmed.
 * @return A new string with trailing whitespace removed.
 */
std::string rtrim_copy(std::string s);

/**
 * @brief Trims both leading and trailing whitespace from a string (copying).
 * 
 * This function removes all whitespace characters from both the beginning 
 * and the end of the string, and returns a new string with the whitespace removed.
 * The original string remains unchanged.
 * 
 * @param s The string to be trimmed.
 * @return A new string with leading and trailing whitespace removed.
 */
std::string trim_copy(std::string s);

/**
 * @brief Checks if a string ends with a given suffix.
 * 
 * This function checks whether the string `str` ends with the specified suffix.
 * 
 * @param str The string to check.
 * @param suffix The suffix to look for.
 * @return `true` if `str` ends with `suffix`, `false` otherwise.
 */
bool ends_with(const std::string& str, const std::string& suffix);

/**
 * @brief Checks if a string starts with a given prefix.
 * 
 * This function checks whether the string `str` starts with the specified prefix.
 * 
 * @param str The string to check.
 * @param prefix The prefix to look for.
 * @return `true` if `str` starts with `prefix`, `false` otherwise.
 */
bool starts_with(const std::string& str, const std::string& prefix);
