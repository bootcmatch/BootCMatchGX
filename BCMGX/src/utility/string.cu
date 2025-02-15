#include "utility/string.h"

#include <algorithm>
#include <cctype>
#include <locale>

// trim from start (in place)
void ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
void rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(),
        s.end());
}

// trim from both ends (in place)
void trim(std::string& s)
{
    rtrim(s);
    ltrim(s);
}

// trim from start (copying)
std::string ltrim_copy(std::string s)
{
    ltrim(s);
    return s;
}

// trim from end (copying)
std::string rtrim_copy(std::string s)
{
    rtrim(s);
    return s;
}

// trim from both ends (copying)
std::string trim_copy(std::string s)
{
    trim(s);
    return s;
}

bool ends_with(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string& str, const std::string& prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}
