#include <vector>
#include <string>
#include <sstream>

//·Ö½âstring×Ö·û´®
std::vector<std::string> split(std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}