#include <string>
#include <fstream>

#include "cuda_common.hpp"

/*
 *  Reads the string assigned to a setting
 */
std::string readString
(std::ifstream& input, const char * setting);

/*
 *  Reads an integer assigned to a setting
 */
int readInt
(std::ifstream& input, const char * setting);

/*
 *  Find if tl_use_cg is in the input file
 */
bool paramEnabled
(std::ifstream& input, const char* param);

/*
 *  Returns index of desired device, or -1 if some error occurs (none specified, invalid specification, etc)
 */
int preferredDevice
(std::ifstream& input);

/*
 *  Find out the value of a parameter
 */
std::string matchParam
(FILE * input,
 const char* param_name);

