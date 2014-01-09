/*
 *   Command-line parsing functionality
 *
 *   Copyright (C) 2013-2014 by
 *   Cheng Yichao        onesuperclark@gmail.com
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 */


#pragma once

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>



namespace morgen {

namespace util {

/**
 * CommandLineArgs interface
 */
class CommandLineArgs
{
protected:

    std::map<std::string, std::string> pairs;

public:

    // Constructor
    CommandLineArgs(int argc, char **argv)
    {
        using namespace std;

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-')) {
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find( '=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }

    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;
        map<string, string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end()) {
            return true;
        }
        return false;
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val);

    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);

    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return pairs.size();
    }
};


template <typename T>
void CommandLineArgs::GetCmdLineArgument(
    const char *arg_name,
    T &val)
{
    using namespace std;
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
        istringstream str_stream(itr->second);
        str_stream >> val;
    }
}


template <typename T>
void CommandLineArgs::GetCmdLineArguments(
    const char *arg_name,
    std::vector<T> &vals)
{
    using namespace std;

    // Recover multi-value string
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {

        // Clear any default values
        vals.clear();

        string val_string = itr->second;
        istringstream str_stream(val_string);
        string::size_type old_pos = 0;
        string::size_type new_pos = 0;

        // Iterate comma-separated values
        T val;
        while ((new_pos = val_string.find(',', old_pos)) != string::npos) {

            if (new_pos != old_pos) {
                str_stream.width(new_pos - old_pos);
                str_stream >> val;
                vals.push_back(val);
            }

            // skip over comma
            str_stream.ignore(1);
            old_pos = new_pos + 1;
        }

        // Read last value
        str_stream >> val;
        vals.push_back(val);
    }
}


} // Utils

} // Morgen