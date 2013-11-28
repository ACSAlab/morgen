/*
 *   Utils
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



#include <stdio.h>
#include <math.h>
#include <float.h>

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>



/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

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



/******************************************************************************
 * Device initialization
 ******************************************************************************/

void DeviceInit(CommandLineArgs &args)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "No devices supporting CUDA.\n");
		exit(1);
	}
	int dev = 0;
	args.GetCmdLineArgument("device", dev);
	if (dev < 0) {
		dev = 0;
	}
	if (dev > deviceCount - 1) {
		dev = deviceCount - 1;
	}
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	if (deviceProp.major < 1) {
		fprintf(stderr, "Device does not support CUDA.\n");
		exit(1);
	}
    if (!args.CheckCmdLineFlag("quiet")) {
        printf("Using device %d: %s\n", dev, deviceProp.name);
    }

	cudaSetDevice(dev);
}




/******************************************************************************
 * Templated routines for printing keys/values to the console 
 ******************************************************************************/

template<typename T> 
void PrintValue(T val) {
	val.Print();
}

template<>
void PrintValue<char>(char val) {
	printf("%d", val);
}

template<>
void PrintValue<short>(short val) {
	printf("%d", val);
}

template<>
void PrintValue<int>(int val) {
	printf("%d", val);
}

template<>
void PrintValue<long>(long val) {
	printf("%ld", val);
}

template<>
void PrintValue<long long>(long long val) {
	printf("%lld", val);
}

template<>
void PrintValue<float>(float val) {
	printf("%f", val);
}

template<>
void PrintValue<double>(double val) {
	printf("%f", val);
}

template<>
void PrintValue<unsigned char>(unsigned char val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned short>(unsigned short val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned int>(unsigned int val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned long>(unsigned long val) {
	printf("%lu", val);
}

template<>
void PrintValue<unsigned long long>(unsigned long long val) {
	printf("%llu", val);
}



/******************************************************************************
 * Helper routines for list construction and validation 
 ******************************************************************************/

/**
 * Compares the equivalence of two arrays
 */
template <typename T, typename SizeT>
int CompareResults(T* computed, T* reference, SizeT len, bool verbose = true)
{
	for (SizeT i = 0; i < len; i++) {

		if (computed[i] != reference[i]) {
			printf("INCORRECT: [%lu]: ", (unsigned long) i);
			PrintValue<T>(computed[i]);
			printf(" != ");
			PrintValue<T>(reference[i]);

			if (verbose) {
				printf("\nresult[...");
				for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
					PrintValue<T>(computed[j]);
					printf(", ");
				}
				printf("...]");
				printf("\nreference[...");
				for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
					PrintValue<T>(reference[j]);
					printf(", ");
				}
				printf("...]");
			}

			return 1;
		}
	}

	printf("CORRECT");
	return 0;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceResults(
	T *h_reference,
	T *d_data,
	size_t num_elements,
	bool verbose = true,
	bool display_data = false)
{
	// Allocate array on host
	T *h_data = (T*) malloc(num_elements * sizeof(T));

	// Reduction data back
	cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

	// Display data
	if (display_data) {
		printf("Reference:\n");
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_reference[i]);
			printf(", ");
		}
		printf("\n\nData:\n");
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_data[i]);
			printf(", ");
		}
		printf("\n\n");
	}

	// Check
	int retval = CompareResults(h_data, h_reference, num_elements, verbose);

	// Cleanup
	if (h_data) free(h_data);

	return retval;
}

int CompareDeviceResults(
	util::NullType *h_reference,
	util::NullType *d_data,
	size_t num_elements,
	bool verbose = true,
	bool display_data = false)
{
	return 0;
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceDeviceResults(
	T *d_reference,
	T *d_data,
	size_t num_elements,
	bool verbose = true,
	bool display_data = false)
{
	// Allocate array on host
	T *h_reference = (T*) malloc(num_elements * sizeof(T));
	T *h_data = (T*) malloc(num_elements * sizeof(T));

	// Reduction data back
	cudaMemcpy(h_reference, d_reference, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

	// Display data
	if (display_data) {
		printf("Reference:\n");
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_reference[i]);
			printf(", ");
		}
		printf("\n\nData:\n");
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_data[i]);
			printf(", ");
		}
		printf("\n\n");
	}

	// Check
	int retval = CompareResults(h_data, h_reference, num_elements, verbose);

	// Cleanup
	if (h_reference) free(h_reference);
	if (h_data) free(h_data);

	return retval;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
void DisplayDeviceResults(
	T *d_data,
	size_t num_elements)
{
	// Allocate array on host
	T *h_data = (T*) malloc(num_elements * sizeof(T));

	// Reduction data back
	cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

	// Display data
	printf("\n\nData:\n");
	for (int i = 0; i < num_elements; i++) {
		PrintValue(h_data[i]);
		printf(", ");
	}
	printf("\n\n");

	// Cleanup
	if (h_data) free(h_data);
}



/******************************************************************************
 * Timing
 ******************************************************************************/

struct CpuTimer
{
	rusage start;
	rusage stop;

	void Start()
	{
		getrusage(RUSAGE_SELF, &start);
	}

	void Stop()
	{
		getrusage(RUSAGE_SELF, &stop);
	}

	float ElapsedMillis()
	{
		float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
		float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

		return (sec * 1000) + (usec / 1000);
	}

};

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float ElapsedMillis()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};



