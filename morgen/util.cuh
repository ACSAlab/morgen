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


#pragma once


#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>


/**************************************************************************
 * Some macros
 **************************************************************************/
#define MORGEN_MAX(a, b) ((a > b) ? a : b)

#define MORGEN_MIN(a, b) ((a < b) ? a : b)

/**************************************************************************
 * Single variable on GPU
 **************************************************************************/
template<typename Value>
struct var {

    Value  *elem;
    Value  *d_elem;      

    var(Value v = 0) { 
        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;

        if (HandleError(cudaHostAlloc((void **)&elem, sizeof(Value) * 1, flags),
                        "var: cudaHostAlloc(elem) failed", __FILE__, __LINE__)) 
            exit(1);

        *elem = v;

        // Get the device pointer
        if (HandleError(cudaHostGetDevicePointer((void **) &d_elem, (void *) elem, 0),
                        "var: cudaHostGetDevicePointer(d_elem) failed", __FILE__, __LINE__))
            exit(1);
    }

    Value getVal() {
    	return *elem;
    }

    void set(Value v) {
    	*elem = v;
    }

    void del() {
        HandleError(cudaFreeHost(elem), "var: cudaFreeHost(elem) failed",
                    __FILE__, __LINE__);
        elem = NULL;
        d_elem = NULL;
    }

 };


/**************************************************************************
 * Static queue
 **************************************************************************/
template<typename Value, typename SizeT>
struct hashed {

    SizeT   slot_num;
    SizeT   n;                // size in all
    SizeT   each_slot_size;

    Value  	*elems;
    SizeT   *slot_sizes;      //the logical size will be changed on gpu 
    SizeT   *slot_offsets;    

    Value   *d_elems;
    SizeT   *d_slot_sizes;
    SizeT   *d_slot_offsets;

    hashed() : n(0), slot_num(0), elems(NULL), slot_sizes(NULL), slot_offsets(NULL), d_elems(NULL), d_slot_sizes(NULL), d_slot_offsets(NULL) {} 

    hashed(SizeT _n, SizeT s_num) {  

        slot_num = s_num;

        // e.g. 7 elements will fit into 4 slots
        // each slots has 2 elements
    	each_slot_size = _n / slot_num + 1;
    	n = each_slot_size * slot_num;

        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;

        for (int i = 0; i < slot_num; i++) {
        	if (HandleError(cudaHostAlloc((void **)&slot_sizes, sizeof(SizeT) * slot_num, flags),
          	                "hashed: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) exit(1);
        	if (HandleError(cudaHostAlloc((void **)&slot_offsets, sizeof(SizeT) * slot_num, flags),
                            "hashed: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) exit(1);
        	if (HandleError(cudaHostAlloc((void **)&elems, sizeof(SizeT) * n, flags),
                            "hashed: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) exit(1);
    	}

    	// initalize
    	for (int i = 0; i < slot_num; i++) {
    		slot_sizes[i] = 0;
    		slot_offsets[i] = i * each_slot_size;
    	}

        // Get the device pointer
        if (HandleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                        "hashed: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__)) exit(1);
        if (HandleError(cudaHostGetDevicePointer((void **) &d_slot_sizes, (void *) slot_sizes, 0),
                        "hashed: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__)) exit(1);
        if (HandleError(cudaHostGetDevicePointer((void **) &d_slot_offsets, (void *) slot_offsets, 0),
                        "hashed: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__)) exit(1);
    }


	// insert on cpu end
    int insert(Value key) {
    	SizeT hash = key % slot_num;
    	slot_sizes[hash] += 1;  // increase before writing
    	SizeT pos = slot_offsets[hash] + slot_sizes[hash];
    	elems[pos] = key;
    	return 0;  // succeed
    }

    // get the largest slot in the hash table
    int max_slot_size() {
    	SizeT  logical_size = 0;
    	for (int i = 0 ; i < slot_num; i++) {
    		logical_size = MORGEN_MAX(logical_size, slot_sizes[i]);
    	}
    	return logical_size;
    }

    // sum each slot size up
    int sum_slot_size() {
    	SizeT  logical_size = 0;
    	for (int i = 0 ; i < slot_num; i++) {
    		logical_size += slot_sizes[i];
    	}
    	return logical_size;
    }

    void del() {
        HandleError(cudaFreeHost(elems), "hashed: cudaFreeHost(elems) failed", __FILE__, __LINE__);
        HandleError(cudaFreeHost(slot_sizes), "hashed: cudaFreeHost(slot_sizes) failed", __FILE__, __LINE__);
        HandleError(cudaFreeHost(slot_offsets), "hashed: cudaFreeHost(slot_offsets) failed", __FILE__, __LINE__);
        
        n = 0;
        slot_num = 0;
        each_slot_size = 0;
        
        elems = NULL;
        slot_sizes = NULL;     
        slot_offsets = NULL;  
        d_elems = NULL;
        d_slot_sizes = NULL;     
        d_slot_offsets = NULL; 
    }

 };


/**************************************************************************
 * Static queue
 **************************************************************************/
template<typename Value, typename SizeT>
struct queued {

    SizeT     n;         //maximal allocated size
    Value  *elems;
    SizeT     *sizep;      //the logical size will be changed on gpu 

    Value  *d_elems;
    SizeT     *d_sizep;

    queued() : n(0), sizep(NULL), elems(NULL), d_sizep(NULL), d_elems(NULL) {} 

    queued(SizeT _n) { 

        n = _n;
    
        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;
        if (HandleError(cudaHostAlloc((void **)&elems, sizeof(SizeT) * n, flags),
                        "queued: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) 
            exit(1);
        if (HandleError(cudaHostAlloc((void **)&sizep, sizeof(SizeT) * 1, flags),
                        "queued: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) 
            exit(1);

        *sizep = 0;

        // Get the device pointer
        if (HandleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                        "queued: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__))
            exit(1);
        if (HandleError(cudaHostGetDevicePointer((void **) &d_sizep, (void *) sizep, 0),
                        "queued: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__))
            exit(1);

    }


    /**
     * A.K.A. enqueue
     */
    int append(Value v) {
        if (*sizep >= n)    // has been full
            return -1;
        else {
            elems[*sizep] = v; 
            *sizep += 1;
            return 0;
        }
    }


    int size() {
        if (sizep)  return *sizep;
        else        return -1;
    }

    void print() {
        for (int i = 0; i < *sizep; i++) {
            printf("%lld\n", (long long)elems[i]);
        }
        printf("\n");
    }

    void del() {
        HandleError(cudaFreeHost(elems), "queued: cudaFreeHost(elems) failed",
                    __FILE__, __LINE__);
        HandleError(cudaFreeHost(sizep), "queued: cudaFreeHost(sizep) failed",
                    __FILE__, __LINE__);
        n = 0;
        sizep = NULL;
        d_sizep = NULL;
        elems = NULL;
        d_elems = NULL;
    }

 };



/******************************************************************************
 * Auxiliary list
 ******************************************************************************/
template<typename Value, typename SizeT>
struct list
{
	SizeT   n;
	Value   *elems;
	Value   *d_elems;

	list() : n(0), elems(NULL), d_elems(NULL) {}

	list(SizeT _n) {

		n = _n;

		// mapped & pinned
		int flags = cudaHostAllocMapped;
        if (HandleError(cudaHostAlloc((void **)&elems, sizeof(Value) * n, flags),
                        "list: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) 
        	exit(1);
        if (HandleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                        "list: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__))
            exit(1);
	}

	void del() {
		HandleError(cudaFreeHost(elems), "list: cudaFreeHost(elems) failed",
                    __FILE__, __LINE__);
		elems = NULL;
		d_elems = NULL;
		n = 0;
	}

	// setting to some value on CPU serially
	void all_to(Value x) {
		for (int i = 0; i < n; i++) {
			elems[i] = x;
		}
	}

	void print_log() {
		FILE* log = fopen("log.txt", "w");
		for (int i = 0; i < n; i++) {
			fprintf(log, "%lld\n", (long long)elems[i]);
		}
		fprintf(log, "\n");
	}

	void set(SizeT i, Value x) { elems[i] = x; }

};


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
 * Timing
 ******************************************************************************/

struct CpuTimer
{
	rusage _start;
	rusage _stop;

	void start()
	{
		getrusage(RUSAGE_SELF, &_start);
	}

	void stop()
	{
		getrusage(RUSAGE_SELF, &_stop);
	}

	float elapsedMillis()
	{
		float sec = _stop.ru_utime.tv_sec - _start.ru_utime.tv_sec;
		float usec = _stop.ru_utime.tv_usec - _start.ru_utime.tv_usec;

		return (sec * 1000) + (usec / 1000);
	}

};

struct GpuTimer
{
	cudaEvent_t _start;
	cudaEvent_t _stop;

	GpuTimer()
	{
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(_start);
		cudaEventDestroy(_stop);
	}

	void start()
	{
		cudaEventRecord(_start, 0);
	}

	void stop()
	{
		cudaEventRecord(_stop, 0);
	}

	float elapsedMillis()
	{
		float elapsed;
		cudaEventSynchronize(_stop);
		cudaEventElapsedTime(&elapsed, _start, _stop);
		return elapsed;
	}
};



