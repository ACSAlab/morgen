/*
 *   Timing Utils
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

namespace morgen {

namespace util {


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

        return (sec * 1000.0) + (usec / 1000.0);
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

} // Utils
} // Morgen

