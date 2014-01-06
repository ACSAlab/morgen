/*
 *   Utils for CUDA
 *
 *   Copyright (C) 2013 by
 *   Yichao Cheng        onesuperclark@gmail.com
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

#include <stdio.h>



namespace morgen {

namespace util {

/******************************************************************************
 * Wrap up the cudaError_t
 ******************************************************************************/

/*
 * Locate the error when happened
 */
cudaError_t handleError(
    cudaError_t error,
    const char *msg,
    const char *filename,
    int line)
{
    if (error) {
        fprintf(stderr,
                "%s <%s, %d> CUDA Error %d: %s\n",
                msg,
                filename,
                line,
                error,
                cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}


/*
 * just print the error string
 */
cudaError_t handleError(cudaError_t error)
{
    if (error) {
        fprintf(stderr,
                "CUDA Error %d: %s\n",
                error,
                cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

}
}


