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


namespace morgen {

namespace util {

int getLogOf(int outDegree) {
	if (outDegree == 0) 
		return -1;
	else if (outDegree > 0 && outDegree <= 1)
		return 0;
	else if (outDegree > 1 && outDegree <= 2)
		return 1;
	else if (outDegree > 2 && outDegree <= 4)
		return 2;
	else if (outDegree > 4 && outDegree <= 8)
		return 3;
	else if (outDegree > 8 && outDegree <= 16)
		return 4;
	else if (outDegree > 16 && outDegree <= 32)
		return 5;
	else if (outDegree > 32 && outDegree <= 64)
		return 6;
	else if (outDegree > 64 && outDegree <= 128)
		return 7;
	else if (outDegree > 128 && outDegree <= 256)
		return 8;
	else if (outDegree > 256 && outDegree <= 512)
		return 9;
	else if (outDegree > 512 && outDegree <= 1024)
		return 10;
	else if (outDegree > 1024 && outDegree <= 2048)
		return 11;
	else if (outDegree > 2048 && outDegree <= 4096)
		return 12;
	else if (outDegree > 4096 && outDegree <= 8192)
		return 13;
	else if (outDegree > 8192 && outDegree <= 16384)
		return 14;
	else if (outDegree > 16384 && outDegree <= 32768)
		return 15;
	else if (outDegree > 32768 && outDegree <= 65536)
		return 16;
	else if (outDegree > 65536 && outDegree <=131072)
		return 17;
	else if (outDegree > 131072 && outDegree <=262144)
		return 18;
	else {
		fprintf(stderr, "[log] can't handle outdegree: %d \n", outDegree);
		return -2;

	}

}


}
}



