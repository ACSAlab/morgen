/*
 *   For testing
 *
 *   Copyright (C) 2013-2014 by
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


#include "graphgen.cuh"
#include "graph.cuh"
#include "cuda_util.cuh"
#include "bfs.cu"

int main(int argc, char **argv) {
	
	typedef int VertexId;
	typedef int SizeT;
	typedef int Value;

	graph<VertexId, SizeT, Value> ga;

	if (argc < 2) {
		fprintf(stderr, "at least 2 arguments\n");
		exit(1);
	}

	FILE *fp = fopen(argv[1], "r");

	if (!fp) {
		fprintf(stderr, "cannot open file\n");
		exit(1);
	}

	myGraphGen<VertexId, SizeT, Value>(fp, ga);

	ga.printInfo(true);
	ga.printOutDegrees();

	// traverse it
	BFSGraph<VertexId, SizeT, Value>(ga, (VertexId) 0);


	fclose(fp);
	ga.del();



	return 0;
}