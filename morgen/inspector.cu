/*
 *   For inspecting graph structure before runs
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
#include "graph.cuh"
#include "util.cuh"
#include "cuda_util.cuh"
#include "bfs_serial.cpp"

void usage() {
	printf("\ninspect <graph> [--distribution]\n"
			"\n"
			"\n");
}



int main(int argc, char **argv) {
	
	CommandLineArgs args(argc, argv);

	if ((argc < 2) || args.CheckCmdLineFlag("help")) {
		usage();
		return 1;
	}


	typedef int VertexId;
	typedef int SizeT;
	typedef int Value;

	graph<VertexId, SizeT, Value> ga;

	FILE *fp = fopen(argv[1], "r");

	if (!fp) {
		fprintf(stderr, "cannot open file\n");
		exit(1);
	}

	myGraphGen<VertexId, SizeT, Value>(fp, ga);

	ga.printInfo(false);
	ga.printOutDegrees();


	if (args.CheckCmdLineFlag("distribution"))
		BFSGraph_serial<VertexId, SizeT, Value>(ga, (VertexId) 0, true);


	fclose(fp);
	ga.del();



	return 0;
}