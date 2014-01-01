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
#include "bfs_queue.cu"
#include "bfs_serial.cpp"
#include "bfs_bitmask.cu"

void usage() {
	printf("\ntest <graph type> <graph type args> [--device=<device index>] "
			"[--v] [--instrumented] [--i=<num-iterations>] [--undirected]"
			"[--src=< <source idx> | randomize >]\n"
			"\n"
			"\n");
}



int main(int argc, char **argv) {
	
	CommandLineArgs args(argc, argv);

	if ((argc < 3) || args.CheckCmdLineFlag("help")) {
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


	std::string bfs_type = argv[2];


	// traverse it
	if (bfs_type == "serial") {

		BFSGraph_serial<VertexId, SizeT, Value>(ga, (VertexId) 0);

	} else if (bfs_type == "bitmask") {

		BFSGraph_gpu_bitmask<VertexId, SizeT, Value>(ga, (VertexId) 0);

	} else if (bfs_type == "queue") {

		BFSGraph_gpu_queue<VertexId, SizeT, Value>(ga, (VertexId) 0);

	} else {
		fprintf(stderr, "no traverse type is specified\n");
	}


	fclose(fp);
	ga.del();



	return 0;
}