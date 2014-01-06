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



#include <morgen/graph/graph.cuh>
#include <morgen/graph/gen/mine.cuh>

#include <morgen/bfs/bitmask.cu>
#include <morgen/bfs/queue.cu>
#include <morgen/bfs/hash.cu>
#include <morgen/bfs/serial.cu>

#include <morgen/utils/command_line.cuh>

using namespace morgen;



void usage() {
	printf("\ntest <graph file> <graph type> [bfs type] [--device=<device index>] "
			"[--slots=<number of slots>] [--outdegree] [--distribution]"
			"[--src=<source idx>]\n"
			"\n"
			"\n");
}



int main(int argc, char **argv) {
	

	typedef int VertexId;
	typedef int SizeT;
	typedef int Value;

	graph::CsrGraph<VertexId, SizeT, Value> ga;


	/*********************************************************************
	 * Commandline parsing
	 *********************************************************************/
	util::CommandLineArgs args(argc, argv);
	// 0: name   1: path   2:graph_type    3: bfs_type
	if ((argc < 4) || args.CheckCmdLineFlag("help")) {
		usage();
		return 1;
	}

	std::string graph_file_path = argv[1];
	std::string graph_type = argv[2];
	// if user only wants to inspect the graph, just skip this argument 
	std::string bfs_type = argv[3];

	// --outdegree : print out degrees of the graph?
	bool print_outdegree = args.CheckCmdLineFlag("outdegree");

	// --distribution : print the edge distribution each level?
	bool print_distribution = args.CheckCmdLineFlag("distribution");

	// --source=<source node ID>
	VertexId source = 0;
	args.GetCmdLineArgument("source", source);
	printf("traverse from %lld\n", source);

	// --slots=<number of slots>
	int slots = 0;
	args.GetCmdLineArgument("slots", slots);

	/*********************************************************************
	 * Build the graph from a file
	 *********************************************************************/
	FILE *fp = fopen(graph_file_path.c_str(), "r");
	if (!fp) {
		fprintf(stderr, "cannot open file\n");
		return 1;
	}
	

	if (graph_type == "mine") {
	
		graph::gen::myGraphGen<VertexId, SizeT, Value>(fp, ga);
	

	} else {
		fprintf(stderr, "no graph type is specified\n");
		fclose(fp);
		return 1;
	}

	// Graph Information display
	ga.printInfo(false);  //  not verbose

	if (print_outdegree) ga.printOutDegrees();
	
	if (print_distribution) 
		bfs::BFSGraph_serial<VertexId, SizeT, Value>(ga, (VertexId) 0, true);


	/*********************************************************************
	 * Traversing
	 *********************************************************************/

	if (bfs_type == "serial") {

		bfs::BFSGraph_serial<VertexId, SizeT, Value>(ga, source);

	} else if (bfs_type == "bitmask") {

		bfs::BFSGraph_gpu_bitmask<VertexId, SizeT, Value>(ga, source);

	} else if (bfs_type == "queue") {

		bfs::BFSGraph_gpu_queue<VertexId, SizeT, Value>(ga, source);

	} else if (bfs_type == "hash") {

		bfs::BFSGraph_gpu_hash<VertexId, SizeT, Value>(ga, source, slots);

	} else {
		fprintf(stderr, "no traverse type is specified\n");
	}


	fclose(fp);
	ga.del();
	return 0;
}