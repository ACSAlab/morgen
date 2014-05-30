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



#include <morgen/graph/csr_graph.cuh>
#include <morgen/graph/gen/mine.cuh>
#include <morgen/graph/gen/dimacs.cuh>
#include <morgen/graph/gen/coo.cuh>

#include <morgen/bfs/bitmask.cu>
#include <morgen/bfs/queue.cu>
#include <morgen/bfs/hash.cu>
#include <morgen/bfs/topo.cu>
#include <morgen/bfs/hybrid.cu>
#include <morgen/bfs/serial.cu>
#include <morgen/bfs/coo_bitmask.cu>

#include <morgen/utils/stats.cuh>
#include <morgen/utils/command_line.cuh>
#include <morgen/utils/random_node.cuh>
#include <morgen/utils/timing.cuh>

using namespace morgen;


void usage() {
    printf("\ntest <graph> <bfs type> [--device=<device index>] ");
}


void check_open(FILE *fp, const char *filename) {
    if (!fp) {
        fprintf(stderr, "cannot open file: %s\n", filename);
        exit(1);
    }
}


int main(int argc, char **argv) {
    

    typedef int VertexId;
    typedef int SizeT;
    typedef int Value;


    /*********************************************************************
     * Commandline parsing
     *********************************************************************/
    util::CommandLineArgs args(argc, argv);
    // 0: prog   1: graph    2: bfs_type
    if ((argc < 1) || args.CheckCmdLineFlag("help")) {
        usage();
        return 1;
    }



    VertexId source = 0;
    std::string src_str;
    bool randomized_source = false;
    args.GetCmdLineArgument("source", src_str);
    if (src_str.compare("random") == 0) {
        randomized_source = true;
        printf("Source node:\trandomized\n");
    } else {
        args.GetCmdLineArgument("source", source);
        printf("Source node:\t%d\n", source);
    }


    // primary arguments
    std::string graph_file_path;
    args.GetCmdLineArgument("graph", graph_file_path);
    printf("Graph Path:\t%s\n", graph_file_path.c_str());


    std::string graph_file_format;
    args.GetCmdLineArgument("format", graph_file_format);
    printf("Graph Format:\t%s\n", graph_file_format.c_str());


    std::string graph_layout;
    args.GetCmdLineArgument("layout", graph_layout);
    printf("Graph Layout:\t%s\n", graph_layout.c_str());


    std::string bfs_type;
    args.GetCmdLineArgument("bfs", bfs_type);
    printf("BFS type:\t%s\n", bfs_type.c_str());


    // secondary arguments
    int block_size = 256;
    args.GetCmdLineArgument("block_size", block_size);
    printf("BLock size(threads):\t%d\n", block_size);

    int group_size = 1;
    args.GetCmdLineArgument("group_size", group_size);
    printf("Group size(threads):\t%d\n", group_size);

    int num_of_iteration = 1;
    args.GetCmdLineArgument("iteration", num_of_iteration);
    printf("num_of_iteration:\t%d\n", num_of_iteration);

    bool instrument = args.CheckCmdLineFlag("instrument");
    printf("Instrument?\t\t%s\n", (instrument ? "Yes" : "No"));


    graph::CooGraph<VertexId, SizeT, Value> ga;

    /*********************************************************************
     * Build the graph from a file
     *********************************************************************/
    FILE *fp = fopen(graph_file_path.c_str(), "r");
    check_open(fp, graph_file_path.c_str());

    if (graph_file_format == "coo") {

        if (graph::gen::cooGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;


    } else {
        fprintf(stderr, "graph format unknown\n");
        return 1;
    }


    /********************************************************************
     * Transfer the graph to GPU memory
     *********************************************************************/
     ga.transfer();


    /********************************************************************
     * Source Node Selection
     *********************************************************************/
    source =  (randomized_source ? util::randomNode(ga.n) : source % ga.n);
    printf("Traversing from %d\n", source);    


    util::CpuTimer t0;
    t0.start();

    for (int i=0; i<num_of_iteration; i++) {

    /*********************************************************************
     * Traversing
     *********************************************************************/
    if (bfs_type == "bitmask") {


        bfs::BFSGraph_gpu_bitmask_coo<VertexId, SizeT, Value>(
            ga, 
            source,
            block_size,
            instrument);


    } else {
        fprintf(stderr, "no traverse type is specified. exit quietly\n");
    }

    } // for-loop

    t0.stop();
    printf("Total:\t%f\n", t0.elapsedMillis() / 1000);
    

    fclose(fp);
    ga.del();
    return 0;
}