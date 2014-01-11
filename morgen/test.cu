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
#include <morgen/bfs/serial.cu>
#include <morgen/utils/command_line.cuh>
#include <morgen/utils/random_node.cuh>
#include <morgen/utils/utilizing_efficiency.cuh>


using namespace morgen;


void usage() {
    printf("\ntest <graph> <bfs type> [--device=<device index>] "
            "[--slots=<number of slots>] [--outdegree] [--distribution] [--workset]"
            "[--src=<source idx>] [--instrument] [--random_source] "
            "[--group_size=<group size>]\n"
            "\n"
            "<graph>\n"
            "  tiny: tiny graph for debugging\n"
            "  fla: Florida Road map\n"
            "  mesh: 6-point 2D mesh\n"
            "  kkt: Optimal power flow, nonlinear optimization (KKT)\n"
            "  copaper: CopaperCiteSeer\n"
            "  audi: symmetric rb matrix\n"
            "  rmat1: random small world graph (n=5M  m=60M)\n"
            "  rmat2: random small world graph (n=2M  n=100M)\n"
            "  amazon: Amazon product co-buying\n"
            "  random1: Erdos-Renyi or uniformly random graph (n=5M n=60M)\n"
            "  random2: Erdos-Renyi or uniformly random graph (n=2M n=100M)\n"
            "  eco: circuit theory applied to animal/gene flow\n"
            "  thermal: FEM 3D nonlinear thermal problem, 8-node bricks as volume elements\n"
            "  livejournal: LiveJournal's social network\n"
            "\n");
}


void check_open(FILE *fp, char *filename) {
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
    if ((argc < 3) || args.CheckCmdLineFlag("help")) {
        usage();
        return 1;
    }

    std::string graph = argv[1];
    std::string bfs_type = argv[2];

    printf("================================================================\n");
    printf("Graph: %s\n", graph.c_str());


    // --outdegree : print out degrees of the graph?
    bool display_outdegree = args.CheckCmdLineFlag("outdegree");
    printf("Display outdegree?   %s\n", (display_outdegree ? "Yes" : "No"));

    // --distribution : print the edge distribution each level?
    bool display_distribution = args.CheckCmdLineFlag("distribution");
    printf("Display distribution?   %s\n", (display_distribution ? "Yes" : "No"));

    // --workset :
    bool display_workset = args.CheckCmdLineFlag("workset");
    printf("Display workset?   %s\n", (display_workset ? "Yes" : "No"));

    // --metrics :
    bool display_metrics = args.CheckCmdLineFlag("metrics");
    printf("Display metrics?   %s\n", (display_metrics ? "Yes" : "No"));

    // --warp_map :
    bool warp_mapped = args.CheckCmdLineFlag("warp_map");
    printf("Warp Mapping?   %s\n", (warp_mapped ? "Yes" : "No"));

    // --instrument : whether instrument each frontier
    bool instrument = args.CheckCmdLineFlag("instrument");
    printf("Instrument?   %s\n", (instrument ? "Yes" : "No"));

    // --instrument : whether instrument each frontier
    bool random_source = args.CheckCmdLineFlag("random_source");
    printf("Random Source?   %s\n", (random_source ? "Yes" : "No"));

    // --source=<source node ID>
    VertexId source = 0;
    args.GetCmdLineArgument("source", source);
    printf("Source node: %lld\n", source);

    // --slots=<number of slots>
    int slots = 0;
    args.GetCmdLineArgument("slots", slots);
    printf("Slot number: %d\n", slots);

    // --block_size=<block size>
    int block_size = 256;
    args.GetCmdLineArgument("block_size", block_size);
    printf("BLock size: %d (threads)\n", block_size);

    int group_size = 32;
    args.GetCmdLineArgument("group_size", group_size);
    printf("Group size: %d (threads)\n", group_size);


    graph::CsrGraph<VertexId, SizeT, Value> ga;


    /*********************************************************************
     * Build the graph from a file
     *********************************************************************/
    FILE *fp;

    if (graph == "tiny") {
        fp = fopen(getenv("TINY_GRAPH"), "r");
        check_open(fp, "tiny");
        if (graph::gen::myGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "fla") {
        fp = fopen(getenv("FLA_GRAPH"), "r");
        check_open(fp, "fla");
        if (graph::gen::myGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "mesh") {
    
        fp = fopen(getenv("MESH_GRAPH"), "r");
        check_open(fp, "mesh");
        if (graph::gen::myGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "rmat1") {
    
        fp = fopen(getenv("RMAT1_GRAPH"), "r");
        check_open(fp, "rmat1");
        if (graph::gen::myGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "rmat2") {
    
        fp = fopen(getenv("RMAT2_GRAPH"), "r");
        check_open(fp, "rmat2");
        if (graph::gen::cooGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "random1") {
    
        fp = fopen(getenv("RANDOM1_GRAPH"), "r");
        check_open(fp, "random1");
        if (graph::gen::cooGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "random2") {
    
        fp = fopen(getenv("RANDOM2_GRAPH"), "r");
        check_open(fp, "random2");
        if (graph::gen::cooGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "kkt") {

        fp = fopen(getenv("KKT_GRAPH"), "r");
        check_open(fp, "kkt");
        if (graph::gen::dimacsGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "copaper") {

        fp = fopen(getenv("COPAPER_GRAPH"), "r");
        check_open(fp, "copaper");
        if (graph::gen::dimacsGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "audi") {

        fp = fopen(getenv("AUDI_GRAPH"), "r");
        check_open(fp, "audi");
        if (graph::gen::dimacsGraphGen<VertexId, SizeT, Value>(fp, ga) !=0 ) return 1;

    } else if (graph == "amazon") {

        fp = fopen(getenv("AMAZON_GRAPH"), "r");
        check_open(fp, "amazon");
        if (graph::gen::cooGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;
    
    } else if (graph == "thermal") {

        fp = fopen(getenv("THERMAL_GRAPH"), "r");
        check_open(fp, "thermal");
        if (graph::gen::dimacsGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "eco") {

        fp = fopen(getenv("ECO_GRAPH"), "r");
        check_open(fp, "eco");
        if (graph::gen::dimacsGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else if (graph == "livejournal") {

        fp = fopen(getenv("LIVE_GRAPH"), "r");
        check_open(fp, "livejournal");
        if (graph::gen::cooGraphGen<VertexId, SizeT, Value>(fp, ga) != 0) return 1;

    } else {
        fprintf(stderr, "no graph is specified\n");
        return 1;
    }

    /*********************************************************************
     * Display
     *********************************************************************/

    // Graph Information display(not verbose)
    ga.printInfo(false); 

    if (display_outdegree) 
        ga.printOutDegrees();

    if (display_distribution || display_workset) 
        bfs::BFSGraph_serial<VertexId, SizeT, Value>(
            ga,
            source, 
            instrument, 
            display_distribution,
            display_workset);


    if (display_metrics)
        util::displayUtilizingEfficiency(ga);

    /*********************************************************************
     * Traversing
     *********************************************************************/

     if (random_source) source = util::randomNode(ga.n);
     if (source > ga.n) {
        printf("Error: source %d is out of range\n", source);
        return 1;
     } 

     printf("Start traversing from %d... \n", source);

    if (bfs_type == "serial") {

        bfs::BFSGraph_serial<VertexId, SizeT, Value>(
            ga, 
            source,
            instrument,
            display_distribution,
            display_workset);

    } else if (bfs_type == "bitmask") {

        bfs::BFSGraph_gpu_bitmask<VertexId, SizeT, Value>(
            ga,
            source,
            instrument);

    } else if (bfs_type == "queue") {

        bfs::BFSGraph_gpu_queue<VertexId, SizeT, Value>(
            ga,                                            
            source,
            instrument,
            block_size,
            warp_mapped,
            group_size);

    } else if (bfs_type == "hash") {

        bfs::BFSGraph_gpu_hash<VertexId, SizeT, Value>(
            ga,
            source, 
            slots, 
            instrument);

    } else {
        fprintf(stderr, "no traverse type is specified. exit quietly\n");
    }


    fclose(fp);
    ga.del();
    return 0;
}