#!/bin/bash

OUTDIR="threshold_for_topo"

#for graph in livejournal
for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do

	#for t in 32 64 96 128 192 256 320 384 448 512 640 768 896 1024 1280 1536 1792 2048 4096 8192 16384 32768
	for t in 32 64 96 128 256 512 1024 2048 3072 4096 6144 8192 16384 32768 65536
	do 
		echo $t >> ./$OUTDIR.out
		echo ./bin/test $graph topo --block_size=256 --threshold=$t
		./bin/test $graph topo  --block_size=256 --threshold=$t | grep -E 'Time' >> ./$OUTDIR_$graph.out
	done
done

