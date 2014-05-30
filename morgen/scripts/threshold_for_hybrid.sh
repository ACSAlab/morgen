#!/bin/bash

OUTDIR="threshold_for_hybrid"

#for graph in livejournal
for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2 random3 kron
do

	for t in 32 64 96 128 256 512 1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072 262144 524288
	do 
		echo $t >> ./$OUTDIR.out
		echo ./bin/test $graph hybrid --block_size=256   --theta=$t --iteration=10 --instrument
		./bin/test $graph hybrid  --block_size=256  --theta=$t --iteration=10 --instrument | grep -E 'Expand' >> ./$OUTDIR_$graph.out
	done
done

