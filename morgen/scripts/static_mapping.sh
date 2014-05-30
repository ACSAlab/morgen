#!/bin/bash

OUTDIR="static_mapping"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2 random3 kron
do


	for i in 1 2 4 8 16 32
	do
		echo ./bin/test $graph queue  --group_size=$i --block_size=256 --iteration=10 --instrument
		./bin/test $graph queue  --group_size=$i --block_size=256 --iteration=10 --instrument | grep -E 'Expand' >> ./$OUTDIR_$graph.out
	done
done