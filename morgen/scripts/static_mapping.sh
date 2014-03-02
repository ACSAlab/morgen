#!/bin/bash

OUTDIR="static_mapping"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do

	echo ./bin/test $graph queue  --block_size=128
	./bin/test $graph queue --block_size=128 | grep -E 'Time' >> ./$OUTDIR_$graph.out

	for i in 2 4 8 16 32
	do
		echo ./bin/test $graph queue --warp_map --group_size=$i --block_size=128
		./bin/test $graph queue --warp_map --group_size=$i --block_size=128 | grep -E 'Time' >> ./$OUTDIR_$graph.out
	done
done