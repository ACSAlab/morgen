#!/bin/bash

OUTDIR="block_size_for_queue"


for j in 16 32 64 96 128 160 192 224 256 288 320
do
	for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
	do
		echo ./bin/test $graph queue --block_size=$j
		./bin/test $graph queue --block_size=$j | grep -E 'Time' >> ./$OUTDIR.out
	done
done