#!/bin/bash

OUTDIR="tune_alpha"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	for i in 1 2 3 4 5 6 7 8 9 10
	do
		echo ./bin/test $graph topo --block_size=256 --alpha=$i
		./bin/test $graph topo --block_size=256 --alpha=$i | grep 'Time' >> ./$OUTDIR.out
	done
done