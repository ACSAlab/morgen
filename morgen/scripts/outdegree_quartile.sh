#!/bin/bash


OUTDIR="outdegree_quartile"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do

	echo ./bin/test $graph nil
	./bin/test $graph nil | grep -E 'Graph|Quartiles' >> $OUTDIR.out

done