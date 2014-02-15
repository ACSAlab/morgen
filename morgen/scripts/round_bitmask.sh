#!/bin/bash

OUTDIR="round"

mkdir -p $OUTDIR

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
do
	echo ./bin/test $graph round 
	./bin/test $graph round | grep 'Time' >> ./$OUTDIR/$graph.out
done