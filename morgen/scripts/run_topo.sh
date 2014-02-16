#!/bin/bash

OUTDIR="run_topo"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
do
	echo ./bin/test $graph topo 
	./bin/test $graph topo | grep 'Time' >> ./$OUTDIR.out
done