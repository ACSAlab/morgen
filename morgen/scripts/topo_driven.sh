#!/bin/bash

OUTDIR="topo"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
do
	echo ./bin/test $graph topo --intrument 
	./bin/test $graph topo --instrument | grep 'slot' >> ./$OUTDIR/$graph.out
done