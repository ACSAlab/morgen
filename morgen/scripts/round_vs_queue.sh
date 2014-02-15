#!/bin/bash

OUTDIR="round_vs_queue"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
do
	echo ./bin/test $graph queue
	./bin/test $graph queue | grep -E 'Graph|Time' >> ./$OUTDIR.out
	echo ./bin/test $graph round 
	./bin/test $graph round  | grep -E 'Graph|Time' >> ./$OUTDIR.out
done