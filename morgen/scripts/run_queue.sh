#!/bin/bash

OUTDIR="run_queue"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
do
	echo ./bin/test $graph queue
	./bin/test $graph queue | grep -E 'Graph|Time' >> ./$OUTDIR.out
done