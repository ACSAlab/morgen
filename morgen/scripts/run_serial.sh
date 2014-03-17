#!/bin/bash

OUTDIR="run_serial"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo ./bin/test $graph seial
	./bin/test $graph serial | grep -E 'Time' >> ./$OUTDIR.out
done