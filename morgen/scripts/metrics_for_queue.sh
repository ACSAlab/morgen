#!/bin/bash

OUTDIR="metrics_for_queue"


for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2 random3 kron
do


	echo -e '\n' >> ./$OUTDIR.out

	for i in 1 2 4 8 16 32
	do
		echo $i >> ./$OUTDIR.out

		echo ./bin/test $graph queue --group_size=$i  --instrument
		./bin/test $graph queue --group_size=$i --instrument | grep -E 'metric' >> ./$OUTDIR.out
		
		echo -e '\n' >> ./$OUTDIR.out
	done
done

