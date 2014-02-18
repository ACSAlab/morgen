#!/bin/bash

OUTDIR="metrics_for_queue"


for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo $graph >> ./$OUTDIR.out

	echo 1 >> ./$OUTDIR.out

	echo ./bin/test $graph queue --metrics
	./bin/test $graph queue  --metrics | grep -E 'metric' >> ./$OUTDIR.out

	echo -e '\n' >> ./$OUTDIR.out

	for i in 2 4 8 16 32
	do
		echo $i >> ./$OUTDIR.out

		echo ./bin/test $graph queue --warp_map  --group_size=$i  --metrics
		./bin/test $graph queue --warp_map --group_size=$i --metrics | grep -E 'metric' >> ./$OUTDIR.out
		
		echo -e '\n' >> ./$OUTDIR.out
	done
done

