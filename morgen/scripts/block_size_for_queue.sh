#!/bin/bash

OUTDIR="block_size_for_queue"


for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo $graph >> ./$OUTDIR.out

	echo 1 >> ./$OUTDIR.out
	for k in 32 64 128 192 256 512 1024
	do
		echo ./bin/test $graph queue --block_size=$k
		./bin/test $graph queue  --block_size=$k | grep -E 'Time' >> ./$OUTDIR.out
	done
	echo -e '\n' >> ./$OUTDIR.out

	for i in 2 4 8 16 32 64
	do
		echo $i >> ./$OUTDIR.out
		for j in 32 64 128 192 256 512 1024
		do
			echo ./bin/test $graph queue --warp_map  --group_size=$i  --block_size=$j
			./bin/test $graph queue --warp_map --group_size=$i --block_size=$j | grep -E 'Time' >> ./$OUTDIR.out
		done
		echo -e '\n' >> ./$OUTDIR.out
	done
done

