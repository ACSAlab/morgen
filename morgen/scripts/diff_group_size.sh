#!/bin/bash



for exe in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	for i in 32 16 8 4 2
	do
		echo ./bin/test $exe queue --warp_map --group_size=$i
		./bin/test $exe queue --warp_map --group_size=$i | grep -E 'Graph|Group size|Time' >> diff_group_size.out
	done
done