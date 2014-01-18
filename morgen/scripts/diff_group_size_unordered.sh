#!/bin/bash



for exe in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	for i in 32 16 8 4 2
	do
		echo ./bin/test $exe queue --warp_map --group_size=$i --unordered
		./bin/test $exe queue --warp_map --group_size=$i --unordered | grep -E 'Graph|Group size|Time' >> diff_group_size_unordered.out
	done
done