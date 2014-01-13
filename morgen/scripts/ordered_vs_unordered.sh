#!/bin/bash



for exe in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo ./bin/test $exe queue 
	./bin/test $exe queue | grep -E 'Graph|Unorderd|Time' >> ordered_vs_unordered.out
	echo ./bin/test $exe queue --unordered
	./bin/test $exe queue --unordered | grep -E 'Graph|Unorderd|Time' >> ordered_vs_unordered.out
done