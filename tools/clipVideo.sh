#!/bin/bash
#	-ss 01:58.30 -to 02:39.90 \

file=`basename $1`

count=`cat /z/camera/communitycats/referenceVideos/.count`
count=$(($count+1))
echo $count > /z/camera/communitycats/referenceVideos/.count

outfile=`printf "/z/camera/communitycats/referenceVideos/%04d-$file" $count`

ffmpeg -y -hide_banner \
	-ss $2 -to $3 \
	-i $1 \
	-c copy \
	$outfile