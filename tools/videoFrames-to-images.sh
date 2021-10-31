#!/bin/sh
filename=`basename $1 | sed -E 's/\..+$//g'`
if [ ! -d "/z/camera/communitycats/custom_data/images/$filename" ]; then
	mkdir -p /z/camera/communitycats/custom_data/images/$filename
fi
ffmpeg -skip_frame nokey -i $1 -vsync 0 -f image2 -qmin 2 -q:v 5 /z/camera/communitycats/custom_data/images/$filename/frame-%08d.jpg

#ffmpeg -skip_frame nokey -i my-film.mp4 -vsync 0 -f image2 stills/my-film-%06d.png
