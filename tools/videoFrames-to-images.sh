#!/bin/sh
filename=`basename $1 | sed -E 's/\..+$//g'`
if [ ! -d "custom_data/images/$filename" ]; then
	mkdir -p custom_data/images/$filename
fi
ffmpeg -i $1 -vf fps=3  custom_data/images/$filename/frame-%08d.jpg

