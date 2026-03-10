#!/bin/sh
filename=$1
outdir=$2
bname=`basename $filename | sed -E 's/\..+$//g'`
if [ ! -d "$outdir" -a ! -f "$filename"  ]; then
    echo "can't find outdir or filename" 
    exit 1
fi
ffmpeg -skip_frame nokey -i $filename -vsync 0 -f image2 -qmin 2 -q:v 5 -vf "fps=1" $outdir/frame-%08d.jpg


#ffmpeg -skip_frame nokey -i my-film.mp4 -vsync 0 -f image2 stills/my-film-%06d.png
