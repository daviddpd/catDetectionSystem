#!/bin/sh
# filename="referenceVideos-images"
# if [ ! -d "/Volumes/camera/communitycats/referenceVideos/$filename" ]; then
# 	mkdir -p /Volumes/camera/communitycats/referenceVideos/$filename
# fi
# 
# 
# for f in `/bin/ls -1 /Volumes/camera/communitycats/referenceVideos/*.mp4`; do
#     orgfile=`basename $f | sed -E 's/\..+$//g'`
#     echo "$f $filename/$orgfile "
#     ffmpeg -hide_banner -y -skip_frame nokey -i $f -vsync 0 -f image2 -qmin 2 -q:v 5 /Volumes/camera/communitycats/referenceVideos/${filename}/${orgfile}-%08d.jpg
# done

#ffmpeg -skip_frame nokey -i my-film.mp4 -vsync 0 -f image2 stills/my-film-%06d.png



basepath="/Volumes/camera/uploads/c2/2023/09"
refimages_dir="referenceVideos-images"
if [ ! -d "$basepath/$refimages_dir" ]; then
	mkdir -p $basepath/$refimages_dir
fi


for f in `find ${basepath} -name "*.mp4"`; do
    orgfile=`basename $f | sed -E 's/\..+$//g'`
    echo "$f $filename/$orgfile "
    ffmpeg -hide_banner -y -skip_frame nokey -i $f -f image2 -r 1/3 $basepath/$refimages_dir/${orgfile}-%08d.jpg
done
