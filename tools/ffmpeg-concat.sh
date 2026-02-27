#!/bin/sh
/bin/ls -1 *.mp4 *.mkv | xargs -I % echo "file '%'" >> filelist.txt
ffmpeg -hide_banner -hwaccel videotoolbox -y  -f concat -safe 0 \
    -i "filelist.txt" -c:v hevc_videotoolbox -an -b:v 1250k -tag:v hvc1 "all.mp4"