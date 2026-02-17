#!/bin/bash

for file in `ls -1 | sed -E 's/\..*//g' | sort | uniq  | sort -R | head -n 250`; do
    mv -v $file.txt /Users/dpd/Documents/projects/github/catDetectionSystem/dataset/images-cds/
    mv -v $file.xml /Users/dpd/Documents/projects/github/catDetectionSystem/dataset/images-cds/
    mv -v $file.jpg /Users/dpd/Documents/projects/github/catDetectionSystem/dataset/images-cds/
done


