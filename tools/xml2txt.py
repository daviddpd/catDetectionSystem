import simplexml
import cv2 as cv
import argparse
import numpy as np
import sys
import os
import time
import pprint
import re
import uuid
import hashlib
import shutil
 
def get_checksum(filename, hash_function):
    hash_function = hash_function.lower() 
    with open(filename, "rb") as f:
        bytes = f.read()  # read file as bytes
        if hash_function == "md5":
            readable_hash = hashlib.md5(bytes).hexdigest()
        elif hash_function == "sha256":
            readable_hash = hashlib.sha256(bytes).hexdigest()
        else:
            Raise("{} is an invalid hash function. Please Enter MD5 or SHA256")
    return readable_hash
    
pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dir', help='directory to scane for XML files')
parser.add_argument('--classes', help='text file of strings of the class names')
#parser.add_argument('--versionstr', help='Version/name of the training set')

args, _ = parser.parse_known_args()

#cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input) if args.input else 0, cv.CAP_FFMPEG)

classFile = args.classes
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

classhash = {}
classcounts = {}
i = 0
for c in classNames:
    classhash[c] = i
    classcounts[c] = 0
    i += 1

xmlfiles = []
for root, dirs, files in os.walk(args.dir, topdown=False):
   for name in files:
      if re.search("\.xml$", name):
          #print(os.path.join(root, name))
          xmlfiles.append(os.path.join(root, name))
          
          
#pp.pprint (xmlfiles)

testfiles = []
jpgfiles = [] 
trainFiles = []

for xmlfile in xmlfiles:
    f = open(xmlfile,'r')
    xml = simplexml.loads( f.read() )
    f.close()

    txtfile = xmlfile.replace("xml", "txt")
    jpgfile = xmlfile.replace("xml", "jpg")
    #print ("txt file:  " + txtfile + "\n")
    testfiles.append(txtfile)
    jpgfiles.append(jpgfile)

    # <object-class-id> <x-centre> <y-centre> <width> <height>
    # <object-class-id> an integer from 0 to (classes - 1) corresponding to the classes in the custom_data/custom.names file
    # height, width - Actual height and width of the image
    # x, y - centre coordinates of the bounding box
    # h, w - height and width of the bounding box
    # <x-centre> : x / width
    # <y-centre> : y / height
    # <width> : w / width
    # <height> : h / height
    
    height = int(xml['annotation']['size']['height'])
    width = int(xml['annotation']['size']['width'])
    data = ""
    objects = []
    try: 
        if isinstance(xml['annotation']['object'], list):
            objects = xml['annotation']['object']
        else:
            objects.append( xml['annotation']['object'] )
            #pp.pprint (objects)
    except KeyError:
        continue
    
    imageClasses = {}

    data = ""
    for o in objects:
        #pp.pprint (o)
        h = int(o['bndbox']['ymax']) - int(o['bndbox']['ymin'])
        w = int(o['bndbox']['xmax']) - int(o['bndbox']['xmin'])
        x = w/2 + int(o['bndbox']['xmin'])
        y = h/2 + int(o['bndbox']['ymin'])
        try:
            c = classhash[o['name']]
            classcounts[o['name']] += 1
            try:
                imageClasses[o['name']] +=1
            except KeyError:
                imageClasses[o['name']] = 1
        except KeyError:
            continue
        data  += "{} {} {} {} {}\n".format( c,  x/width, y/height,  w/width,  h/height)
        if re.search("^cat-.*", o['name']):
            c = classhash["cat"]    
            data  += "{} {} {} {} {}\n".format( c,  x/width, y/height,  w/width,  h/height)
            classcounts["cat"] += 1
    if len(data) > 0:
        #print (" === " + data)
        f = open(txtfile,'w')
        f.write(data)
        f.close
        trainFiles.append(jpgfile)
    imageAminial = ""
    AminialInt = -1
    for ic in imageClasses:
        i = classhash[ic]
        if ( i > AminialInt ):
            imageAminial = ic
            AminialInt = i            
    u = uuid.uuid1()    
    cs = get_checksum(jpgfile, "md5")
    # MD5-5c05-f7ca-cd5d-940f1d76f6-95e8-8ee79d
    # MD5-0123 4567 8901 2345678901 2345 678901
    #                 1          2           3    
    csfmt = ("{}-{}-{}-{}-{}-{}").format( cs[0:4],cs[4:8],cs[8:12],cs[12:22],cs[22:26],cs[26:32]  )    
    print ( " %-80s : classes ( %1d ) ( %-16s ) ( %-38s )  :  %s" % (xmlfile, len(imageClasses), imageAminial, csfmt, imageClasses)) 
    bn = "custom_data/imagebyclass/" + imageAminial + "/" + csfmt
    if len(imageAminial) > 0:
        if not os.path.isfile(bn + ".jpg"):
            try:
                if not os.path.isfile(bn + ".txt"):
                    shutil.copy2( txtfile, bn + ".txt" )
                if not os.path.isfile(bn + ".jpg"):
                    shutil.copy2( jpgfile, bn + ".jpg" )
                if not os.path.isfile(bn + ".xml"):
                    shutil.copy2( xmlfile, bn + ".xml" )
            except FileNotFoundError:
                pass
# /z/camera/communitycats/custom_data/images/imagebyclass
#pp.pprint ( jpgfiles )
#pp.pprint ( testfiles )
pp.pprint (classcounts)


#f = open("custom_data/jpgfiles.txt",'w')
#for data in jpgfiles:
#   f.write(data + "\n")
#f.close


#custom_data/images/C3-Oct-0000/C3-2021-Oct-0000-frame-00000092.xml

# f = open("custom_data/train.txt",'w')
# for data in trainFiles:
#     f.write(data + "\n")
# f.close
# 
# 
