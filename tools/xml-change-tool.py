import simplexml
import cv2
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
parser.add_argument('--re', help='RegEx to filter files by')
parser.add_argument('--changeto', help='Change then Name/class.')

args, _ = parser.parse_known_args()


def printLabels(videoStreamType="file"):
    labels = {
                'frame': "frame",
                'class': "class",
                'conf': "conf", 
                'x':"x", 
                'y':"y", 
                'w':"w", 
                'h':"h",
                "catCounter": "consecCatFr",
                "catTimer": "sec/meow",
                "date": "date",
                "percent": "%",
                'area': "area",
                'pix': "pix",
                'filename': "filename",
                'changeto': "changeto"
                }
    if videoStreamType == "file":
        print( "\n\n%(filename)32s %(class)16s %(changeto)16s %(x)6s %(y)6s %(w)6s %(h)6s %(pix)12s %(area)12s %(percent)5s \n" %  labels )

def printDataLine(videoStreamType="file", d={} ):
    if videoStreamType == "file":
        print( "%(filename)32s %(class)16s %(changeto)16s %(x)6d %(y)6d %(w)6d %(h)6d %(pix)12d %(area)12d     %(percent)3.2f" % d )


# classhash = {}
# classcounts = {}
# i = 0
# for c in classNames:
#     classhash[c] = i
#     classcounts[c] = 0
#     i += 1

xmlfiles = []
for root, dirs, files in os.walk(args.dir, topdown=False):
    if dirs == ".DS_Store":
        continue
    for name in files:
        if re.search("\.xml$", name):
            try:
                if re.search(args.re, name):
                    xmlfiles.append(os.path.join(root, name))
            except:
                xmlfiles.append(os.path.join(root, name))
xmlfiles.sort()                
printLabels()
for xmlfile in xmlfiles:
    f = open(xmlfile,'r')
    xml = simplexml.loads( f.read() )
    f.close()
    #pp.pprint(xml)

    # <object-class-id> <x-centre> <y-centre> <width> <height>
    # <object-class-id> an integer from 0 to (classes - 1) corresponding to the classes in the custom_data/custom.names file
    # height, width - Actual height and width of the image
    # x, y - centre coordinates of the bounding box
    # h, w - height and width of the bounding box
    # <x-centre> : x / width
    # <y-centre> : y / height
    # <width> : w / width
    # <height> : h / height
    
    try:
        height = int(xml['annotation']['size']['height'])
        width = int(xml['annotation']['size']['width'])
    except:
        #img = cv2.imread(jpgfile)
        #(height, width) = img.shape[:2]        
        pass
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
    i = 0
    for o in objects:
        #pp.pprint (o)
        h = int(o['bndbox']['ymax']) - int(o['bndbox']['ymin'])
        w = int(o['bndbox']['xmax']) - int(o['bndbox']['xmin'])
        area = w*h
        data = {
                    'filename': os.path.basename(xmlfile),
                    'frame': 0,
                    'class': o["name"],
                    'conf': 0, 
                    'x':  int(o['bndbox']['xmin']), 
                    'y':  int(o['bndbox']['ymin']),
                    'w': w, 
                    'h': h,
                    "catCounter": 0,
                    "catTimer": 0,
                    "date": 0,
                    "percent": 0,
                    'area': area,
                    'pix': 0,
                    }
        try:
            data['changeto'] = args.changeto
        except:
            pass
        try:
            if i == 0:
                xml['annotation']['object']["name"] = args.changeto
            else:
                xml['annotation']['object'][i]["name"] = args.changeto
        except:
            pass
        i += 1
        printDataLine(d=data)

    f = open(xmlfile,'w')
    f.write(simplexml.dumps(xml))
    f.close
        


