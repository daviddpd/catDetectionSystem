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
from PIL import Image
 
 
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


def move_into(filename, directory_path):
    # if directory_path.is_dir():
    # Check if path exists: os.path.exists("path/to/file")
    # Check if it is a file: os.path.isfile("path/to/file") 
    if not os.path.isdir(directory_path):
        try:
            os.mkdir(directory_path)
            if args.verbose:
                print(f"Directory '{directory_path}' created.")
        except FileExistsError:
            if args.verbose:
                print(f"Directory '{directory_path}' already exists.")
        except FileNotFoundError:
            if args.verbose:
                print("Parent directory does not exist.")
        except OSError as e:
            if args.verbose:
                print(f"An OS error occurred: {e}")
    else:
        try:
            # Move the file
            shutil.move(filename, directory_path)
            if args.verbose:
                print(f"Moved '{filename}' to '{directory_path}'")
        except FileNotFoundError:
            if args.verbose:
                print(f"Error: Source or destination path not found")
        except Exception as e:
            if args.verbose:
                print(f"An error occurred: {e}")


pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dir', help='Directory to scan for XML files.')
parser.add_argument('--re', help='RegEx to filter files by')
parser.add_argument('--changeName', action="store_true", help='enable change name')
parser.add_argument('--matchre', help='Regex to the data to')
parser.add_argument('--changeto', help='string to replace it with')
parser.add_argument('--move', action="store_true", help='Move into class named directory')
parser.add_argument('--verbose', action="store_true", help='Verbose')


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
imageClasses = {}
outliers = []
classes = ["cat", "dog", "person", "opossum", "raccoon", "skunk"]

for xmlfile in xmlfiles:
    f = open(xmlfile,'r')
    xml = simplexml.loads( f.read() )
    f.close()
    jpgfile = re.sub(r'\.(xml)$', ".jpg", xmlfile )
    txtfile = re.sub(r'\.(xml)$', ".txt", xmlfile )
    
# 
#     # <object-class-id> <x-centre> <y-centre> <width> <height>
#     # <object-class-id> an integer from 0 to (classes - 1) corresponding to the classes in the custom_data/custom.names file
#     # height, width - Actual height and width of the image
#     # x, y - centre coordinates of the bounding box
#     # h, w - height and width of the bounding box
#     # <x-centre> : x / width
#     # <y-centre> : y / height
#     # <width> : w / width
#     # <height> : h / height
# 
#     try:
#         image = Image.open(jpgfile)
#     except FileNotFoundError:
#         continue
#     
#     try:
#         height = int(xml['annotation']['size']['height'])
#         width = int(xml['annotation']['size']['width'])
#     except:
#         #img = cv2.imread(jpgfile)
#         #(height, width) = img.shape[:2]
#         xml['annotation']['size'] = {}
#         xml['annotation']['size']['height']  = image.height
#         xml['annotation']['size']['width']   = image.width
#         height = image.height
#         width  = image.width
# 
#     try:
#         xml['annotation']['imagefilename'] = os.path.basename(jpgfile)
#         xml['annotation']['filename'] = os.path.basename(jpgfile)
#         mydir = os.path.dirname(jpgfile)
#         xml['annotation']['path'] = mydir
#         xml['annotation']['folder'] = os.path.basename(mydir)
#     except:
#         pass
#         
#     data = ""
    objects = []

    try: 
        if isinstance(xml['annotation']['object'], list):
            objects = xml['annotation']['object']
        else:
            objects.append( xml['annotation']['object'] )
        #print( "== Objects ================================================================" )
        #pp.pprint (objects)
    except KeyError:
        continue
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
                    'changeto': None 
                    }
        try:
            imageClasses[o["name"]] = imageClasses[o["name"]] + 1
        except:
            imageClasses[o["name"]] = 1

        if o["name"] not in classes:
            outliers.append(xmlfile)


        if args.move is True:
            dest = os.path.join(args.dir, o["name"])
            move_into(xmlfile, dest)
            move_into(jpgfile, dest)
            move_into(txtfile, dest)
            
        try:
            data['changeto'] = args.changeto
        except:
            pass
        if args.changeName is True:
            try: 
                is_match = re.search(args.matchre, o['name'], flags=0)
                if is_match is not None:
                    o['name'] = args.changeto
                    #pp.pprint(o['name'])
            except KeyError:
                continue
        printDataLine(d=data)
    
    if args.changeName is True:
        f = open(xmlfile,'w')
        f.write(simplexml.dumps(xml))
        f.close

pp.pprint(imageClasses)
pp.pprint(outliers)


