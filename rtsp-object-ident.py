import cv2
import simplexml
import argparse
import time
import os
import pprint
import re
import numpy as np
import pygame
import queue
import threading
import signal
import sys
import ffmpeg
import random
from pathlib import Path




def excludeClass(s):
    try:
        o = {}
        o['class'], o['low'], o['high'] = s.split(',')
        o['low'] = float(o['low'])
        o['high'] = float(o['high'])
        return o
    except:
        raise argparse.ArgumentTypeError("Must be ClassName,float,float; where floats are between 0 and 1.")


pp = pprint.PrettyPrinter(indent=4)
_pid  = os.getpid()
_name = sys.argv[0]
_SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--uri', help='Path to a video file or RTSP stream.', required=True)

parser.add_argument('--ratein', help='Fetch rate limit, frame rate denominator (1/VALUE).', default=0, type=int)

parser.add_argument('--repoPath', help='Repository path.', default=str(_SCRIPT_DIR))
parser.add_argument('--communityCatsPath', help='Community cats data directory.', default="/Volumes/camera/communitycats")

parser.add_argument('--writePath', help='Path to save images.', default="")
parser.add_argument('--writeImages', help='Write images with objects detected.',  default=False, action='store_true')
parser.add_argument('--writeXmlOnly', help='Write XML with objects detected in the same directory as the source.',  default=False, action='store_true')

parser.add_argument('--writeImagesNotCats', help='Only write non-cat frames.',  default=False, action='store_true')
parser.add_argument('--excludeClass', help="excludeClass,%,%", type=excludeClass, nargs=1, action='extend')

parser.add_argument('--name', help='Window name.', default=_name + ":" + str(_pid) )
parser.add_argument('--conf', help='Confidence threshold.', default=0.9, type=float)
parser.add_argument('--confcutoff', help='Confidence upper cutoff.', default=1.1, type=float)
parser.add_argument('--nms', help='NMS threshold.', default=0.5 , type=float)
parser.add_argument('--scalefactor', help='Scale factor denominator (1/VALUE).', default=300, type=int)
parser.add_argument('--scaleimg', help='Scale image.',  default=False, action='store_true')
parser.add_argument('--areaReject', help='Reject match if pixel area percent is greater than this value.', default=101.00 , type=float)
parser.add_argument('--backframes', help='Track last X number of frames.', default=10, type=int)
parser.add_argument('--model', help='Model name bundle.', default="yolo.416v6.64" )

parser.add_argument('--quiet', help='Turn off text object printing.',  default=False, action='store_true')

parser.add_argument('--nomeow', help='Turn off meow.',  default=False, action='store_true')
parser.add_argument('--shuffle', help='Shuffle file list.',  default=False, action='store_true')
parser.add_argument('--nodotfiles', help='Ignore dot files in directories.',  default=True, action='store_false')
parser.add_argument('--wait', help='Wait for key press (set waitkey=0).',  default=False, action='store_true')

parser.add_argument('--arm', help='Run on ARM64.',  default=False, action='store_true')



args, _ = parser.parse_known_args()

framesIn  = queue.Queue(30)
framesOut = queue.Queue(30)
framesToWrite = queue.Queue(300)
classNames = []
_sound_filename = "Meow-cat-sound-effect.mp3"
_sound_candidates = (
    _SCRIPT_DIR / "assets" / _sound_filename,
    _SCRIPT_DIR / "assests" / _sound_filename,
)
soundMeow = str(next((p for p in _sound_candidates if p.is_file()), _sound_candidates[-1]))
_pygame_noSound = False
if not args.nomeow:
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(soundMeow)
    except:
        _pygame_noSound = True
        pass
else:
    _pygame_noSound = True
    
    

cw = {}
cw["yolo.416v3.64"] = {}
cw["yolo.416v3.64"]["configPath"] = args.repoPath + "/yolo/cfg/yolov-tiny-custom-416v3-64.cfg"
cw["yolo.416v3.64"]["weightsPath"] = args.communityCatsPath + "/custom_data/backup/yolov-tiny-custom-416v3-64_final.weights"
cw["yolo.416v3.64"]["coconames"] = args.repoPath + "/yolo/cfg/custom-names-7v3.txt"

cw["yolo.416v4.64"] = {}
cw["yolo.416v4.64"]["configPath"] = args.repoPath + "/yolo/cfg/yolov-tiny-custom-416v4-64.cfg"
cw["yolo.416v4.64"]["weightsPath"] = args.communityCatsPath + "/custom_data/backup/yolov-tiny-custom-416v4-64_final.weights"
cw["yolo.416v4.64"]["coconames"] = args.repoPath + "/yolo/cfg/custom-names-v4.txt"

cw["yolo.416v5.64"] = {}
cw["yolo.416v5.64"]["configPath"] = args.repoPath + "/yolo/cfg/yolov-tiny-custom-416v5-64.cfg"
cw["yolo.416v5.64"]["weightsPath"] = args.communityCatsPath + "/custom_data/backup/yolov-tiny-custom-416v5-64_final.weights"
#cw["yolo.416v5.64"]["weightsPath"] = args.communityCatsPath + "/custom_data/backup/yolov-tiny-custom-416v5-64_last.weights"
cw["yolo.416v5.64"]["coconames"] = args.repoPath + "/yolo/cfg/custom-names-v4.txt"


cw["yolo.416v6.64"] = {}
cw["yolo.416v6.64"]["configPath"] = args.repoPath + "/yolo/cfg/yolov-tiny-custom-416v6-64.cfg"
cw["yolo.416v6.64"]["weightsPath"] = args.communityCatsPath + "/custom_data/backup/yolov-tiny-custom-416v6-64_final.weights"
#cw["yolo.416v6.64"]["weightsPath"] = args.communityCatsPath + "/custom_data/backup/yolov-tiny-custom-416v6-64_last.weights"
cw["yolo.416v6.64"]["coconames"] = args.repoPath + "/yolo/cfg/custom-names-v4.txt"


cw["yolo.9k"] = {}
cw["yolo.9k"]["configPath"] = "/home/dpd/darknet/cfg/yolo9000.cfg"
cw["yolo.9k"]["weightsPath"] = "/home/dpd/yolo-9000/yolo9000-weights/yolo9000.weights"
cw["yolo.9k"]["coconames"] = "/home/dpd/darknet/data/9k.names"

pkgVer = args.model

classNamesToIds = {}

with open(cw[pkgVer]["coconames"], "r") as f:
    classes = [line.strip() for line in f.readlines()]

colorsPencils = {}
colorsPencils['lime'] = (0, 250, 142)
colorsPencils['maraschino'] = (0, 38, 255)
colorsPencils['tangerine'] = (0, 147, 255)
colorsPencils['lemon'] = (0, 251, 255)
colorsPencils['blueberry'] = (255, 51, 4)
colorsPencils['strawberry'] = (146, 47, 255)
colorsPencils['stawberry'] = colorsPencils['strawberry']  # Legacy key compatibility
colorsPencils['snow'] = (255, 255, 255)
colorsPencils['lead'] = (33, 33, 33)
colorsPencils['turquoise'] = (255, 253, 0)

colors = [] 
colors = (colorsPencils['turquoise'],
          colorsPencils['tangerine'],
          colorsPencils['lime'],
          colorsPencils['maraschino'],
          colorsPencils['strawberry'],
          colorsPencils['lemon'],
          colorsPencils['blueberry'],
          colorsPencils['lead'],
          colorsPencils['lead'],
          colorsPencils['lead']
         )

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def getObjects(imgObject, net, confThres, confThresCutOff, nmsThersh, scaleFactor=1/300, netSize=(416,416), frameCounter=0, lastIndexes=[]):
    img = imgObject["image"]
    layers = net.getLayerNames()
#    pp.pprint(layers)
#    pp.pprint(net.getUnconnectedOutLayers())

    i = net.getUnconnectedOutLayers()        
    output_layers = [layers[i[0] - 1]]
    data = []
    mergeIndexes = []
    height, width = img.shape[:2]
    pix = height*width
    blob = cv2.dnn.blobFromImage(img, scaleFactor, netSize, swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    indices, class_ids, confidences, b_boxes = [], [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThres and confidence < confThresCutOff:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    try:
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, confThres, nmsThersh).flatten().tolist()
    except AttributeError:
        indices = []        
    try:
        test1 = indices[0]        
        mergeIndexes.append( { "indices": indices,
                               "b_boxes": b_boxes , 
                               "confidences": confidences, 
                               "class_ids": class_ids,
                               "frameCounter": frameCounter
                              } 
                            )
    except IndexError:
        mergeIndexes.append ( 
                               {   'b_boxes': [[0,0,0,0]],
                                  'class_ids': [0],
                                  'confidences': [0],
                                  'frameCounter': frameCounter,
                                  'indices': [-1]
                               }
                            )
    i=0
    mIdx = args.backframes
    ih, iw, ic = img.shape
    is_cat = False
    mergeIndexes += lastIndexes
    lastIndexes = []
    for indexObject in mergeIndexes:
        class_ids, confidences, b_boxes = [], [], []
        class_ids    = indexObject["class_ids"]
        confidences  = indexObject["confidences"]
        b_boxes      = indexObject["b_boxes"]
        for index in indexObject["indices"]:
            if mIdx > 0:
                lastIndexes.append(indexObject)
                mIdx=mIdx-1
            x=0
            y=0
            w=200
            h=200
            area = 0
            className = "NotSet"
            conf = "0.00"
            is_empty = False
            _draw_color = colorsPencils['lead']
            percent = 0.0
            if index == -1:
                try:
                    x, y, w, h = b_boxes[0]
                    className = " - "
                    classId = -1
                    is_empty = True
                    _draw_color = colorsPencils['lead']
                    conf = "0.00"
                except IndexError as e:
                    print("IndexError, getObjects( index == -1 ):", e);
            else:
                try:
                    x, y, w, h = b_boxes[index]
                    className = classes[class_ids[index]]
                    classId = class_ids[index]
                    colorId = classId
                    if classId > len(colors):
                        colorId = classId % len(colors)
                    _draw_color = colors[colorId]
                    area = w*h
                    percent = round(area/pix*100, 2)
                    if percent > args.areaReject:
                        x, y, w, h = [-1,-1,-1,-1]
                        className = " - "
                        classId = -1
                        is_empty = True
                        _draw_color = colorsPencils['lead']
                except IndexError as e:
                    print("IndexError, getObjects (else):", e,index,classId,colorId);
            try:
                conf = str(round(confidences[index]*100,2))
            except IndexError as e: 
                print("IndexError:getObjects list index out of range index:{} ".format(index), e)
                print("class_ids: ", class_ids )
                print("confidences", confidences)
            try:                
                if not is_empty: 
                    _draw_color = _draw_color
                    try:
                        cv2.rectangle(img, (x, y), (x + w, y + h), _draw_color, 2)
                    except Exception as e:
                        print (x,y,w,h,_draw_color)
                        raise e
                    labelx = x + 10
                    if re.search("^cat-", className):
                        labelx = int(x + w/2 + 10)
                    if re.search("^cat", className):
                        is_cat = True
                    data.append( {
                                'frame': frameCounter,
                                'class': className,
                                'conf': conf, 
                                'x':x, 
                                'y':y, 
                                'w':w, 
                                'h':h,
                                'area': area,
                                'pix': pix,
                                'percent': percent,
                                "is_cat": is_cat,
                                "date": imgObject["time"]
                                }
                            )
                    if i == 0:
                        cv2.putText(img, className, (labelx, y - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, _draw_color, 2)
                        cv2.putText(img, conf, (labelx, y - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, _draw_color, 2)            
                cv2.putText(img, className, ( 125, 75+(35*i)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, _draw_color, 2)
                cv2.putText(img, conf, (20, 75+(35*i)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, _draw_color, 2)
                cv2.putText(img, "( " + str(indexObject["frameCounter"]) + " )", ( 400, 75+(35*i)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, _draw_color, 2)
                i += 1
            except IndexError:
                pass
    return img, is_cat, data, lastIndexes

def queueFrames(quitEvent, videoObject, videoQueueLoop):
    time.sleep(5)
    v = videoObject['videoFile']
    vc = videoObject['comment']
    vidImgFiles = []
    videoStreamType = "file"
    if os.path.isdir(v):
        for root, dirs, files in os.walk(v, topdown=False):
            if dirs == ".DS_Store":
                continue
            for name in files:
                if re.search("^\.", name) and args.nodotfiles:
                    continue
                elif re.search("\.(jpg|jpeg|mkv|mp4)$", name):
                  vidImgFiles.append(os.path.join(root, name))
    else:
      vidImgFiles.append(v)            
    if args.shuffle:
        random.shuffle(vidImgFiles)
    else:
        vidImgFiles.sort(reverse=True)
    pp.pprint ( vidImgFiles )
#    print ("Number of Files: %d" % len(vidImgFiles))
    if len(vidImgFiles) == 0 and videoStreamType != "rtsp":
        quitEvent.set()
    while not quitEvent.is_set():
            if videoStreamType != "rtsp":
                try:
                    v = vidImgFiles.pop()
#                    print ("Number of Files: %d" % len(vidImgFiles))
                except IndexError:
                    while framesIn.qsize() > 0 and framesOut.qsize() > 0:
                        time.sleep(1/30)
                    videoQueueLoop.set()
                    quitEvent.set()
                    return
            framesIn.put({ "noimage": True, "path": v })                    
            if re.search("^rtsp", v):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|reorder_queue_size;200|buffer_size;1048576|hwaccel;h264_videotoolbox"
                videoStreamType = "rtsp"
                #print("stream type (1) : %s" % ( videoStreamType) )
            else:
#                vid = ffmpeg.probe(v)
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
                videoStreamType = "video"
#                 for stream in vid['streams']:
#                     #print ("Opening video %s code:[%s][%s] " % ( v,stream['codec_type'], stream['codec_name']))
#                     if  stream['codec_type'] == "video" and not args.arm: 
#                         videoStreamType = "video"
#                         if stream['codec_name'] == "h264":
#                             os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
#                         elif stream['codec_name'] == "hevc" or stream['codec_name'] == "h265":
#                             os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;hevc_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
#                         elif stream['codec_name'] == "png" or stream['codec_name'] == "mjpeg" or stream['codec_name'] == "jpeg":
#                             os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
#                             videoStreamType = "image"
#                         else: 
#                             os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
#                             videoStreamType = "file"
#                     else:
#                         os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
#                         videoStreamType = "video"
                        
            #print("stream type : %s" % ( videoStreamType) )
            #print("Openning stream : %s" % (v) )
            if videoStreamType == "video" or videoStreamType == "rtsp":
                cap = cv2.VideoCapture(v,cv2.CAP_FFMPEG)
            videoQueueLoop.clear()
            failures = 0
            while not videoQueueLoop.is_set():
                if videoStreamType == "video" or videoStreamType == "rtsp":
                    success, img = cap.read()
                elif videoStreamType == "image":
                    img = cv2.imread(v)
                    success = True
                    videoQueueLoop.set()
                t = time.localtime()
                strTime = time.strftime("%Y-%m-%d %H:%M:%S %Z", t)
                if success:
                    framesIn.put({ "image": img, "time": strTime, "videoStreamType": videoStreamType, "path": v, "noimage": False })
                else:
                    #print("queueFrames: failure")
                    failures+=1
                if failures > 5:
                    if videoStreamType == "video" or videoStreamType == "rtsp":
                        cap.release()
                        while framesIn.qsize() > 0 and framesOut.qsize() > 0:
                            time.sleep(1/30)
                        #print("shutting down - stream type : %s" % ( videoStreamType) )
                        videoQueueLoop.set()
                        if len(vidImgFiles) == 0:
                            quitEvent.set()
                    else:
                        #print("shutting down - stream type : %s" % ( videoStreamType) )
                        if len(vidImgFiles) == 0:
                            quitEvent.set()
    return

def mainLoop(quitEvent, videoObject):
    frameCounter = 0 
    data = {}
    vc = videoObject['comment']
    net = None
#    net = cv2.dnn.readNetFromDarknet(cw[pkgVer]["configPath"],cw[pkgVer]["weightsPath"]);
#    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    time.sleep(5)
    lastIndexes = []
    while not quitEvent.is_set():
        is_cat = False
        frameCounter += 1
        while framesIn.empty():
            #print ("mainLoop: queue empty")
            time.sleep(1/60)
            if framesIn.empty():
                if quitEvent.is_set():
                    return
        try:
            if args.ratein > 0:
                time.sleep(1/args.ratein)
            imgObject = framesIn.get(True, 1)
            if imgObject["noimage"]:
                #print (" ============ New File / Reseting DNN ===================\n")
                net = cv2.dnn.readNetFromDarknet(cw[pkgVer]["configPath"],cw[pkgVer]["weightsPath"]);
#                 net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#                 net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                #net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                #time.sleep(2)
                continue
        except queue.Empty:
            continue
        #img = imgObject["image"]
        img2 = None
        if args.writeImages:
            img2 = imgObject["image"].copy()
        if args.scaleimg:
            img = imgObject["image"]
            #height, width = img.shape[:2]
            #ih, iw, ic = img.shape
            img = ResizeWithAspectRatio(img, 1280)
        CONF_THRESH = args.conf
        CONF_THRESH_CUTOFF = args.confcutoff
        NMS_THRESH  = args.nms
        SCALE_FACTOR = 1/args.scalefactor
        img, is_cat, data, lastIndexes  = getObjects(imgObject,net,CONF_THRESH, CONF_THRESH_CUTOFF,NMS_THRESH, SCALE_FACTOR, frameCounter=frameCounter, lastIndexes=lastIndexes)
        #pp.pprint(data)
        framesOut.put({ "image": img, "data": data, "videoStreamType": imgObject["videoStreamType"], "path": imgObject["path"] })
        writeImagesOverride = True
        if args.writeImagesNotCats:
            for d in data:
                if d["is_cat"]:
                    writeImagesOverride = False
        if args.writeImages and len(data)>0 and writeImagesOverride:
            framesToWrite.put({ "image": img2, "data": data, "videoStreamType": imgObject["videoStreamType"],  "path": imgObject["path"] })
       

def printLabels(videoStreamType="file"):
    if args.quiet:
        return
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
                'pix': "pix"
                }
    if videoStreamType == "video" or videoStreamType == "file" or videoStreamType == "image": 
        print( "\n\n%(frame)8s %(catCounter)10s %(catTimer)10s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s %(pix)12s %(area)12s %(percent)5s \n" %  labels )
    else:
        print( "\n\n%(date)24s %(catCounter)10s %(catTimer)10s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s %(pix)12s %(area)12s %(percent)5s \n" %  labels )

def printDataLine(videoStreamType="file", d={} ):
    if args.quiet:
        return
    if videoStreamType == "video" or videoStreamType == "file" or videoStreamType == "image": 
        try:
            print( "%(frame)8d %(catCounter)12d %(catTimer)5d/%(meowTime)4d %(class)16s %(conf)8s %(x)6d %(y)6d %(w)6d %(h)6d %(pix)12d %(area)12d     %(percent)3.2f" % d )
        except (TypeError, IndexError) as e:
            print ("!! ========== TypeError/IndexError : ", e )
    else:
        try:
            print( "%(date)24s %(catCounter)12d %(catTimer)5d/%(meowTime)4d %(class)16s %(conf)8s %(x)6d %(y)6d %(w)6d %(h)6d %(pix)12d %(area)12d     %(percent)3.2f" % d )
        except (TypeError, IndexError) as e:
            print ("!! ========== TypeError/IndexError : ", e )

def writeImages(quitEvent, videoObject):
    subDir = None
    writePathBase = None
    serial = 0
    _excludeClasses = {}
    try:
        for o in args.excludeClass:
            c = o["class"]
            _excludeClasses[c] = {}
            _excludeClasses[c]['h'] = o["high"]
            _excludeClasses[c]['l'] = o["low"]
    except TypeError:
        pass
    pp.pprint(_excludeClasses)
    while not quitEvent.is_set():
        excludedThisFrame = False
        while framesIn.empty() and not quitEvent.is_set():
            time.sleep(1/30)
            if quitEvent.is_set():
                return
        try:
            imgObject = framesToWrite.get(True, 1)
        except queue.Empty:
            continue
            
        
        if not args.writeXmlOnly:
            frameNumber = imgObject['data'][0]['frame']
            baseFileNmae = re.sub(r'\.(jpg|jpeg|mkv|mp4)$', "", os.path.basename(imgObject["path"]) )
            frameDir = 128 * int(frameNumber / 128)
            subDir = baseFileNmae + "/" +"%08d" % ( frameDir )
            writePathBase = os.path.dirname(args.writePath)
            destDirPath = writePathBase + "/" + subDir
            filename = destDirPath + "/" + "%s-%08d" % (baseFileNmae, frameNumber)
        else:
            baseFileNmae = re.sub(r'\.(jpg|jpeg|mkv|mp4)$', "", os.path.basename(imgObject["path"]) )
            writePathBase = os.path.dirname(args.writePath)
            destDirPath = writePathBase
            filename = destDirPath + "/" + "%s" % (baseFileNmae)
        if not os.path.isdir(destDirPath):
            try:
                os.makedirs(destDirPath)
            except FileExistsError:
                pass
        xml2 = {}
        xml2['annotation'] = {}
        xml2['annotation']['folder'] = "none"
        xml2['annotation']['filename'] =  filename + ".jpg"
        xml2['annotation']['path'] = destDirPath
        xml2['annotation']['source'] = {}
        xml2['annotation']['source']['database'] = "Unknown"
        xml2['annotation']['segmented'] = 0
        xml2['annotation']['object'] = []
        for data in imgObject["data"]:
            c = data["class"]
            conf = float(data["conf"])
            h = -1
            l = -1
            try:
                h = _excludeClasses[c]['h']
                l = _excludeClasses[c]['l']
                if conf < h and conf > l:
                    excludedThisFrame = True
            except Exception as e:
                pass
            xml2['annotation']['object'].append( { 'name': data["class"],
                                                      'bndbox': { 'xmin': data["x"],
                                                                  'ymin': data["y"],
                                                                  'xmax': data["x"] + data["w"],
                                                                  'ymax': data["y"] + data["h"]
                                                                }
                                                            }
                                                        )
            print ( "{} {} {} {} W:{}".format(c,data["conf"],l, h,not excludedThisFrame) )
        serial+=1
        if not excludedThisFrame:
            f = open(filename + ".xml",'w')
            f.write(simplexml.dumps(xml2))
            f.close
            if not args.writeXmlOnly:
                cv2.imwrite(filename + ".jpg", imgObject["image"])
        

def displayImage(quitEvent, videoObject):
    waitKeyDealy = 1
    if args.wait:
       waitKeyDealy = 0 
    catTimer = time.time() - 11
    catCounter = 0
    lineCounter = 0
    meowTime_inital = 15
    meowTime_MAX = 960
    meowTime = meowTime_inital
    modulo = 30
    vc = videoObject['comment']
    cv2.namedWindow(vc, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    _windowOpen = False

#     while framesOut.empty():
#         #print ("displayImage: queue empty")
#         time.sleep(1/60)
#     imgObject = framesOut.get()
#     img2 = imgObject["image"]
    while not quitEvent.is_set():
        #if framesOut.empty():
        #    if quitEvent.is_set():
        #        return
        #    else:
        #       time.sleep(1/60)
        #        continue
        try:
            imgObject = framesOut.get()
        except queue.Empty:
            continue
        img2 = imgObject["image"]
        data = imgObject["data"]

        if _windowOpen == False:
            height, width = img2.shape[:2]
            print( "\n\n%8s %12d %14d" %  ( " h/w ", height, width ) )
            new_height = 1280
            reduceBy =  height / new_height
            print( "\n\n%8s %12d %14.2f" %  ( " new_height", new_height, reduceBy ) )
            new_width = int(width/reduceBy)
            print( "\n\n%8s %12d %14.2f" %  ( " new_width", new_width, reduceBy ) )
            cv2.resizeWindow(vc, new_width, new_height )
            cv2.moveWindow(vc, 75,50)
            _windowOpen = True
            #cv2.imshow(vc,img2)
            #cv2.waitKey(1)    
        is_cat_frame = False
        for d in data:
            if d['is_cat']:
                is_cat_frame = True
            d['catTimer'] = time.time() - catTimer
            c = d["class"]
            d['catCounter'] = catCounter
            d['meowTime'] = meowTime
            if not args.writeImages:
                if lineCounter % 25 == 0:
                    printLabels(imgObject["videoStreamType"])
                printDataLine(imgObject["videoStreamType"],d)
            lineCounter+=1
        if is_cat_frame:
            catCounter+=1
        else:
            catCounter=0
        if time.time() - catTimer > meowTime_MAX:
            meowTime = meowTime_inital
        if catCounter > 0 and catCounter % modulo == 0:
             end = time.time()
             seconds = end - catTimer
             if seconds > meowTime:
                meowTime = meowTime*2
                if meowTime > meowTime_MAX:
                    meowTime = meowTime_inital
                catTimer = time.time()
                if not _pygame_noSound:
                    pygame.mixer.music.play()
        cv2.imshow(vc,img2)
        cv2.waitKey(waitKeyDealy)

if __name__ == "__main__":
    videoQueueLoop = threading.Event()
    quitEvent = threading.Event()
    quitEvent.clear()
    videoQueueLoop.clear()
    
    videoObject = {}
    videoObject["videoFile"] = args.uri
    videoObject["comment"] = args.name
    vc = videoObject["comment"]

    if args.writeImages:
        writeImagesThread = threading.Thread(target=writeImages, args=(quitEvent, videoObject, ))
        writeImagesThread.start()
        time.sleep(1)
        writeImagesThread2 = threading.Thread(target=writeImages, args=(quitEvent, videoObject, ))
        writeImagesThread2.start()
        time.sleep(1)
        writeImagesThread3 = threading.Thread(target=writeImages, args=(quitEvent, videoObject, ))
        writeImagesThread3.start()
        time.sleep(1)


    
#    displayImageThread = threading.Thread(target=displayImage, args=(quitEvent, videoObject, ))
    mainLoopThread = threading.Thread(target=mainLoop, args=(quitEvent, videoObject, ))
    queueFramesThread = threading.Thread(target=queueFrames, args=(quitEvent, videoObject, videoQueueLoop ))

#    displayImageThread.start()
#    time.sleep(1)
    queueFramesThread.start()
    mainLoopThread.start()
    displayImage(quitEvent, videoObject)
    #time.sleep(1)
#     while not quitEvent.is_set():
#         try:
#             while not quitEvent.is_set():
#                 if not mainLoopThread.is_alive() and not quitEvent.is_set():
#                     mainLoopThread = threading.Thread(target=mainLoop, args=(quitEvent, videoObject, ))
#                     mainLoopThread.start()
#                 if not displayImageThread.is_alive() and not quitEvent.is_set():
#                     displayImageThread = threading.Thread(target=displayImage, args=(quitEvent, videoObject, ))
#                     displayImageThread.start()
#                 if not queueFramesThread.is_alive() and not quitEvent.is_set():
#                     queueFramesThread = threading.Thread(target=queueFrames, args=(quitEvent, videoObject, videoQueueLoop ))
#                     queueFramesThread.start()
#                 mainLoopThread.join(1)
#                 displayImageThread.join(1)
#                 queueFramesThread.join(1)
#                 if args.writeImages:
#                     writeImagesThread.join(1)
#                     writeImagesThread2.join(1)
#                     writeImagesThread3.join(1)
#         except KeyboardInterrupt:
#             quitEvent.set()
#             queueFramesThread.join(5)
#             displayImageThread.join(5)
#             if args.writeImages:
#                 writeImagesThread.join(5)
#                 writeImagesThread2.join(5)
#                 writeImagesThread3.join(5)
#             sys.exit(0)
# 
