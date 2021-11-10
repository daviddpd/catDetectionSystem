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

pp = pprint.PrettyPrinter(indent=4)
_pid  = os.getpid()
_name = sys.argv[0]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--uri', help='path to video file or rtsp ', required=True)
parser.add_argument('--writePath', help='path to save images', default="/z/camera/communitycats/custom_data/images-auto-extract")
parser.add_argument('--writeImages', help='Write Images with Objects detected',  default=False, action='store_true')
parser.add_argument('--writeImagesNotCats', help='Only write non-cat frames',  default=False, action='store_true')

parser.add_argument('--name', help='Window Name', default=_name + ":" + str(_pid) )
parser.add_argument('--conf', help='confidence threshold', default=0.9, type=float)
parser.add_argument('--nms', help='NMS threshold', default=0.5 , type=float)
parser.add_argument('--scalefactor', help='SCALE_FACTOR denominator ( 1/VALUE )', default=300, type=int)
parser.add_argument('--scaleimg', help='Scale Image',  default=False, action='store_true')
parser.add_argument('--areaReject', help='Reject Match if % pixel area is greater than this precent ', default=101.00 , type=float)


args, _ = parser.parse_known_args()

framesIn  = queue.Queue(100)
framesOut = queue.Queue(100)
framesToWrite = queue.Queue(1000)
classNames = []
soundMeow = "/z/camera/Meow-cat-sound-effect.mp3"
_pygame_noSound = False
try:
    pygame.mixer.init()
    pygame.mixer.music.load(soundMeow)
except:
    _pygame_noSound = True
    pass
    

cw = {}
cw["yolo.416v3.64"] = {}
cw["yolo.416v3.64"]["configPath"] = "/z/camera/catDectionSystem/yolo/cfg/yolov-tiny-custom-416v3-64.cfg"
cw["yolo.416v3.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom-416v3-64_final.weights"
cw["yolo.416v3.64"]["coconames"] = "/z/camera/catDectionSystem/yolo/cfg/custom-names-7v3.txt"

cw["yolo.416v4.64"] = {}
cw["yolo.416v4.64"]["configPath"] = "/z/camera/catDectionSystem/yolo/cfg/yolov-tiny-custom-416v4-64.cfg"
cw["yolo.416v4.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom-416v4-64_final.weights"
cw["yolo.416v4.64"]["coconames"] = "/z/camera/catDectionSystem/yolo/cfg/custom-names-v4.txt"

cw["yolo.416v5.64"] = {}
cw["yolo.416v5.64"]["configPath"] = "/z/camera/catDectionSystem/yolo/cfg/yolov-tiny-custom-416v5-64.cfg"
cw["yolo.416v5.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom-416v5-64_final.weights"
cw["yolo.416v5.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom-416v5-64_last.weights"
cw["yolo.416v5.64"]["coconames"] = "/z/camera/catDectionSystem/yolo/cfg/custom-names-v4.txt"


pkgVer = "yolo.416v5.64"

classNamesToIds = {}

with open(cw[pkgVer]["coconames"], "r") as f:
    classes = [line.strip() for line in f.readlines()]

colorsPencils = {}
colorsPencils['lime'] = (0, 250, 142)
colorsPencils['maraschino'] = (0, 38, 255)
colorsPencils['tangerine'] = (0, 147, 255)
colorsPencils['lemon'] = (0, 251, 255)
colorsPencils['blueberry'] = (255, 51, 4)
colorsPencils['stawberry'] = (146, 47, 255)
colorsPencils['snow'] = (255, 255, 255)
colorsPencils['lead'] = (33, 33, 33)
colorsPencils['turquoise'] = (255, 253, 0)

colors = [] 
colors = (colorsPencils['turquoise'],
          colorsPencils['tangerine'],
          colorsPencils['lime'],
          colorsPencils['maraschino'],
          colorsPencils['stawberry'],
          colorsPencils['lemon'],
          colorsPencils['blueberry']
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


def getObjects(imgObject, net, confThres, nmsThersh, scaleFactor=1/300, netSize=(416,416), frameCounter=0, lastIndexes=[]):
    img = imgObject["image"]
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    data = []
    mergeIndexes = []
    height, width = img.shape[:2]
    pix = height*width
    blob = cv2.dnn.blobFromImage(img, scaleFactor, netSize, swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    #print (layer_outputs)
    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThres:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    try:
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, confThres, nmsThersh).flatten().tolist()
        test1 = indices[0]
        
        mergeIndexes.append( { "indices": indices,
                               "b_boxes": b_boxes , 
                               "confidences": confidences, 
                               "class_ids": class_ids,
                               "frameCounter": frameCounter
                              } 
                            )
        #pp.pprint(mergeIndexes)
    except AttributeError:
        #print ("AttributeError: no objects detected\n");   
        mergeIndexes.append ( 
                               {   'b_boxes': [[-1,-1,-1,-1]],
                                  'class_ids': [-1],
                                  'confidences': [0],
                                  'frameCounter': frameCounter,
                                  'indices': [-1]
                               }
                            )
#        else:
#            return img, False, data, lastIndexes
    i=0
    mIdx = 11
    ih, iw, ic = img.shape
    is_cat = False
    mergeIndexes += lastIndexes
    lastIndexes = []
    for indexObject in mergeIndexes:
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
            className = "NotSet"
            conf = "0.00"
            is_empty = False
            _draw_color = colorsPencils['lead']
            percent = 0.0
            try:
                if index == -1:
                    x, y, w, h = b_boxes[0]
                    className = " - "
                    classId = -1
                    is_empty = True
                    _draw_color = colorsPencils['lead']
                else:
                    x, y, w, h = b_boxes[index]
                    className = classes[class_ids[index]]
                    classId = class_ids[index]
                    _draw_color = colors[classId]
                    area = w*h
                    percent = round(area/pix*100, 2)
                    if percent > args.areaReject:
                        x, y, w, h = [-1,-1,-1,-1]
                        className = " - "
                        classId = -1
                        is_empty = True
                        _draw_color = colorsPencils['lead']
            except IndexError:
                print("IndexError");
            try:
                conf = str(round(confidences[index]*100,2))
            except IndexError: 
                print("IndexError: list index out of range index:{} ".format(index))
                print("class_ids: ", class_ids )
                print("confidences", confidences)
            try:
                
                if not is_empty: 
                    _draw_color = _draw_color
                    cv2.rectangle(img, (x, y), (x + w, y + h), _draw_color, 2)
                    labelx = x + 10
                    if re.search("^cat-", className):
                        labelx = int(x + w/2 + 10)
                    if re.search("^cat", className):
                        is_cat = True
                    if i == 0:
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
    failures = 0
    v = videoObject['videoFile']
    vc = videoObject['comment']
    videoStreamType = "file"
    if re.search("^rtsp", v):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|reorder_queue_size;200|buffer_size;1048576"
        videoStreamType = "rtsp"
    else:
        print("\n\n");
        vid = ffmpeg.probe(v)
        for stream in vid['streams']:
            print ("Opening video %s code:[%s][%s] " % ( v,stream['codec_type'], stream['codec_name']))
            if  stream['codec_type'] == "video": 
                if stream['codec_name'] == "h264":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
                elif stream['codec_name'] == "hevc" or stream['codec_name'] == "h265":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;hevc_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
                else: 
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
    print("stream type : %s" % ( videoStreamType) )
    while not quitEvent.is_set():
        print("Openning stream : %s" % (v) )
        cap = cv2.VideoCapture(v,cv2.CAP_FFMPEG)
        videoQueueLoop.clear()
        while not videoQueueLoop.is_set():
            success, img = cap.read()
            t = time.localtime()
            strTime = time.strftime("%Y-%m-%d %H:%M:%S %Z", t)
            if success:
                framesIn.put({ "image": img, "time": strTime, "videoStreamType": videoStreamType })
            else:
                print("queueFrames: failure")
                failures+=1
            if failures > 5:
                cap.release()
                while framesIn.qsize() > 0 and framesOut.qsize() > 0:
                    time.sleep(1/30)
                if videoStreamType == "file":
                    print("shutting down - stream type : %s" % ( videoStreamType) )                        
                    quitEvent.set()
                    videoQueueLoop.set()
                else:
                    videoQueueLoop.set()
                    time.sleep(1)
    return                    

def mainLoop(quitEvent, videoObject):
    frameCounter = 0 
    data = {}
    vc = videoObject['comment']    
    net = cv2.dnn.readNetFromDarknet(cw[pkgVer]["configPath"],cw[pkgVer]["weightsPath"]);
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
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
            imgObject = framesIn.get(True, 1)
        except queue.Empty:
            continue
        img = imgObject["image"]
        img2 = None
        if args.writeImages:
            img2 = imgObject["image"].copy()
        if args.scaleimg:
            #height, width = img.shape[:2]
            #ih, iw, ic = img.shape
            img = ResizeWithAspectRatio(img, 1280)
        CONF_THRESH = args.conf
        NMS_THRESH  = args.nms
        SCALE_FACTOR = 1/args.scalefactor
        img, is_cat, data, lastIndexes  = getObjects(imgObject,net,CONF_THRESH,NMS_THRESH, SCALE_FACTOR, frameCounter=frameCounter, lastIndexes=lastIndexes)
        framesOut.put({ "image": img, "data": data, "videoStreamType": imgObject["videoStreamType"] })
        writeImagesOveride = True
        if args.writeImagesNotCats:
            for d in data:
                if d["is_cat"]:
                    writeImagesOveride = False
        if args.writeImages and len(data)>0 and writeImagesOveride:
            framesToWrite.put({ "image": img2, "data": data, "videoStreamType": imgObject["videoStreamType"] })
       

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
                'pix': "pix"
                }
    if videoStreamType == "file":
        print( "\n\n%(frame)8s %(catCounter)12s %(catTimer)14s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s %(pix)12s %(area)12s %(percent)5s \n" %  labels )
    else:
        print( "\n\n%(date)24s %(catCounter)12s %(catTimer)14s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s %(pix)12s %(area)12s %(percent)5s \n" %  labels )

def printDataLine(videoStreamType="file", d={} ):
    if videoStreamType == "file":
        print( "%(frame)8d %(catCounter)12d %(catTimer)14d %(class)16s %(conf)8s %(x)6d %(y)6d %(w)6d %(h)6d %(pix)12d %(area)12d     %(percent)3.2f" % d )
    else:
        print( "%(date)24s %(catCounter)12d %(catTimer)14d %(class)16s %(conf)8s %(x)6d %(y)6d %(w)6d %(h)6d %(pix)12d %(area)12d     %(percent)3.2f" % d )

def writeImages(quitEvent, videoObject):
    session = time.time()
    serial = 0
    print ("Starting Write Images Thread : Session %d" % (session) )
    while not quitEvent.is_set():
        print ("Writing for Image : Session %d" % (session) )
        while framesIn.empty() and not quitEvent.is_set():
            #print ("writeImages: queue empty")
            time.sleep(1/30)
            if quitEvent.is_set():
                return
        try:
            imgObject = framesToWrite.get(True, 1)
        except queue.Empty:
            continue
        filename = args.writePath + "/" + "%10d-%08d" % (session, serial)
        print ("Writing : File %s" % (filename) )
        cv2.imwrite(filename + ".jpg", imgObject["image"])
        xml2 = {}
        xml2['annotation'] = {}
        xml2['annotation']['folder'] = "none"
        xml2['annotation']['filename'] = "%10d-%08d.jpg" % (session, serial)
        xml2['annotation']['path'] = args.writePath
        xml2['annotation']['source'] = {}
        xml2['annotation']['source']['database'] = "Unknown"
        xml2['annotation']['segmented'] = 0
        xml2['annotation']['object'] = []
        for data in imgObject["data"]:
            xml2['annotation']['object'].append( { 'name': data["class"],
                                                      'bndbox': { 'xmin': data["x"],
                                                                  'ymin': data["y"],
                                                                  'xmax': data["x"] + data["w"],
                                                                  'ymax': data["y"] + data["h"]
                                                                }
                                                            }
                                                        )
        pp.pprint (xml2)
        print (simplexml.dumps(xml2))
        serial+=1
        f = open(filename + ".xml",'w')
        f.write(simplexml.dumps(xml2))
        f.close

def displayImage(quitEvent, videoObject):
    catTimer = time.time() - 11
    catCounter = 0
    lineCounter = 0
    meowTime = 10
    modulo = 10
    vc = videoObject['comment']
    cv2.namedWindow(vc, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    while framesOut.empty():
        #print ("displayImage: queue empty")
        time.sleep(1/60)
    imgObject = framesOut.get()
    img2 = imgObject["image"]
    height, width = img2.shape[:2]
    print( "\n\n%8s %12d %14d" %  ( " h/w ", height, width ) )
    new_height = 800
    reduceBy =  height / new_height
    print( "\n\n%8s %12d %14.2f" %  ( " new_height", new_height, reduceBy ) )
    new_width = int(width/reduceBy)
    print( "\n\n%8s %12d %14.2f" %  ( " new_width", new_width, reduceBy ) )
    cv2.resizeWindow(vc, new_width, new_height )
    cv2.moveWindow(vc, 100,50)
    cv2.imshow(vc,img2)
    cv2.waitKey(1)    

    while not quitEvent.is_set():
        if framesOut.empty():
            if quitEvent.is_set():
                return
            else:
                time.sleep(1/60)
                continue
        try:
            imgObject = framesOut.get(True, 1)
        except queue.Empty:
            continue
        img2 = imgObject["image"]
        data = imgObject["data"]
        is_cat_frame = False
        for d in data:
            if d['is_cat']:
                is_cat_frame = True
            d['catTimer'] = time.time() - catTimer
            c = d["class"]
            d['catCounter'] = catCounter
            if lineCounter % 25 == 0:
                printLabels(imgObject["videoStreamType"])
            printDataLine(imgObject["videoStreamType"],d)
            lineCounter+=1
        if is_cat_frame:
            catCounter+=1
        else:
            catCounter=0
        if meowTime > 150:
            meowTime = 10
        if catCounter > 0 and catCounter % modulo == 0:
             end = time.time()
             seconds = end - catTimer
             if seconds > meowTime:
                meowTime = meowTime*2
                if meowTime > 150:
                    meowTime = 10
                catTimer = time.time()
                if not _pygame_noSound:
                    pygame.mixer.music.play()
        cv2.imshow(vc,img2)
        cv2.waitKey(1)        

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
    
    displayImageThread = threading.Thread(target=displayImage, args=(quitEvent, videoObject, ))
    queueFramesThread = threading.Thread(target=queueFrames, args=(quitEvent, videoObject, videoQueueLoop ))

    displayImageThread.start()
    queueFramesThread.start()

    while not quitEvent.is_set():
        try:
            mainLoop(quitEvent, videoObject)
        except KeyboardInterrupt:
            quitEvent.set()
            displayImageThread.join()
            queueFramesThread.join()
            if args.writeImages:
                writeImagesThread.join()
            sys.exit(0)

