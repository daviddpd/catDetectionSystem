import argparse
import cv2
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
parser.add_argument('--uri', help='path to video file or rtsp ')
parser.add_argument('--name', help='Window Name', default=_name + ":" + str(_pid) )
parser.add_argument('--conf', help='confidence threshold', default=0.9 )
parser.add_argument('--nms', help='NMS threshold', default=0.5 )
parser.add_argument('--scale', help='SCALE_FACTOR denominator ( 1/VALUE )', default=300 )


#parser.add_argument('--classes', help='text file of strings of the class names')
args, _ = parser.parse_known_args()

framesIn  = queue.Queue(100)
framesOut = queue.Queue(100)
classNames = []
soundMeow = "/z/camera/Meow-cat-sound-effect.mp3"
pygame.mixer.init()
pygame.mixer.music.load(soundMeow)

cw = {}
cw["yolo.416v3.64"] = {}
cw["yolo.416v3.64"]["configPath"] = "/z/camera/communitycats/custom_data/cfg/yolov-tiny-custom-416v3-64.cfg"
#cw["yolo.416v3.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom_final-416v3-64.weights"
cw["yolo.416v3.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom-416v3-64_final.weights"
cw["yolo.416v3.64"]["coconames"] = "/z/camera/communitycats/custom_data/backup/custom-names-7v3.txt"
pkgVer = "yolo.416v3.64"

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


def getObjects(img, net, confThres, nmsThersh, scaleFactor=1/300, netSize=(416,416), frameCounter=0):
    layers = net.getLayerNames()
    #print ( layers )
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    data = []

    height, width = img.shape[:2]

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
    except AttributeError:
        #print ("AttributeError: no objects detected\n");        
        return img, False, data
    i=0
    ih, iw, ic = img.shape
    is_cat = False
    for index in indices:
        x=0
        y=0
        w=200
        h=200
        className = "NotSet"
        conf = "0.00"
        try:
            x, y, w, h = b_boxes[index]
            className = classes[class_ids[index]]
            classId = class_ids[index]
        except IndexError:
            print("IndexError");
        try:
            conf = str(round(confidences[index]*100,2))
        except IndexError: 
            print("IndexError: list index out of range index:{} ".format(index))
            print("class_ids: ", class_ids )
            print("confidences", confidences)
        try:
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[classId], 2)
            labelx = x + 10
            if re.search("^cat-", className):
                labelx = int(x + w/2 + 10)            
            cv2.putText(img, className, (labelx, y - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, colors[classId], 2)
            cv2.putText(img, conf, (labelx, y - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, colors[classId], 2)
            
            cv2.putText(img, className, ( 125, 75+(35*i)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, colors[classId], 2)
            cv2.putText(img, conf, (20, 75+(35*i)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, colors[classId], 2)
            i += 1
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
                        "is_cat": is_cat
                        }
                    )
        except IndexError:
            pass
    return img, is_cat, data

def queueFrames(quitEvent, videoObject):
    failures = 0
    v = videoObject['videoFile']
    vc = videoObject['comment']    
    if re.search("^rtsp", v):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        print("This is rtsp stream.")
    cap = cv2.VideoCapture(v,cv2.CAP_FFMPEG)
    while not quitEvent.is_set():
        success, img = cap.read()
        if success:
            framesIn.put(img)
        else:
            print("queueFrames: failure")
            failures+=1
        if failures > 30:
            quitEvent.set()

def mainLoop(quitEvent, videoObject):
    labels = {
                'frame': "frame",
                'class': "class",
                'conf': "conf", 
                'x':"x", 
                'y':"y", 
                'w':"w", 
                'h':"h",
                "catCounter": "consecCatFr",
                "catTimer": "sec/meow"
                }
    print( "\n\n%(frame)8s %(catCounter)12s %(catTimer)14s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s\n" %  labels )
    frameCounter = 0 
    lineCounter = 0
    catCounter = 1
    modulo = 10
    catTimer = time.time() - modulo 
    data = {}
    vc = videoObject['comment']
    
    net = cv2.dnn.readNetFromDarknet(cw[pkgVer]["configPath"],cw[pkgVer]["weightsPath"]);
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    while not quitEvent.is_set():
        is_cat = False
        frameCounter += 1
        #print("main[%s]: wait for framesIn get " % vc)
        #print( "%8s %12s %14s " %  ( "main", framesIn.qsize(), framesOut.qsize() ) )
        img = framesIn.get()
        CONF_THRESH = args.conf
        NMS_THRESH  = args.nms
        SCALE_FACTOR = 1/args.scale

        img, is_cat, data  = getObjects(img,net,CONF_THRESH,NMS_THRESH, SCALE_FACTOR, frameCounter=frameCounter)
        
        #print("main[%s]: wait for frame out put" % vc)
        framesOut.put(img)
        if is_cat:
            catCounter+=1
        else:
            catCounter=1
        for d in data:
            d['catTimer'] = time.time() - catTimer
            c = d["class"]
            if lineCounter % 25 == 0:
                print( "\n\n%(frame)8s %(catCounter)12s %(catTimer)14s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s\n" %  labels )
            print( "%(frame)8d %(catCounter)12d %(catTimer)14d %(class)16s %(conf)8s %(x)6d %(y)6d %(w)6d %(h)6d" % d )
            lineCounter+=1
        if (catCounter) % modulo == 0:
             end = time.time()
             seconds = end - catTimer
             if seconds > 10:
                catTimer = time.time()
                pygame.mixer.music.play()


def displayImage(quitEvent, videoObject):
    vc = videoObject['comment']
    cv2.namedWindow(vc, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    while not quitEvent.is_set():
        img2 = framesOut.get()
        cv2.imshow(vc,img2)
        cv2.waitKey(1)

# def sigterm_handler(_signo, _stack_frame):
#     print("sigterm_handler")
#     sys.exit(0)
# 
# signal.signal(signal.SIGTERM, sigterm_handler)

if __name__ == "__main__":
    quitEvent = threading.Event()
    quitEvent.clear()
    
    videoObject = {}
    videoObject["videoFile"] = args.uri
    videoObject["comment"] = args.name
    vc = videoObject["comment"]
    
    displayImageThread = threading.Thread(target=displayImage, args=(quitEvent, videoObject, ))
    queueFrames = threading.Thread(target=queueFrames, args=(quitEvent, videoObject, ))

    displayImageThread.start()
    queueFrames.start()

    while not quitEvent.is_set():
        try:
            mainLoop(quitEvent, videoObject)
        except KeyboardInterrupt:
            quitEvent.set()
            displayImageThread.join()
            queueFrames.join()
            sys.exit(0)

