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
CONF_THRESH, NMS_THRESH = 0.9, 0.5
            
classNames = []
soundMeow = "/z/camera/Meow-cat-sound-effect.mp3"
pygame.mixer.init()
pygame.mixer.music.load(soundMeow)
#pygame.mixer.music.play()


cw = {}
cw["yolov3"] = {}
cw["yolov3"]["configPath"] = "/home/dpd/darknet-yolo4/cfg/yolov3.cfg"
cw["yolov3"]["weightsPath"] = "/home/dpd/darknet/yolov3.weights"
cw["yolov3"]["coconames"] = "/home/dpd/darknet-yolo4/data/coco.names"


cw["yolov4tiny"] = {}
cw["yolov4tiny"]["configPath"] = "/home/dpd/darknet-yolo4/cfg/yolov4-tiny.cfg"
cw["yolov4tiny"]["weightsPath"] = "/z/camera/yolov4-tiny.conv.29"
cw["yolov4tiny"]["coconames"] = "/home/dpd/darknet-yolo4/data/coco.names"


cw["yolov4tiny.custom"] = {}
cw["yolov4tiny.custom"]["configPath"] = "/z/camera/communitycats/custom_data/cfg/yolov-tiny-custom.cfg"
cw["yolov4tiny.custom"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom_final.weights"
cw["yolov4tiny.custom"]["coconames"] = "/z/camera/communitycats/custom_data/custom.names.7"

cw["yolo.320v1"] = {}
cw["yolo.320v1"]["configPath"] = "/z/camera/communitycats/custom_data/cfg/yolov-tiny-custom-320v1.cfg"
cw["yolo.320v1"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom_final-320v1.weights"
cw["yolo.320v1"]["coconames"] = "/z/camera/communitycats/custom_data/custom.names.7"


cw["yolo.416v1.64"] = {}
cw["yolo.416v1.64"]["configPath"] = "/z/camera/communitycats/custom_data/cfg/yolov-tiny-custom-416v1-64.cfg"
cw["yolo.416v1.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom_final-416v1-64.weights"
cw["yolo.416v1.64"]["coconames"] = "/z/camera/communitycats/custom_data/backup/custom.names.7"


cw["yolo.416v2.64"] = {}
cw["yolo.416v2.64"]["configPath"] = "/z/camera/communitycats/custom_data/cfg/yolov-tiny-custom-416v2-64.cfg"
cw["yolo.416v2.64"]["weightsPath"] = "/z/camera/communitycats/custom_data/backup/yolov-tiny-custom_final-416v2-64.weights"
cw["yolo.416v2.64"]["coconames"] = "/z/camera/communitycats/custom_data/backup/custom.names.9"

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

# colors[classNamesToIds['cat']]          = colorsPencils['blueberry']
# colors[classNamesToIds['cat-domino']]   = colorsPencils['lemon']
# colors[classNamesToIds['cat-kitten6']]    = colorsPencils['stawberry']
# colors[classNamesToIds['cat-olive']]    = colorsPencils['maraschino']
# colors[classNamesToIds['opossum']]    = colorsPencils['lime']
# colors[classNamesToIds['raccoon']]    = colorsPencils['tangerine']
# colors[classNamesToIds['skunk']]    = colorsPencils['turquoise']


colors = (colorsPencils['turquoise'],
          colorsPencils['tangerine'],
          colorsPencils['lime'],
          colorsPencils['maraschino'],
          colorsPencils['stawberry'],
          colorsPencils['lemon'],
          colorsPencils['blueberry']
          )


print (colors)


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

def getObjects(img, net, thres, nms, draw=True, objects=[], frameCounter=0):

    # Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    data = []

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    #img = cv2.imread(args.image)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1/300, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    #indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
    try:
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
        test1 = indices[0]
    except AttributeError:
        #print ("AttributeError: no objects detected\n");        
        return img, False, data
    i=0
    ih, iw, ic = img.shape
    #print ( " X x Y: {} {}", ih,iw)
    is_cat = False
#     for index in indices:
#         className = classes[class_ids[index]]
#         if re.search("^cat", className):
#             is_cat = 1
#             is_cat_hash[className] = 1
    
    #pp.pprint (is_cat_hash)
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


    #classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
#     if len(objects) == 0: objects = classNames
#     objectInfo =[]
#     if len(classIds) != 0:
#         for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
#             className = classNames[classId - 1]
#             if className in objects:
#                 objectInfo.append([box,className])
#                 if (draw):
#                     cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#                     cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#                     cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

#    return img,objectInfo





frameCounter = 0 
resettimer = 1
frames = 0
catCounter = 1 
cap = object

#v = "/z/camera/2021-Oct-Night.mkv"
#v = "/z/camera/uploads/c3/2021/10/13/.C3_01_20211013205816-1280x960.mkv"
#v = "/z/camera/uploads/c3/2021/10/12/.C3_01_20211012204541-1280x960.mkv"
#v = "/z/camera/C3_01_20211016002548-0000-0800-1280x960.mkv"

#v = "/z/camera/C3/C3-2021-10-17-0000-0800.mkv"


videos = queue.Queue()


# v = {
#         "videoFile": "/z/camera/2021-Oct-Night-fullrez.mkv",
#         "comment" :  ""
#     }
# videos.append(v)


play = (    "C3-2021-10-20-0000-0800.mkv",
            "C3-2021-10-20-0801-1600.mkv",
            "C3-2021-10-20-1601-2359.mkv",
            "C3-2021-10-21-0000-0800.mkv",
            "C3-2021-10-21-0801-1600.mkv",
            "C3-2021-10-21-1601-2359.mkv",
            "C3-2021-10-22-0000-0800.mkv",
            "C3-2021-10-22-0801-1600.mkv",
            "C3-2021-10-22-1601-2359.mkv",
            "C3-2021-10-23-0000-0800.mkv",
            "C3-2021-10-23-0801-1600.mkv",
            "C3-2021-10-23-1601-2359.mkv",
        )
# 
# 
for p in play:
    v = {
             "videoFile": "/z/camera/C3/" + p,
             "comment" :  ""
         }
    videos.put(v)

# v = {
#         "videoFile": "/z/camera/uploads/c3/2021/10/17/C3_01_20211017235405.mp4",
#         "comment" :  "Olive approaching, 1:55"
#     }
# videos.append(v)
# v = {
#         "videoFile": "/z/camera/uploads/c3/2021/10/17/C3_01_20211017220622.mp4",
#         "comment" :  "Domion approaching, 1:02"
#     }
# videos.append(v)
# v = {
#         "videoFile": "/z/camera/uploads/c3/2021/10/17/C3_01_20211017225239.mp4",
#         "comment" :  "Domion returns, approaching from left, eating 5:22"
#     }
# videos.append(v)


# v = {
#          "videoFile": "/z/camera/C3/C3-2021-10-23-1601-2359.mkv",
#          "comment" :  ""
#      }
# videos.append(v)
# 


#cap = cv2.VideoCapture(v)
# cap = cv2.VideoCapture(v, cv2.CAP_FFMPEG)

#cap = acapture.open(v) # Camera 0,  /dev/video0

# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# if int(major_ver)  < 3 :
#     fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#     print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
# else :
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# 
framesIn = queue.Queue(100)
framesOut = queue.Queue(100)
ready = queue.Queue(1)

def openVideoStream():
    videoObject = videos.get()
    v = videoObject['videoFile']
    
    if re.search("^rtsp", v):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        print("This is rtsp stream.")
    else:
        vid = ffmpeg.probe(v)
        for stream in vid['streams']:
            print ("\n\nOpening video %s code:[%s][%s] " % ( v,stream['codec_type'], stream['codec_name']))
            if  stream['codec_type'] == "video": 
                if stream['codec_name'] == "h264":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
                elif stream['codec_name'] == "hevc":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;hevc_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
                elif stream['codec_name'] == "h265":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;hevc_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
                else: 
                    #os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(v,cv2.CAP_FFMPEG)
    return cap
    

def queueFrames():
    failures = 0
    cap = openVideoStream()
    while True:
        success, img = cap.read()
        if success:
            framesIn.put(img)
        else:
            failures+=1
        if failures > 30:
            cap.release()
            try:
                cap = openVideoStream()
                failures=0
            except:
                sys.exit(0)

#             if framesIn.qsize() < 10:
#                 framesIn.put(img)
#             else:
#                 dropFrame = framesIn.get()
#                 print( "%8s %12s"  %  ( ' ', 'drop frame' )  )
#                 #print( "%8d %s" % ( 0, "dropping frame" ))
        
def mainLoop():
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
    objectCounter = {}
    objectCounterRef = {}
    objectCounterLastFrame = {}
    objectCounterQ = {} 
    for c in classes:
        objectCounter[c] = 0
        objectCounterLastFrame[c] = 0
        objectCounterRef[c] = 0
        objectCounterQ[c] = []

    modulo = 10
    catTimer = time.time() - modulo 
    data = {}
    
    net = cv2.dnn.readNetFromDarknet(cw[pkgVer]["configPath"],cw[pkgVer]["weightsPath"]);
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    while True:
        for c in classes:
            objectCounterRef[c] = 0
        is_cat = False
        frameCounter += 1
        img = framesIn.get()
        img, is_cat, data  = getObjects(img,net,0.45,0.2, frameCounter=frameCounter)
        framesOut.put(img)
        if is_cat:
            catCounter+=1
        else:
            catCounter=1
        for d in data:
            d['catTimer'] = time.time() - catTimer
            c = d["class"]
            d['catCounter'] = catCounter
            objectCounterRef[c] += 1
            frameGap = d['frame'] - objectCounterLastFrame[c]
            if frameGap == 0 and len(objectCounterQ[c]) < 11:
                pp.pprint( objectCounterQ[c] )
            if frameGap > 0 and frameGap < 11:
                objectCounterQ[c].append(d['frame'])
            elif frameGap > 10:
                objectCounterQ[c] = []
            objectCounterLastFrame[c] = d['frame']
            if lineCounter % 25 == 0:
                print( "\n\n%(frame)8s %(catCounter)12s %(catTimer)14s %(class)16s %(conf)8s %(x)6s %(y)6s %(w)6s %(h)6s\n" %  labels )
            print( "%(frame)8d %(catCounter)12d %(catTimer)14d %(class)16s %(conf)8s %(x)6d %(y)6d %(w)6d %(h)6d" % d )
            #print( "%8s %12s %6d %6d  %16s" % ( " ", " ", frameGap, len(objectCounterQ[c]), c ) )
            lineCounter+=1
        if (catCounter) % modulo == 0:
             end = time.time()
             seconds = end - catTimer
             if seconds > 10:
                catTimer = time.time()
                pygame.mixer.music.play()
#         for c in classes:
#             if objectCounterRef[c] == 0:
#                 objectCounter[c] = 0
#             else:
#                 objectCounter[c] += objectCounterRef[c]


def displayImage():
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    while True:
        img = framesOut.get()
        cv2.imshow("Output",img)
        cv2.waitKey(1)

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    print("sigterm_handler")
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)

if __name__ == "__main__":
    mainThread = threading.Thread(target=mainLoop, daemon=True)
    mainThread.start()
    displayImageThread = threading.Thread(target=displayImage, daemon=True)
    displayImageThread.start()
    queueFrames = threading.Thread(target=queueFrames, daemon=True)
    queueFrames.start()
    while True:
        try:
            #print( "%8s %12s %14s " %  ( "main", framesIn.qsize(), framesOut.qsize() ) )
            time.sleep(2)
        except KeyboardInterrupt:
            sys.exit(0)


    #out = cv2.VideoWriter('/home/camera/output.mkv',cv2.VideoWriter_fourcc('H','E','V','C'), 30, (1024, 768))

#    cap = cv2.VideoCapture("/z/camera/uploads/c3/2021/09/30/.C3_01_20210930201654-1280x960.mkv")
#    cap = cv2.VideoCapture("/z/camera/C3/C3-2021-10-02-1601-2359.mkv")
#    cap = cv2.VideoCapture("/z/camera/C3/C3-2021-10-02-0000-0800.mkv")
    
    
    # VID_PATH,cv2.CAP_FFMPEG
    #cap.set(3,640)
    #cap.set(4,480)
    #cap.set(10,70)
        #resize = ResizeWithAspectRatio(img, width=1280)
        #result, objectInfo = getObjects(img,0.45,0.2, objects=['cat',"bowl"])
        #objectInfo(print)
#       cv2.imshow("Output",img)
        #out.write(img)
#         end = time.time()
#         seconds = end - start
#         frames=frames+1
#         if seconds > 5 :
#             print ("Time taken : {0} seconds".format(seconds))
#             fps  = frames / seconds
#             print("Estimated frames per second : {0}".format(fps))
#             resettimer = 1
#             frames = 0
    #v = "/z/camera/C3/C3-2021-10-08-1601-2359.mkv"
    #v = "/z/camera/uploads/c3/2021/09/30/.C3_01_20210930201654-1280x960.mkv"
    #cap = cv2.VideoCapture("/z/camera/uploads/c3/2021/09/30/.C3_01_20210930201654-1280x960.mkv", cv2.CAP_FFMPEG)    
    
        #v = "/z/camera/C3-2021-Sept.mkv"
    #v ="/z/camera/C3-2021-Sept-0000.mkv"
    
    #v = "/z/camera/C3-2021-10-09.mkv"
    
    #v = "/z/camera/C3/C3-2021-10-11-0000-0800.mkv"
    #v ="/z/camera/C3/C3-2021-10-12-0000-0800.mkv"
    #v ="/z/camera/C3/C3-2021-10-13-0000-0800.mkv"
    #v = "/z/camera/C3/C3-2021-10-14-1601-2359.mkv"
    #v = "/z/camera/C1-2021-10-Night.mkv"
    #v = "/z/camera/C4/C4-2021-10-16-0000-0800.mkv"    
    #v = "/z/camera/2021-Oct-Night.mkv"
    #v = "/z/camera/C3/C3-2021-10-16-0000-0800.mkv"    
    #v = "/z/camera/C3/C3-2021-10-15-0000-0800.mkv"
    #v = "/z/camera/C3/C3-2021-10-14-0000-0800.mkv"
    #v = "/z/camera/C3_01_20211016002548-0000-0800-1280x960.mkv"
    #v = "/z/camera/uploads/c4/2021/10/15/.C4_01_20211015194005-1280x960.mkv"
