# Feature - Triggering threshold. 

Right now, I believe the object detections triggers on the first frame of detection. However, I think what we need to create is a little hysteresis in detection. 

triggers.audio and triggers.hooks ... add an options, something like "sequential frames" (possible select a better name for this feature and config option, calling FRAMES_DETECT for now) .  The number of frames, in a row, before the detection of the object is considered TRUE. The default is 10. For the sake of the live stream, where the ingest to inference may drop frames, we are going to ignore these dropped frames. It will be frames in a row processed by inference.  Detections will stay TRUE, FRAMES_DETECT_OFF value is hit, which will be another configutation option for both triggers, but is unset, by default is 50% of FRAMES_DETECT. 

Also, for Detection to trigger, FRAMES_DETECT_SIZE will be defined.  By default this is zero (disabled), so all objects detected will flip Detection to true.  However, if set, the detected object will need to be so-many pixels (or percentage of the image) to trigger detection to TRUE. Add percentage & pixel area to the logs.  Add percentage annotations in the overlays. 

Also, create a circular buffer (stack, fifo, you are free to select best data structure) of frames (probably frame objects), it's length also set in the config, by default will be FRAMES_DETECT * 3.  The the FRAMES_DETECT-th frame is detected, it's saved into this buffer, without all graphical overlays. However, the detection metadata should also be stored with this, so the bounding box can be redrawn or a VOC XML file could be created. Another frame isn't added to the buffer until Detections fall to false, then back to true. 

Multiple objects detected in the frame, only needs to preserve the frame once. This buffer should be it's own thread, so any memory/storage management processing doesn't impact the ingest and inference processing. 

There will need to be another window, that will be cds-detections, that will display, and some how be able to scroll through the static images of the FRAMES_DETECT-th Detections. Default is the last 6, and should be in the configuration file as well.  When displaying these, the bounding box of all the objects needs to be draw, and then scale the image down, maybe by 50%, so it's fits on the screen. 

# Feature Add auto frame grabs for training

This will build on the above, but only be able to be used when --benchmark is specified.  Let's not enable this for the live streams. 
--export-frames will enable the option, and --export-frames-dir will set the output directory. 

The idea here, is to grap frames that are likely what we are trying to detect, but have a low confidence value, to use to re-train the model, in attempt to improve object detection. 

--confidence will be as is, but the user will set this lower than normal in this operating mode. 
--confidence-min is what we have normally been running the model at to prevent false positives.

The frames of the confidence values between confidence and confidence-min are we want to export.  Always export the frame just before hitting confidence-min, and the frame after dropping below confidence-min. Then take a random sample of frames in-between confidence and confidence-min to export, and the number of frames to randomly additionally export, by default is 10%, but make this config options and command line option as well. Export the image as jpeg, at the same resolution of the video. Also export/create VOC XML. Creating the YOLO TXT file doesn't make sense at this point, that will be done with another tool.  Don't overwrite files already in the directory, rename them as need. naming convention - probably some sort of time-stamp+serial number (or frame number) extracted if possible from the metadata of the video file. 

