# Import packages
import os
import serial
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import subprocess
# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 480
IM_WIDTH = 640   # Use smaller resolution for
IM_HEIGHT = 480  # slightly faster framerate
ser = serial.Serial("/dev/ttyACM0",9600,timeout=1)


# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.rotation = 180
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)
    initial_pico_argument = "pico2wave -w test.wav \"{} {} \" && aplay test.wav"
    temp=""

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        #print(boxes[0,0])
        #print(classes[0,0])


        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        # Temporary Code ----------------------------------- 
        threshold = 0.4
        position = ["Left","Ahead","Right"]
        for index,value in enumerate(classes[0]):
            label=""
            argument=""
            object_dict={}
            if scores[0,index] > threshold:
                object_name = (category_index.get(value)).get('name').encode('utf-8')
                bounding_box = boxes[0][index]
                width_object = bounding_box[3]*640+bounding_box[1]*640
                width_object_threshold = int(width_object/2)
                #angle = int(width_object_threshold)/10
                length_object = bounding_box[2]*480 - bounding_box[0]*480
                if length_object>=280:
                    if width_object_threshold<212:
                        label="Left"
                        #print(object_name," ",  "at Left")
                    elif width_object_threshold>=212 and width_object_threshold<424:
                        label="Ahead"
                        #print(object_name," ", "at Ahead")
                    elif width_object_threshold>=424 and width_object_threshold<=638:
                        label="Right"
                        #print(object_name," ", "at Right")

                if label == "Left":
                    angle = 48
                elif label == "Ahead":
                    angle= 32
                else:
                    angle = 11
                temp=""
                argument = str(angle) + '\n'
                ser.write(argument.encode())
                while(True):
                    inputi = ser.read().decode('utf-8')
                    if (inputi != '\r'):
                        temp = temp + inputi
                    else:
                        print(object_name," at ",label," at ",temp," angle ",angle)
                        temp=""
                        break


                

            #if scores[0,index] > threshold:
                #final_pico_argument=initial_pico_argument.format(((category_index.get(value)).get('name').encode("utf8")),"ahead")
                #subprocess.call(final_pico_argument,shell=True)

        #----------------------------------------------------        

        #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        #cv2.line(frame,(212,0),(212,480),(255,0,0),1)
        #cv2.line(frame,(424,0),(424,480),(255,0,0),1)
        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

cv2.destroyAllWindows()

