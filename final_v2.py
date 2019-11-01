from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import sys
import pyrealsense2 as rs
import threading
import RPi.GPIO as GPIO
import time
import keyboard


flag = 0 # 0 means the hand is free and 1 means the hand is busy. free -> busy controled by YOLO detected result and busy-> free controlled by the GPIO thread.

# Pin definitions
control_pin = 12 # Board pin 12


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, color_img, depth_img):
    
    global flag, control_pin
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

      #  cv2.circle(color_img, (int(x), int(y)), 10, [0, 255, 0])
        dist = depth_img[int(y), int(x)]

        #depth = depth_img[int(x)-10:int(x)+10, int(y)-10:int(y)+10].astype(float)
        #masked = np.ma.masked_equal(depth, 0)
       # depth = depth * 0.03137

       # print("Detected a {} {} meters away.".format(detection[0].decode(), dist*0.001))

        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

	# if Find yotuba
        if (detection[0].decode() == "four"):
           # print("Find Four Clover!")
            boxColor = (255, 0, 255)
            if (0.28 < dist * 0.001 < 0.33 and 370 < int(x) < 420):
                boxColor = (0, 0, 255)
                GPIO.output(control_pin, GPIO.HIGH)
                GPIO.output(control_pin, GPIO.LOW)
                
            cv2.rectangle(color_img, pt1, pt2, boxColor , 3)
            cv2.putText(color_img,
                    detection[0].decode() +
                    " [" + "{0:.2f}".format(dist * 0.001) + " meters away]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            
    return color_img

netMain = None
metaMain = None
altNames = None

def gripper_control():
    # Pin Setup:
    global flag
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(control_pin, GPIO.OUT)  # LED pin set as output

    # Initial state for LEDs:
    GPIO.output(control_pin, GPIO.HIGH)

    
    try:
        while True:
            if keyboard.is_pressed('q'):
                break
            if (flag == 1):
              #  print("Waiting for button event")
              #  GPIO.wait_for_edge(but_pin, GPIO.FALLING)

                # event received when button pressed
               # print("Button Pressed!")
                time.sleep(2)

                GPIO.output(control_pin, GPIO.LOW)
                print("Grasp")
                time.sleep(2)

                GPIO.output(control_pin, GPIO.HIGH)
                print("Release")
                time.sleep(2)

                flag = 0
    finally:
        GPIO.cleanup()  # cleanup all GPIOs
        
        

def YOLO():

    
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(control_pin, GPIO.OUT)  # LED pin set as output

    # Initial state for LEDs:
    GPIO.output(control_pin, GPIO.LOW)

    global metaMain, netMain, altNames

    configPath = "models/clover/yolov3-tiny-clover.cfg"
    weightPath = "models/clover/yolov3-tiny-clover_final.weights"
    metaPath = "models/clover/clover.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")

    if metaMain is None:
        print(metaPath)
        metaMain = darknet.load_meta(metaPath.encode("ascii"))

    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1


    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("Starting the YOLO loop...")
    # _, frame_test = cap.read()

    darknet_image = darknet.make_image(640, 480, 3)

    while True:
        if keyboard.is_pressed('q'):
            print("Quite is triggered")
            break
        prev_time = time.time()

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames(5000)
        # frames.get_depth_frame() is a 640x480 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_frame_read = np.asanyarray(aligned_depth_frame.get_data())
        color_frame_read = np.asanyarray(color_frame.get_data())
        #color_frame = frame_read[:, :, :3].copy()
        #depth_frame = frame_read[:, :, 3].copy()

        frame_rgb = cv2.cvtColor(color_frame_read, cv2.COLOR_BGR2RGB)

        darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.3)

        color_image = cvDrawBoxes(detections, color_frame_read, depth_frame_read)
        

        print(1/(time.time()-prev_time))
        color_image = cv2.resize(color_image, (1280, 960))
        cv2.imshow('color', color_image)
       # cv2.imshow('depth', depth_image)
        cv2.waitKey(3)
    cv2.destroyAllWindows()   



if __name__ == "__main__":
    
    
    yolo_thread = threading.Thread(target=YOLO)
    # control_thread = threading.Thread(target=gripper_control)
    yolo_thread.start()
    # control_thread.start()
    yolo_thread.join()
    # control_thread.join()

