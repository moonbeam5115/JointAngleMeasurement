import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import sys
import time

import pandas as pd
import tensorflow as tf
from tf_pose import common
import cv2 as cv
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
from math import atan

protoFile = "openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "openpose/models/pose/mpi/pose_iter_160000.caffemodel"

def joint_analysis_setup():
    # Read the network into Memory
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    #Load model made in PoseDetection_Model_Testing Notebook
    Pose_Analyzer_Model = tf.keras.models.load_model('saved_model_info/Pose_Analyzer_VGG16_model.h5')
    return net, Pose_Analyzer_Model

def plots(ims, figsize=(24, 12), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims - np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((1, 2, 3, 0))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) %2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i]/255, interpolation=None if interp else 'none')


def read_resize_prep_img(path_to_img, net, Pose_Analyzer_Model, const=2):
    string = path_to_img
    unknown_img_classify = cv.imread(string)

    const = const
    inWidth = 224
    inHeight = 224

    unknown_img_classify = cv.resize(unknown_img_classify, (inWidth, inHeight))

    unknown_img = cv.imread(string)
    unknown_img = cv.resize(unknown_img, (const*inWidth, const*inHeight))
    width, height, channels = unknown_img.shape
        

    return unknown_img, unknown_img_classify

def predict_pose(img_classify, net, Pose_Analyzer_Model):
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Text to display
    unknown_img_NN = np.expand_dims(img_classify, axis=0)
    num_predict = Pose_Analyzer_Model.predict(unknown_img_NN)

    num_df = pd.DataFrame(num_predict)
    num_predict = np.array(num_predict)
    num_predict = tf.argmax(num_predict, axis=1)
    text_predict = ''
    if np.array(num_predict) == 0:
        text_predict = 'squatting'
    elif np.array(num_predict) == 1:
        text_predict = 'bending'
    elif np.array(num_predict) == 2:
        text_predict = 'raising arms'
    else:
        text_predict = 'unknown pose detected!'

    plt.axis('off')
    plt.imshow(cv.cvtColor(img_classify, cv.COLOR_BGR2RGB))
    plt.text(112, -30, text_predict, size=30, rotation=0,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(0.2, 0.5, 0.5),
                    fc=(0.6, 0.8, 0.8),
                    )
            )
    plt.show()

def detect_draw_joints(string, net, Pose_Analyzer_Model, const=2):
    const = const
    inWidth = 224
    inHeight = 224

    unknown_img = cv.imread(string)
    img = unknown_img
    img = cv.resize(img, (inWidth, inHeight))

        # Prepare the frame to be fed to the network
    inpBlob = cv.dnn.blobFromImage(img, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    thickness = 2

    # Text to display
    unknown_img_NN = np.expand_dims(img, axis=0)
    num_predict = Pose_Analyzer_Model.predict(unknown_img_NN)

    num_df = pd.DataFrame(num_predict)
    num_predict = np.array(num_predict)
    num_predict = tf.argmax(num_predict, axis=1)
    text_predict = ''
    if np.array(num_predict) == 0:
        text_predict = 'squatting'
    elif np.array(num_predict) == 1:
        text_predict = 'bending'
    elif np.array(num_predict) == 2:
        text_predict == 'raising arms'
    else:
        text_predict == 'unknown pose detected!'


    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(15):
    # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
    # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
    # Scale the point to fit on the original image
        x = (inWidth* point[0])/H
        y = (inHeight* point[1])/W
        if prob:
            cv.circle(img, (int(x), int(y)), 5, (25, 22, 250), thickness=-1, lineType=cv.FILLED)
            # cv2.putText(unknown_img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    figure = plt.figure()

    #Head(0) to Neck(1)
    cv.line(img, points[0], points[1], (255, 255, 0), 2)

    #Neck(1) to Chest(14)
    cv.line(img, points[1], points[14], (255, 255, 0), 2)

    #Chest(14) to R(8) and L Hip(11)
    cv.line(img, points[14], points[8], (255, 255, 0), 2)
    cv.line(img, points[14], points[11], (255, 255, 0), 2)

    #Right Hip(8) to Right Knee(9)
    cv.line(img, points[8], points[9], (255, 255, 0), 2)
    #Right Knee(9) to Right Ankle(10)
    cv.line(img, points[9], points[10], (255, 255, 0), 2)

    #Left Hip(11) to Left Knee(12)
    cv.line(img, points[11], points[12], (255, 255, 0), 2)
    #Left Knee(12) to Left Ankle(13)
    cv.line(img, points[12], points[13], (255, 255, 0), 2)

    #Neck(1) to Right Shoulder(2)
    cv.line(img, points[1], points[2], (255, 255, 0), 2)
    #Right Shoulder(2) to Right Elbow(3)
    cv.line(img, points[2], points[3], (255, 255, 0), 2)
    #Right Elbow(3) to Right Wrist(4)
    cv.line(img, points[3], points[4], (255, 255, 0), 2)

    #Neck(1) to Left Shoulder(5)
    cv.line(img, points[1], points[5], (255, 255, 0), 2)
    #Left Shoulder(5) to Left Elbow(6)
    cv.line(img, points[5], points[6], (255, 255, 0), 2)
    #Left Elbow(6) to Left Wrist(7)
    cv.line(img, points[6], points[7], (255, 255, 0), 2)

    # rotate_knee = knee_angle*180/math.pi
    # cv.ellipse(unknown_img,(L_knee[0],L_knee[1]),(25,25),-rotate_knee, 0, knee_flexion,(230, 80, 60),-1)

    # rotate_hip = hip_angle*180/math.pi
    # cv.ellipse(unknown_img,(L_hip[0],L_hip[1]),(25,25),180-rotate_knee, 0, hip_flexion,(120, 230, 230),-1)

    #Pose Prediction Text
    plt.text(112, -35, text_predict, size=30, rotation=0,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(0.2, 0.5, 0.5),
                    fc=(0.6, 0.8, 0.8),
                    )
            )
    #Angle Text Info
    # plt.text(L_knee[0]-40, L_knee[1], 'Knee Flexion: \n{} degrees'.format(round(knee_flexion)), size=8, rotation=0,
    #          ha="center", va="center",
    #          bbox=dict(boxstyle="round",
    #                    ec=(0.2, 0.5, 0.5),
    #                    fc=(0.6, 0.8, 0.8),
    #                    )
    #          )

    # plt.text(L_hip[0]+50, L_hip[1], 'Hip Flexion: \n{} degrees'.format(round(hip_flexion)), size=8, rotation=0,
    #          ha="center", va="center",
    #          bbox=dict(boxstyle="round",
    #                    ec=(0.2, 0.5, 0.5),
    #                    fc=(0.6, 0.8, 0.8),
    #                    )
    #          )

    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.savefig('img/pred_result_jointAngle_002.jpg', dpi=400,  bbox_inches='tight')
    plt.show()




def measure_joint_angles(string, net, Pose_Analyzer_Model, const=2):
    const = const
    inWidth = 224
    inHeight = 224

    unknown_img = cv.imread(string)
    img = unknown_img
    img = cv.resize(img, (inWidth, inHeight))

        # Prepare the frame to be fed to the network
    inpBlob = cv.dnn.blobFromImage(img, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    thickness = 2

    # Text to display
    unknown_img_NN = np.expand_dims(img, axis=0)
    num_predict = Pose_Analyzer_Model.predict(unknown_img_NN)

    num_df = pd.DataFrame(num_predict)
    num_predict = np.array(num_predict)
    num_predict = tf.argmax(num_predict, axis=1)
    text_predict = ''
    if np.array(num_predict) == 0:
        text_predict = 'squatting'
    elif np.array(num_predict) == 1:
        text_predict = 'bending'
    elif np.array(num_predict) == 2:
        text_predict == 'raising arms'
    else:
        text_predict == 'unknown pose detected!'


    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(15):
    # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
    # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
    # Scale the point to fit on the original image
        x = (inWidth* point[0])/H
        y = (inHeight* point[1])/W
        if prob:
            cv.circle(img, (int(x), int(y)), 5, (25, 22, 250), thickness=-1, lineType=cv.FILLED)
            # cv2.putText(unknown_img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    figure = plt.figure()

    #Head(0) to Neck(1)
    cv.line(img, points[0], points[1], (255, 255, 0), 2)

    #Neck(1) to Chest(14)
    cv.line(img, points[1], points[14], (255, 255, 0), 2)

    #Chest(14) to R(8) and L Hip(11)
    cv.line(img, points[14], points[8], (255, 255, 0), 2)
    cv.line(img, points[14], points[11], (255, 255, 0), 2)

    #Right Hip(8) to Right Knee(9)
    cv.line(img, points[8], points[9], (255, 255, 0), 2)
    #Right Knee(9) to Right Ankle(10)
    cv.line(img, points[9], points[10], (255, 255, 0), 2)

    #Left Hip(11) to Left Knee(12)
    cv.line(img, points[11], points[12], (255, 255, 0), 2)
    #Left Knee(12) to Left Ankle(13)
    cv.line(img, points[12], points[13], (255, 255, 0), 2)

    #Neck(1) to Right Shoulder(2)
    cv.line(img, points[1], points[2], (255, 255, 0), 2)
    #Right Shoulder(2) to Right Elbow(3)
    cv.line(img, points[2], points[3], (255, 255, 0), 2)
    #Right Elbow(3) to Right Wrist(4)
    cv.line(img, points[3], points[4], (255, 255, 0), 2)

    #Neck(1) to Left Shoulder(5)
    cv.line(img, points[1], points[5], (255, 255, 0), 2)
    #Left Shoulder(5) to Left Elbow(6)
    cv.line(img, points[5], points[6], (255, 255, 0), 2)
    #Left Elbow(6) to Left Wrist(7)
    cv.line(img, points[6], points[7], (255, 255, 0), 2)


    #Left Ankle
    L_ankle = points[13]

    #Left Knee
    L_knee = points[12]

    #Left Hip
    L_hip = points[11]

    #Chest
    chest = points[14]

    ankle_angle = atan(abs(L_ankle[1] - L_knee[1])/ abs(L_ankle[0] - L_knee[0]))
    knee_angle = atan(abs(L_knee[1] - L_hip[1])/ abs(L_knee[0] - L_hip[0]))
    hip_angle = atan(abs(L_hip[1] - chest[1])/ abs(L_hip[0] - chest[0]))

    knee_flexion = (ankle_angle + knee_angle)*180/math.pi
    hip_flexion = (knee_angle + hip_angle)*180/math.pi

    rotate_knee = knee_angle*180/math.pi
    cv.ellipse(img,(L_knee[0],L_knee[1]),(25,25),-rotate_knee, 0, knee_flexion,(230, 80, 60),-1)

    rotate_hip = hip_angle*180/math.pi
    cv.ellipse(img,(L_hip[0],L_hip[1]),(25,25),180-rotate_knee, 0, hip_flexion,(120, 230, 230),-1)

    #Pose Prediction Text
    plt.text(112, -35, text_predict, size=30, rotation=0,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(0.2, 0.5, 0.5),
                    fc=(0.6, 0.8, 0.8),
                    )
            )
    #Angle Text Info
    plt.text(L_knee[0]-40, L_knee[1], 'Knee Flexion: \n{} degrees'.format(round(knee_flexion)), size=8, rotation=0,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0.2, 0.5, 0.5),
                       fc=(0.6, 0.8, 0.8),
                       )
             )

    plt.text(L_hip[0]+50, L_hip[1], 'Hip Flexion: \n{} degrees'.format(round(hip_flexion)), size=8, rotation=0,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0.2, 0.5, 0.5),
                       fc=(0.6, 0.8, 0.8),
                       )
             )

    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.savefig('img/pred_result_jointAngle_003.jpg', dpi=400,  bbox_inches='tight')
    plt.show()