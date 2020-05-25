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

#Joint Analysis Setup
protoFile = "openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "openpose/models/pose/mpi/pose_iter_160000.caffemodel"
# Read the network into Memory
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
#Load model made in PoseDetection_Model_Testing Notebook
Pose_Analyzer_Model = tf.keras.models.load_model('saved_model_info/Pose_Analyzer_VGG16_model.h5')

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


def read_resize_prep_img(path_to_img, const=2):
    string = path_to_img
    unknown_img_classify = cv.imread(string)

    const = const
    inWidth = 224
    inHeight = 224

    unknown_img_classify = cv.resize(unknown_img_classify, (inWidth, inHeight))

    unknown_img = cv.imread(string)
    unknown_img = cv.resize(unknown_img, (const*inWidth, const*inHeight))
    width, height, channels = unknown_img.shape
        
    # Prepare the frame to be fed to the network
    inpBlob = cv.dnn.blobFromImage(unknown_img, 1.0/255, (const*inWidth, const*inHeight), (0, 0, 0), swapRB=False, crop=False)
        
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)
    return unknown_img, unknown_img_classify

def predict_pose(img_classify):
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