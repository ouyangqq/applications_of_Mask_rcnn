import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import coco
import utils
import model as modellib
import visualize
import cv2
from matplotlib import animation  
from PIL import Image

#matplotlib inline 
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #RPN_NMS_THRESHOLD=0.85

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))



filename="H:/My workshop/CV/visual_test/visual_attention_test/ggg.avi"
filename1="H:/video_jyz/caochang.mp4"
filename2="H:/video_jyz/DJI_0007.avi"
filename3="jiedao.mp4"
cap = cv2.VideoCapture(filename3)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#fourcc=-1
# 第三个参数则是镜头快慢的，20为正常，小于二十为慢镜头**
Video_w=640
Video_h=480
out = cv2.VideoWriter('output.avi',fourcc,20,(Video_w,Video_h))
#img=cv2.imread("images/hezhao.jpg")
cv2.namedWindow("lll")
img_no=0;
while( cap.isOpened() ) :
    ret,img = cap.read()
    if ret == True:
        # Run detection
        results = model.detect([img], verbose=0)
        # Visualize results
        r = results[0]
        img_no=img_no+1;
        visualize.display_instances(img, img_no, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        #plt.savefig('video_images4/'+str(img_no)+'.jpg')
        plt.close()
        maskedimage=plt.imread('video_images4/'+str(img_no)+'.jpg')
        #cv2.imshow("lll",maskedimage)
        out.write(maskedimage)
        #if img_no == 100:
        #    break
    else:
        break
    #k = cv2.waitKey(300)
    #if k == 27:
    #    break
cap.release()
out.release()
cv2.destroyAllWindows()
