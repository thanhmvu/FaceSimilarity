'''

This script reads a text file with names of image files (one per line) to find faces and extract a descriptor
input file should be in ../input_directory and the descriptors would be written to the ../description_vectors directory with _vec in the end
Finally a distnace matrix (in condensed form )in written to ../distance_mat with a _distmat in the end

'''

import numpy as np
# order=[7,1,2,3,4,5,6,0,8,9,10]
# import sys
# sys.path=[sys.path[i] for i in order]
import cv2
from matplotlib import pyplot as plt
import sys
import os
import csv
import vggface
import tensorflow as tf
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import dlib


METHOD = "OpenCV"
DATASET_PATH = "../../feret/"
# DATASET_PATH = "../../faces_5k/"
MODEL_NAME = "2nd_continue_norm_all.ckpt"
MODEL_NAME_NO_EXT = MODEL_NAME.replace(".ckpt", "")
INPUT_FILE = "all_unique_feret_names.csv"
SAVE_IMAGES = False


'''

  Replace the last occurence of a string (used to replace the last / in the file path)
  
'''

def rreplace(s, old, new, occurrence):
  li = s.rsplit(old, occurrence)
  return new.join(li)

'''

  Loads, detects face, resizes and converts the images using OpenCV
  
'''

def loadAndCropCV(s):
  
  global face_cascade
  global total
  
  #read image and convert to gray scale
  img=cv2.imread(s)
  gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.equalizeHist(gray, gray)  
  #detect faces. 
  faces = face_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
  tmp=len(faces[2])
  #If none found use entire image
  if tmp==0:
    total=total+1 
    img2=img
  else:
    #use the last face detected since it is the biggest
    tmp=np.argmax(faces[2])
    length=faces[0][tmp][2]
    #make crop size a bit bigger than the detection 
    offset=round(length*0.05)
    x1=max(0,faces[0][tmp][1]-offset)
    y1=max(0,faces[0][tmp][0]-offset)
    x2=min(faces[0][tmp][1]+faces[0][tmp][3]+offset,img.shape[0])
    y2=min(faces[0][tmp][0]+faces[0][tmp][2]+offset,img.shape[1])
    #print(str(x1) + " " + str(x2) + " " + str(y1) + " " +str(y2))
    img2=img[x1:x2,y1:y2]
  #resize  
  img2 = cv2.resize(img2, (224, 224))
  #save image
  if SAVE_IMAGES:
    cv2.imwrite(rreplace(s, "/", "_opencv_cropped/", 1).replace(".ppm", ".png"), img2)
  img2 = img2.astype(np.float32)
  img2 -= [129.1863,104.7624,93.5940]
  img2 = np.array([img2,])  
  #return cropped image
  return img2

'''

  Loads, detects face, resizes and converts the images using Dlib
  
'''

def loadAndCropDlib(s):
  
  global detector
  global predictor
  global total
  
  #read image and convert to gray scale
  img=cv2.imread(s)  
  detimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.equalizeHist(detimg, detimg)
  #detect faces. 
  dets = detector(detimg, 1)
  #print(dets)
  tmp=len(dets)
  #If none found use entire image
  if tmp==0:
    total=total+1 
    img2=img
  else:
    #use the first face detected
    d = dets[0]
    offset=round(d.width()*0.15)
    #make crop size a bit bigger than the detection 
    x1=max(0,abs(d.left())-offset)
    y1=max(0,abs(d.top())-offset)
    x2=min(abs(d.right())+offset,img.shape[0])
    y2=min(abs(d.bottom())+offset,img.shape[1])
    img2=img[y1:y2,x1:x2]
  #resize  
  img2 = cv2.resize(img2, (224, 224))
  #save image
  if SAVE_IMAGES:
    cv2.imwrite(rreplace(s, "/", "_dlib_cropped/", 1).replace(".ppm", ".png"), img2)
  #convert
  img2 = img2.astype(np.float32)
  img2 -= [129.1863,104.7624,93.5940]
  img2 = np.array([img2,])  
  #return cropped image
  return img2

'''

  Loads, detects face, resizes and converts the images using either Dlib or OpenCV as specified

'''

def loadAndCrop(s, method='OpenCV'):
  if method == 'OpenCV':
    return loadAndCropCV(s)
  elif method == 'Dlib':
    return loadAndCropDlib(s)
  else:
    return Null
  
def loadOnly(s):
  #read image and convert to gray scale
  img=cv2.imread(rreplace(s, "/", "_cropped/", 1))  
  #convert
  img2 = img.astype(np.float32)
  img2 -= [129.1863,104.7624,93.5940]
  img2 = np.array([img2,])  
  #return cropped image
  return img2 
  
'''

Gets output vector from VGG Face Network

'''

def getVector(path1, method='OpenCV'):
  #print("Processing " + path1)
  global input_placeholder
  global network
  global network_evaluated
  #detect faces
  img1=loadOnly(path1)
  #extract features
  output1 = network_evaluated.eval(feed_dict={x_image:img1})[0]
  norm_output1 = output1/np.linalg.norm(output1,2)
  return norm_output1
#   return output1

'''

Variable declaration

'''
#This are variables related to the deep neural network
global input_placeholder
global network
#face_cascade is used in the face detector. Making it global so not to initialize it each time
global face_cascade
#total is the total number of images in which a face was not detected
global total


#setup opencv and dlib
face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlibDetector/shape_predictor_68_face_landmarks.dat")

#setup tensorflow network
x_image = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
ses = tf.InteractiveSession()
network = vggface.VGGFace(1)
saver = tf.train.Saver()
saver.restore(ses, "./vggface/trainedmodels/" + MODEL_NAME)
network_evaluated = network.network_eval(x_image)
total=0

first_img = tf.placeholder(tf.float32, shape=(1,4096))
second_img = tf.placeholder(tf.float32, shape=(1,4096))
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(first_img, second_img)),1))

#storing all description vectors
vectors=[]

#open file with face file names (one per row)
f=open("../input/" + INPUT_FILE,'rU')

reader=csv.reader(f,delimiter='\t')
counter=1
#for all the rows
with tf.device("/gpu:3"):
  for path1 in reader:

    #add correct path
    path1=DATASET_PATH+path1[0]

    #get vector
    vector = getVector(path1, METHOD)  

    #add vector to vectors matrix
    if(counter==1):
      vectors = vector;
    else:
      vectors=np.vstack([vectors,vector])

    #increment counter
    counter=counter+1  

#save vectors 
print('Saving vector file...')
out_file="../description_vectors/"+ INPUT_FILE.replace(".","_vect_" + MODEL_NAME_NO_EXT + "_" + ("dlib" if METHOD == "Dlib" else "") + ".")
np.savetxt(out_file, vectors, delimiter='\t',fmt="%1.4e")

#create condensed euclidian distance matrix ((see scipy.spatial.distance.squareform to convert to square form) and save it)
print('Generating distance matrix..')
dist=pdist(vectors)
out_file="../distance_mat/"+ INPUT_FILE.replace(".","_distmat_" + MODEL_NAME_NO_EXT + "_" + ("dlib" if METHOD == "Dlib" else "") + ".")
np.savetxt(out_file, dist, delimiter='\t',fmt="%1.4e")