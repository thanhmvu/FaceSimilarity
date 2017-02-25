'''

    This script tests the trained VGG Face models on the first vggface experiment (prints the number of
    tests that it got right and writes it to results/test_results.txt)
  
  
'''

import numpy as np
order=[7,1,2,3,4,5,6,0,8,9,10]
import sys
sys.path=[sys.path[i] for i in order]
import cv2
from matplotlib import pyplot as plt
import sys
import os
import csv
import vggface
import tensorflow as tf
from scipy.spatial.distance import pdist
import dlib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import itertools

DATASET = sys.argv[1]
METHOD = "OpenCV"
DATASET_PATH = "../../" + DATASET + "/"
MODEL_NAME = sys.argv[2]
MODEL_NAME_NO_EXT = MODEL_NAME.replace(".ckpt", "")
FORMATTED_FILE = DATASET + "_mechturk_formatted.csv"
BATCH_SIZE = 1
THRESHOLD = 80

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
  
  #read image and convert to gray scale
  img=cv2.imread(s)
  gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.equalizeHist(gray, gray)  
  #detect faces. 
  faces = face_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
  tmp=len(faces[2])
  #If none found use entire image
  if tmp==0:
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
  
def imagePath(s):
  newPath = (DATASET_PATH + s if DATASET=="10k" else (DATASET_PATH + s).replace(".jpg", ".ppm"))
  #print newPath
  return newPath
  
'''
  
  Get test batch
  
'''
def get_test_quadruple(row):
  fp_fi_vectors = []
  fp_si_vectors = []
  sp_fi_vectors = []
  sp_si_vectors = []
  fp_fi_names = []
  fp_si_names = []
  sp_fi_names = []
  sp_si_names = []
  if (int(row[10])==1):
    fp_fi_vectors.append(loadOnly(imagePath(row[2]))[0])
    fp_si_vectors.append(loadOnly(imagePath(row[3]))[0])
    sp_fi_vectors.append(loadOnly(imagePath(row[4]))[0])
    sp_si_vectors.append(loadOnly(imagePath(row[5]))[0])
    # Add image names for debugging
    fp_fi_names.append(row[2])
    fp_si_names.append(row[3])
    sp_fi_names.append(row[4])
    sp_si_names.append(row[5])
  else :
    sp_fi_vectors.append(loadOnly(imagePath(row[2]))[0])
    sp_si_vectors.append(loadOnly(imagePath(row[3]))[0])
    fp_fi_vectors.append(loadOnly(imagePath(row[4]))[0])
    fp_si_vectors.append(loadOnly(imagePath(row[5]))[0])
    # Add image names for debugging
    sp_fi_names.append(row[2])
    sp_si_names.append(row[3])
    fp_fi_names.append(row[4])
    fp_si_names.append(row[5])
  return np.array(fp_fi_vectors), np.array(fp_si_vectors), np.array(sp_fi_vectors), np.array(sp_si_vectors), np.array(fp_fi_names), np.array(fp_si_names), np.array(sp_fi_names), np.array(sp_si_names)

'''

  Variable declaration

'''
#This are variables related to the deep neural network
global input_placeholder
global network
#face_cascade is used in the face detector. Making it global so not to initialize it each time
global face_cascade

'''

  Setup opencv and dlib

'''

face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlibDetector/shape_predictor_68_face_landmarks.dat")

'''

  Setup tensorflow network

'''
# input_placeholder = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
network = vggface.VGGFace(BATCH_SIZE)
tf.logging.set_verbosity(tf.logging.ERROR)
ses = tf.InteractiveSession()
saver = tf.train.Saver()

'''

  Triplet loss implementation

'''

first_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
second_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])

first = tf.nn.l2_normalize(network.network_eval(first_img),1)
second = tf.nn.l2_normalize(network.network_eval(second_img),1)

dist = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(first, second)),1))


'''

  Restore session

'''

saver.restore(ses, "./vggface/trainedmodels/" + MODEL_NAME)  
  
'''

Print information about this run

'''
output=open("../results/test_results.txt",'a')

print "Dataset : " + DATASET
output.write("Dataset : " + DATASET + "\n")
print "Model name : " + MODEL_NAME
output.write("Model name : " + MODEL_NAME + "\n")

'''

  Testing

'''
print "Began testing :"
with tf.device("/gpu:2"):
  result = 0
  total = 0
  f=open("../results/" + FORMATTED_FILE,'rU')
  reader=csv.reader(f,delimiter=',')
  for row in reader:
    if float(row[8]) >= THRESHOLD:
      fp_fi, fp_si, sp_fi, sp_si, fp_fi_names, fp_si_names, sp_fi_names, sp_si_names = get_test_quadruple(row)
      total += 1
      fp_d = dist.eval(feed_dict={first_img:fp_fi, second_img:fp_si})
      sp_d = dist.eval(feed_dict={first_img:sp_fi, second_img:sp_si})
      if (fp_d<=sp_d):
        result += 1
#       else :
#         print fp_fi_names, fp_si_names, sp_fi_names, sp_si_names
#         print fp_d, sp_d
  print "Accuracy : "
  output.write("Accuracy : \n")
  print str(result/2) + " guessed right / " + str(total/2) + " total (" + str(100*(float(result)/total)) + " % )"
  output.write(str(result/2) + " guessed right / " + str(total/2) + " total (" + str(100*(float(result)/total)) + " % ) \n")
  output.write("\n")
