# This scripts generates a webpage that shows the outlying pairs
# with their respective scores, images and cropped input from openCV
# Update : the openCV visualization is now computed here on an image
# by image basis (only for outliers) to save space, and the
# visualize_opencv_output.py is now merged with this script
import csv
order=[7,1,2,3,4,5,6,0,8,9,10]
import sys
sys.path=[sys.path[i] for i in order]
import cv2
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import defaultdict
from shutil import copyfile
import os
import dlib

#A function which loads and crops image using dlib's face detector
#A function which loads and crops image using dlib's face detector
def loadAndCrop(s):
  
  global detector
  global predictor
  global total
  
  #read image and convert to gray scale
  img=cv2.imread(s)  
  #detect faces. 
  dets = detector(img, 1)
  print(dets)
  tmp=len(dets)
  #If none found use entire image
  if tmp==0:
    #print(s + " " + str(len(dets)))
    total=total+1 
    img2=img
  else:
    #use the first face detected
    d = dets[0]
    #print("top: " + str(d.top()) + " bottom: " + str(d.bottom())  + " left: " + str(d.left())  + " right: " + str(d.right()) )
    img2=img[abs(d.top())*0.9: abs(d.bottom())*1.1, abs(d.left())*0.9: abs(d.right())*1.1]
  #resize and subtract average  
  img2 = cv2.resize(img2, (224, 224))
  #return cropped image
  return img2

#make necessary folders for web page
if not os.path.exists('../results/webpage'):
    os.makedirs('../results/webpage')
if not os.path.exists('../results/webpage/faces'):
    os.makedirs('../results/webpage/faces')
if not os.path.exists('../results/webpage/faces_cropped'):
    os.makedirs('../results/webpage/faces_cropped')


#setup tensorflow network
global total
total=0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlibDetector/shape_predictor_68_face_landmarks.dat")
output = open("../results/webpage/index.html", 'w')
with open('../results/oldscore_newscore_euclidean_comparison_outliers_dlib.csv') as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
      #Copy first face and compute it's corresponding opencv output
      copyfile('../../10k/' + row[0], '../results/webpage/faces/' + row[0])
      cv2.imwrite('../results/webpage/faces_cropped/' + row[0], loadAndCrop('../results/webpage/faces/' + row[0]))
      #Copy second face and compute it's corresponding opencv output
      copyfile('../../10k/' + row[1], '../results/webpage/faces/' + row[1])
      cv2.imwrite('../results/webpage/faces_cropped/' + row[1], loadAndCrop('../results/webpage/faces/' + row[1]))
      #Create HTML
      output.write(row[0] + ' compared to ' + row[1] + '<br>\n')
      output.write('FaceNet Score ' + row[2] + ', VGGFace Score : ' + row[3] + '<br>\n')
      output.write('Raw input : <br>\n')
      output.write('<img src="faces/' + row[0] + '"> vs <img src="faces/' + row[1] + '"><br>\n')
      output.write('Cropped input : <br>\n')
      output.write('<img src="faces_cropped/' + row[0] + '"> vs <img src="faces_cropped/' + row[1] + '"><br>\n')
      output.write('<br><br><br>\n')
        
