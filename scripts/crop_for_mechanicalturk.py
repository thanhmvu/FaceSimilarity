import numpy as np
order=[7,1,2,3,4,5,6,0,8,9,10]
import sys
sys.path=[sys.path[i] for i in order]
import cv2
from matplotlib import pyplot as plt
import sys
import os

def loadAndCrop(s):
  
  global face_cascade
  global total
  
  #read image and convert to gray scale
  img=cv2.imread(s)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  #detect faces. 
  faces = face_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
  tmp=len(faces[2])
  #print(tmp)
  #print(faces[2])
  #If none found use entire image
  if tmp==0:
    print(s + " " + str(len(faces)))
    total=total+1 
    img2=img
  else:
    #Find the face detected with highest confidence
    tmp=np.argmax(faces[2])
    length=faces[0][tmp][2]
    #make crop size a bit bigger than the detection 
    w_offset=round(length*0.05)
    h_offset=round(length*0.2)
    x1=max(0,faces[0][tmp][1]-h_offset)
    y1=max(0,faces[0][tmp][0]-w_offset)
    x2=min(faces[0][tmp][1]+faces[0][tmp][3]+h_offset,img.shape[0])
    y2=min(faces[0][tmp][0]+faces[0][tmp][2]+w_offset,img.shape[1])
    #print(str(x1) + " " + str(x2) + " " + str(y1) + " " +str(y2))
    img2=img[x1:x2,y1:y2]
  #resize
  
  ratio=224.0/img2.shape[1]
  img2 = cv2.resize(img2, (224, int(img2.shape[0] * ratio)))
  #return cropped image
  return img2




global face_cascade
global total
face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')
total=0;

if(len(sys.argv)<3):
  sys.exit("This script requires 2 argumenmts: input directory, output_directoty.")
i=0;
#all_files=["00776_941205_fa.ppm"]
all_files=os.listdir(sys.argv[1])
for path in all_files:
  #print(img2.shape[1])
  img=loadAndCrop(sys.argv[1] + "/" + path)
  #plt.imshow(img,), plt.show()
  out_file=sys.argv[2]+"/"+ path.replace(".ppm",".jpg")
  cv2.imwrite(out_file,img)
  print(i)
  i=i+1