'''

    This script trains the VGG Face neural net using triplet loss (using the data
  gathered from the Mechanical Turk experiments.
  
  
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

DATASET = "feret"
METHOD = "OpenCV"
DATASET_PATH = "../../feret/"
MODEL_NAME = "2nd_continue_norm_all.ckpt"
MODEL_NAME_NO_EXT = MODEL_NAME.replace(".ckpt", "")
FORMATTED_FILE = "second_experiment_more_formatted.csv"
RESUME = True
FINE_TUNE = False
REINTILIAZE_FC = False
TRAIN = True
TEST = not(TRAIN)
BATCH_SIZE = 10 if TRAIN else 100

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
  newPath = (DATASET_PATH + s if DATASET=="10k" else (DATASET_PATH + s).replace(".png", ".ppm"))
  #print newPath
  return newPath
  
'''

  Get training batch :

'''

def next_training_batch():
  global reader, f
  anch_vectors = []
  neg_vectors = []
  pos_vectors = []
  anch_names = []
  neg_names = []
  pos_names = []
  while(len(anch_vectors)<BATCH_SIZE):
    try : 
      row = reader.next()
      anch_vectors.append(loadOnly(imagePath(row[0]))[0])
      neg_vectors.append(loadOnly(imagePath(row[1]))[0])
      pos_vectors.append(loadOnly(imagePath(row[2]))[0])
      # Add image names for debugging
      anch_names.append(row[0])
      neg_names.append(row[1])
      pos_names.append(row[2])
    except StopIteration:
      f.seek(0)
  return np.array(anch_vectors), np.array(neg_vectors), np.array(pos_vectors), np.array(anch_names), np.array(neg_names), np.array(pos_names)

'''
  
  Get test batch
  
'''
def get_test_triplets():
  global reader, f
  anch_vectors = []
  neg_vectors = []
  pos_vectors = []
  anch_names = []
  neg_names = []
  pos_names = []
  for i in range(100):
    row = reader.next()
    anch_vectors.append(loadOnly(imagePath(row[0]))[0])
    neg_vectors.append(loadOnly(imagePath(row[1]))[0])
    pos_vectors.append(loadOnly(imagePath(row[2]))[0])
    # Add image names for debugging
    anch_names.append(row[0])
    neg_names.append(row[1])
    pos_names.append(row[2])
  return np.array(anch_vectors), np.array(neg_vectors), np.array(pos_vectors), np.array(anch_names), np.array(neg_names), np.array(pos_names)

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
ses = tf.InteractiveSession()
saver = tf.train.Saver()

'''

  Triplet loss implementation

'''

anch_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
pos_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
neg_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])

anch = tf.nn.l2_normalize(network.network_eval(anch_img),1)
pos = tf.nn.l2_normalize(network.network_eval(pos_img),1)
neg = tf.nn.l2_normalize(network.network_eval(neg_img),1)

d_pos = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(anch, pos)),1))
d_neg = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(anch, neg)),1))

margin = 0.1

loss = tf.maximum(0., tf.add(margin, tf.sub(d_pos, d_neg)))
loss = tf.reduce_mean(loss)

# testing :
test_step = tf.less_equal(d_pos, d_neg)


if FINE_TUNE : 
#   fine_tune_vars = [v.name for v in tf.trainable_variables() ]
#   print fine_tune_vars
#   exit()
  fine_tune_vars = [v for v in tf.trainable_variables() if (v.name.startswith("linear_1/") or v.name.startswith("linear_2/"))]
  fine_tune_names = [v.name for v in tf.trainable_variables() if (v.name.startswith("linear_1/") or v.name.startswith("linear_2/"))]
  print fine_tune_names
  train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=fine_tune_vars)
else : 
  train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

'''

  Restore session

'''

if RESUME :
  saver.restore(ses, "./vggface/trainedmodels/" + MODEL_NAME)
else :
  saver.restore(ses, "./vggface/trainedmodels/initial.ckpt")  

if not(RESUME) and FINE_TUNE and REINTILIAZE_FC:
  reinitialize_op = tf.initialize_variables(fine_tune_vars)
  ses.run(reinitialize_op)
  
  
'''

Print information about this run

'''

print "Dataset : " + DATASET
print "Model name : " + MODEL_NAME

'''

  Open formatted Mechanical Turk results file and start training

'''
if TRAIN :
  print "Began training"
  with tf.device("/gpu:3"):

    global f
    f=open("../results/" + FORMATTED_FILE,'rU')
    global reader
    reader=csv.reader(f,delimiter=',')
        
    for i in range(5000):
      anch_vectors, neg_vectors, pos_vectors, anch_names, neg_names, pos_names = next_training_batch()
    #   print fp_fi_names[0], fp_si_names[0], ' more similar than ', sp_fi_names[0], sp_si_names[0]
      if i % 100 == 0:
        print "Step : " + str(i)
        #Save the model
        save_path = saver.save(ses, "./vggface/trainedmodels/" + MODEL_NAME)
        print("Model saved in file: %s" % save_path)
    #   print "Dpos :"
    #   print d_pos.eval(feed_dict={fp_fi_img:fp_fi_vect, fp_si_img:fp_si_vect, sp_fi_img:sp_fi_vect, sp_si_img:sp_si_vect})
    #   print "Dneg :"
    #   print d_neg.eval(feed_dict={fp_fi_img:fp_fi_vect, fp_si_img:fp_si_vect, sp_fi_img:sp_fi_vect, sp_si_img:sp_si_vect})
    #   print "tf.sub(d_pos, d_neg) :"
    #   print tf.sub(d_pos, d_neg).eval(feed_dict={fp_fi_img:fp_fi_vect, fp_si_img:fp_si_vect, sp_fi_img:sp_fi_vect, sp_si_img:sp_si_vect})
    #   print "tf.add(margin, tf.sub(d_pos, d_neg)) :"
    #   print tf.add(margin, tf.sub(d_pos, d_neg)).eval(feed_dict={fp_fi_img:fp_fi_vect, fp_si_img:fp_si_vect, sp_fi_img:sp_fi_vect, sp_si_img:sp_si_vect})
    #   print "tf.maximum(0., tf.add(margin, tf.sub(d_pos, d_neg))) : "
    #   print tf.maximum(0., tf.add(margin, tf.sub(d_pos, d_neg))).eval(feed_dict={fp_fi_img:fp_fi_vect, fp_si_img:fp_si_vect, sp_fi_img:sp_fi_vect, sp_si_img:sp_si_vect})
        print "Loss : "
        print loss.eval(feed_dict={anch_img:anch_vectors, neg_img:neg_vectors, pos_img:pos_vectors})
        train_step.run(feed_dict={anch_img:anch_vectors, neg_img:neg_vectors, pos_img:pos_vectors})

    save_path = saver.save(ses, "./vggface/trainedmodels/" + MODEL_NAME)
    print("Model saved in file: %s" % save_path)


'''

  Testing

'''
if TEST :
  print "Began testing :"
  with tf.device("/gpu:3"):
    f=open("../results/" + FORMATTED_FILE,'rU')
    reader=csv.reader(f,delimiter=',')    
    anch_vectors, neg_vectors, pos_vectors, anch_names, neg_names, pos_names = get_test_triplets()
    y_scores = np.concatenate((d_pos.eval(feed_dict={anch_img:anch_vectors, pos_img:pos_vectors}),d_neg.eval(feed_dict={anch_img:anch_vectors, neg_img:neg_vectors})),0)
    y_test = [0 for i in range(len(y_scores)/2)] + [1 for i in range(len(y_scores)/2)]
    # Print some of the scores and labels to verify
    for i in range(30):
      print str(y_scores[i])  + " - " + str(y_test[i])
    # Print some of the scores and labels to verify
    for i in range(100,130):
      print str(y_scores[i])  + " - " + str(y_test[i])

    # Compute precision and recall and area under the curve
    auc = average_precision_score(y_test, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Print
    print "AUC : "
    print auc
    print "Precision : "
    print precision
    print "Recall : "
    print recall
    print "Thresholds"
    print thresholds

    # Draw Curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.plot(recall, precision)
    plt.show()