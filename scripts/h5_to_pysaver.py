#This script reads a text file with names of image files (one per line) to find faces and extract a descriptor
#input file should be in ../input_directory and the descriptors would be written to the ../description_vectors directory with _vec in the end
#Finally a distnace matrix (in condensed form )in written to ../distance_mat with a _distmat in the end

import cv2
import numpy as np
import csv
import vggface
import tensorflow as tf
import sys
from scipy.spatial.distance import pdist

#########################Beginning of main code################################

#This are variables related to the deep neural network
global input_placeholder
global network
#face_cascade is used in the face detector. Making it global so not to initialize it each time
global face_cascade
#total is the total number of images in which a face was not detected
global total


#setup tensorflow network
input_placeholder = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
ses = tf.InteractiveSession()
network = vggface.VGGFace()
network.load(ses,input_placeholder)
saver = tf.train.Saver()
save_path = saver.save(ses, "./vggface/trainedmodels/untrained.py")


