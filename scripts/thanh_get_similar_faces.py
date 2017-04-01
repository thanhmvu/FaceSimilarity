
import csv
order=[7,1,2,3,4,5,6,0,8,9,10];
import sys
sys.path=[sys.path[i] for i in order]
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import collections
import os
from shutil import copyfile

import thanh_display_sim_faces as dsf

DATASET_DIR = "../../faces_5k_cropped/"
NAME_FILE = "faces_5k_AZ_names.csv"
OUTPUT_HTML = "../results/results_5k_AZ.html"
OUTPUT_TXT = "../results/results_5k_AZ.txt"


# Finds and returns the index at unique_feret_names corresponding to the image's file name
def getFilename(index):
  f=open("../input/" +NAME_FILE,'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (counter == index) :
      return img[0]
    counter = counter + 1

##
## ============================== Find similar faces ==============================
##

# Load distance matrix
print('Loading ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_.") + " ...")
dist=np.loadtxt("../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
print('Loaded ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
# print(dist)

# Calculate matrix's expanded form
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
print(distsq)

# Find n most similar faces
mostSim = {}
NUM_MOST_SIM = 6  
for i in range(len(distsq)):
  srcImg = getFilename(i)
  dist_i = distsq[i]
  
	# sort dist_i and find the n most similar
  most_sim_idx = []
  for j in np.argsort(dist_i):
    srcName = srcImg.split("_")[0]
    simName = getFilename(j).split("_")[0]
    if simName != srcName: # discard images of people of same first name
	    most_sim_idx.append(j)
    if len(most_sim_idx) == NUM_MOST_SIM:
      break
  
  # add to output map
  print (srcImg)
  mostSim[srcImg] = [(getFilename(j),round(dist_i[j],4)) for j in most_sim_idx]
  mostSim[srcImg] = sorted(mostSim[srcImg], key=lambda x: x[1])
  print (mostSim[srcImg])

	
  
##
## ============================== Display output webpage ==============================
##
# Generate name file for mechanical turk
dsf.print_txt(mostSim, OUTPUT_TXT)

# Generate output webpage
dsf.print_html(mostSim, DATASET_DIR, OUTPUT_HTML)
