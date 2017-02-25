#
#This script processes the brunelle database and gives scores to each one of the doppelgangers
#

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

#computes the normalized distance between two numpy vectors
def getDistance(id1,id2):
  return distsq[id1][id2]

#finds and returns the index at all_unique_brunelle_names corresponding to the image's file name
def getIndex(filename):
  f=open("../input/all_unique_brunelle_names.csv",'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (img[0] == filename) :
      break
    counter = counter + 1
  return counter

#finds and returns the index at all_unique_brunelle_names corresponding to the image's file name
def getFilename(index):
  f=open("../input/all_unique_brunelle_names.csv",'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (counter == index) :
      return img[0]
    counter = counter + 1

##
## Load distance matrix and calculate its expanded form
##

dist=np.loadtxt("../distance_mat/all_unique_brunelle_names_euclidean_distmat_dlib.csv")
print('Loaded ' + "../distance_mat/all_unique_brunelle_names_euclidean_distmat_dlib.csv")
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/all_unique_brunelle_names_euclidean_distmat_dlib.csv")
#make necessary folder for results
if not os.path.exists('../results/brunelle_page'):
    os.makedirs('../results/brunelle_page')
    
# store the binned pairs in a text file (100 per bin) and copy the images to the page's folder
doppelganger_scores = []
index_page= open("../results/brunelle_page/index.html", 'w')
index_page.write("<h2>Results for Brunelle database</h2><br><br>")
index_page.write("<a href='hist.png'><img height='300px' src='hist.png'/></a><br><br>")
for i in range(2, 93) :
    first_image = str(i) + "-1.jpg"
    second_image = str(i) + "-2.jpg"
    first_image_index = getIndex(first_image)
    second_image_index = getIndex(second_image)
    pair_score = distsq[first_image_index][second_image_index]
    doppelganger_scores.append(pair_score)
    index_page.write("<img height='150px' src='../brunelle/" + first_image + "'> vs <img height='150px' src='../brunelle/" + second_image + "'> ("  + str(pair_score) + ") <br><br>")

hist, bins = np.histogram(doppelganger_scores)
fig, ax = plt.subplots()
plt.hist(doppelganger_scores)
fig.savefig("../results/brunelle_page/hist.png")
print('Finished writing ' + "../results/brunelle_page/index.html")


