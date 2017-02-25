# This script takes as input the csv file used in the last Mechanical Turk experiment,
# it reformats it by splitting the pairs and recording their old score, it then calculates
# the new score using vggface
import csv
order=[7,1,2,3,4,5,6,0,8,9,10];
import sys
sys.path=[sys.path[i] for i in order]
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list
dist_function = 'euclidean'


#finds and returns the index at unique_feret_names corresponding to the image's file name
def getIndex(filename):
  namesfile=open("../input/all_used_10k_names.csv",'rU')
  reader=csv.reader(namesfile,delimiter='\n')
  counter = 0
  for img in reader:
    if (img[0] == filename) :
      break
    counter = counter + 1
  return counter

## Load distance matrix and calculate its expanded form
dist=np.loadtxt("../distance_mat/all_used_10k_names_" + dist_function + "_distmat_dlib.csv")
distsq=squareform(dist)

## Step1 : split the pairs of the original file (since input has two pair per line)

# output = open("../results/all_pairs_used_split.csv", 'w')
# with open('../input/all_pairs_used_with_scores.csv', 'rb') as csvfile:
#     inputcsv = csv.reader(csvfile, delimiter=',')
#     for row in inputcsv:
#       output.write(row[0] + ',' + row[1] + ',' + row[4]+ '\n')
#       output.write(row[2] + ',' + row[3] + ',' + row[5]+ '\n')
# output.close()

## Step2 : calculate the new scores using vggface

output = open("../results/oldscore_newscore_" + dist_function + "_comparison_dlib.csv", 'w')
with open('../results/all_pairs_used_split.csv', 'rb') as csvfile:
    inputcsv = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in inputcsv:
      distance = distsq[getIndex(row[0])][getIndex(row[1])]
      #un-comment following line to only look at outliers
      #if float(row[2]) < 0.8 or distance < 1.2 : continue
      output.write(row[0] + ',' + row[1] + ',' + row[2] + ',' + str(distance) + '\n')
      count = count + 1
output.close()

## Step 3 : Graph new score vs old scores

with open('../results/oldscore_newscore_' + dist_function + '_comparison_dlib.csv') as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
plt.plot(columns[2], columns[3], 'ro')
plt.savefig("../histograms/oldvsnew_" + dist_function + "_dlib.png")

## Step 4 : Isolate outliers in another file
output = open("../results/oldscore_newscore_" + dist_function + "_comparison_outliers_dlib.csv", 'w')
with open('../results/all_pairs_used_split.csv', 'rb') as csvfile:
    inputcsv = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in inputcsv:
      distance = distsq[getIndex(row[0])][getIndex(row[1])]
      #un-comment following line to only look at outliers
      if float(row[2]) < 0.8 or distance < 1.2 : continue
      output.write(row[0] + ',' + row[1] + ',' + row[2] + ',' + str(distance) + '\n')
      count = count + 1
output.close()

