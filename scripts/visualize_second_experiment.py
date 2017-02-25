#
# Uses the results from the mechanical turk experiment to generate statistics
# about the experiment conducted
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
import bisect

####################
##   Parameters   ##
####################

DATASET = "feret"
file_name = "all_unique_" + DATASET + "_names.csv"
# threshold = int(sys.argv[1]) # only consider pairs with agreement level above this number
HIT_RESULTS = "second_experiment.results"
output_file = "../results/mech_turk_" + ("10k" if DATASET == "10k" else "page") + "/second_experiment_min_max.html"

####################
##     Methods    ##
####################

def getDistance(name1,name2):
  return distsq[getIndex(name1)][getIndex(name2)]

#finds and returns the index at unique_feret_names corresponding to the image's file name
def getIndex(filename):
  global file_name
  f=open("../input/" + file_name,'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (img[0].startswith(filename)) :
      break
    counter = counter + 1
  return counter

# Open the Mechanical Turk result and start reading data
output= open(output_file, 'w')
numlines = sum(1 for line in open('../input/' + HIT_RESULTS)) -1
with open('../input/' + HIT_RESULTS, 'rb') as f:
    # First line is the header, store it so that we can get the index of the columns we are interested in
    reader = csv.reader(f,delimiter='\t')
    header = reader.next()
    row = reader.next()
    for i in range(1, numlines/10):
      # process the rows, each row contains 10 pairs of images
      master_image_name = row[header.index("Answer.imagename_0")]
      compared_image_names = []
      compared_image_scores = []
      compared_image_chosen = np.empty([6])
      compared_image_chosen_tmp = np.empty([6,10])
      output.write('<img src="../feret_cropped_opencv/' + master_image_name.replace('.jpg', '.png') + '"> : <br><br>')
      for j in range(10):
        for k in range(1, 7):    
          if j == 0:
            compared_image_names.append(row[header.index("Answer.imagename_" + str(k))])
            compared_image_scores.append(row[header.index("Answer.scores_" + str(k))])
          compared_image_chosen_tmp[k-1][j]=row[header.index("Answer.chosen_" + str(k))]
        row = reader.next()
      for k in range(6):
        #print k
        compared_image_chosen[k]=np.mean(compared_image_chosen_tmp[k])
        
      max_image_chosen = compared_image_chosen[np.argmax(compared_image_chosen)]
      max_image_name = compared_image_names[np.argmax(compared_image_chosen)]
      max_image_score = compared_image_scores[np.argmax(compared_image_chosen)]
      
      min_image_chosen = compared_image_chosen[np.argmin(compared_image_chosen)]
      min_image_name = compared_image_names[np.argmin(compared_image_chosen)]
      min_image_score = compared_image_scores[np.argmin(compared_image_chosen)]
      
      #for k in range(6):
      output.write('<img src="../feret_cropped_opencv/' + min_image_name.replace('.jpg', '.png') + '"> (' + min_image_score + ',' + str(min_image_chosen) + ') &nbsp;')
      output.write('<img src="../feret_cropped_opencv/' + max_image_name.replace('.jpg', '.png') + '"> (' + max_image_score + ',' + str(max_image_chosen) + ') &nbsp;')
      output.write('<br><br>')
      
