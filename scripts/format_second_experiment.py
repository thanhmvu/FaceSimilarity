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
output_file = "../results/second_experiment_formatted.csv"

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
output= open(output_file, 'a+')
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
      output.write(master_image_name + ",")
      for j in range(10):
        for k in range(1, 7):    
          if j == 0:
            compared_image_names.append(row[header.index("Answer.imagename_" + str(k))])
            compared_image_scores.append(row[header.index("Answer.scores_" + str(k))])
          compared_image_chosen_tmp[k-1][j]=row[header.index("Answer.chosen_" + str(k))]
        row = reader.next()
      for k in range(6):
        compared_image_chosen[k]=np.mean(compared_image_chosen_tmp[k])
      argmax = np.argsort(compared_image_chosen)[-1] # change -1 to -2 or -3 etc. to get the image that had the second or third (etc) lowerst score
      
      max_image_chosen = compared_image_chosen[argmax]
      max_image_name = compared_image_names[argmax]
      max_image_score = compared_image_scores[argmax]
      
      argmin = np.argsort(compared_image_chosen)[0]  # change 0 to 1 or 2 etc. to get the image that had the second or third (etc) highest score
      min_image_chosen = compared_image_chosen[argmin]
      min_image_name = compared_image_names[argmin]
      min_image_score = compared_image_scores[argmin]
      
      output.write(min_image_name + ",")
      output.write(max_image_name)
      output.write('\n')
      
