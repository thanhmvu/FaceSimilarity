'''

Formats Mechanical Turk results to a better format to later use in training the network
using triplet loss.

'''

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

file_name = "all_unique_feret_names.csv"
output_file = "../results/feret_mechturk_formatted.csv"
HIT_FILE = "external_hit.results"

####################
##     Methods    ##
####################

def getDistance(id1,id2):
  return distsq[id1][id2]

#finds and returns the index at unique_feret_names corresponding to the image's file name
def getIndex(filename):
  namesfile=open("../input/" + file_name,'rU')
  reader=csv.reader(namesfile,delimiter='\n')
  counter = 0
  for img in reader:
    if (filename in img[0]) :
      break
    counter = counter + 1
  return counter

##
## Load distance matrix and calculate its expanded form
##

print('Started loading ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
dist=np.loadtxt("../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
print('Loaded ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
# Create empty agreement array to store the agreement percentage
# The result array is a 10 by 10 by 100 array (10 bins x 10 bins x (100 pairs))
agreement_percentages = []
for i in range(10) :
  ar = []
  for j in range(10):
    ar.append([])
  agreement_percentages.append(ar)

#Initialize arrays 
pairs_bins = {}
pairs_scores = {}
pairs_votes = {}
pairs_agreement = {}
pairs_results = {}

# Open the Mechanical Turk result and start reading data
with open('../input/' + HIT_FILE, 'rb') as f:
    # First line is the header, store it so that we can get the index of the columns we are interested in
    header = []
    is_header = True
    reader = csv.reader(f,delimiter='\t')
    for row in reader:
        if is_header :
          # Store the header
          header=row
          is_header = False
        else :
          # process the rows, each row contains 10 pairs of images
          for i in range(10):
            # generate a unique name for each pair
            first_pair_first_image_name = row[header.index("Answer.imagename_" + str(4*i))]
            first_pair_second_image_name = row[header.index("Answer.imagename_" + str(4*i+1))]
            second_pair_first_image_name = row[header.index("Answer.imagename_" + str(4*i+2))]
            second_pair_second_image_name = row[header.index("Answer.imagename_" + str(4*i+3))]
            first_pair_name = min(first_pair_first_image_name, first_pair_second_image_name) + "#" + max(first_pair_first_image_name, first_pair_second_image_name)
            second_pair_name = min(second_pair_first_image_name, second_pair_second_image_name) + "#" + max(second_pair_first_image_name, second_pair_second_image_name)
            first_pair_first_image_index = getIndex(first_pair_first_image_name)
            first_pair_second_image_index = getIndex(first_pair_second_image_name)
            second_pair_first_image_index = getIndex(second_pair_first_image_name)
            second_pair_second_image_index = getIndex(second_pair_second_image_name)
            # get the scores from vggface
            first_pair_score = getDistance(first_pair_first_image_index,first_pair_second_image_index)
            second_pair_score = getDistance(second_pair_first_image_index,second_pair_second_image_index)
            #store the score for each pair
            pairs_scores[first_pair_name] = first_pair_score
            pairs_scores[second_pair_name] = second_pair_score
            # read the chosen pair (1 or 2)
            chosen = int(row[header.index("Answer.chosen_" + str(i))])
            # append 1 to the chosen pair in the agreement percentages array
            pairs_votes.setdefault(first_pair_name, {})
            pairs_votes[first_pair_name].setdefault(second_pair_name, [])
            pairs_votes[first_pair_name][second_pair_name].append(0 if chosen==2 else 1)
            pairs_votes.setdefault(second_pair_name, {})
            pairs_votes[second_pair_name].setdefault(first_pair_name, [])
            pairs_votes[second_pair_name][first_pair_name].append(0 if chosen==1 else 1)

# Compute agreement for each of the "pair of pairs" by counting the number of votes that got the majority
for first_pair_name, second_pairs in pairs_votes.iteritems():
  for second_pair_name, votes in second_pairs.iteritems():
    pairs_agreement.setdefault(first_pair_name, {})
    pairs_results.setdefault(first_pair_name, {})
    frequencies = np.bincount(pairs_votes[first_pair_name][second_pair_name])
    pairs_agreement[first_pair_name].setdefault(second_pair_name, 100*float(frequencies.max())/len(pairs_votes[first_pair_name][second_pair_name]))
    if(len(frequencies) <=1 or frequencies[0] != frequencies[1]): # if there's total agreement or the agreement is not 50% (because if it's 50% then we can't decide which answer was right)
      pairs_results[first_pair_name].setdefault(second_pair_name, frequencies.argmax())

## Re-compute results ignoring pairs that weren't agreed upon
# Open the Mechanical Turk result and start reading data

# Empty the result array to store the data
agreement_output= open(output_file, 'w')
# Initialize arrays to store statistics data
total_tests = 0
counter = 0
#Computer statistics
for first_pair_name, second_pairs in pairs_results.iteritems():
  for second_pair_name, vote in second_pairs.iteritems():
    #Get the images' names
    first_pair_first_image_name = first_pair_name.split("#")[0] + ".jpg"
    first_pair_second_image_name = first_pair_name.split("#")[1] + ".jpg"
    second_pair_first_image_name = second_pair_name.split("#")[0] + ".jpg"
    second_pair_second_image_name = second_pair_name.split("#")[1] + ".jpg"
    #Get the scores for every pair (already stored when reading the result file)
    first_pair_score = pairs_scores[first_pair_name]
    second_pair_score = pairs_scores[second_pair_name]
    #Get the agreement percentage among voters for this comparison (already stored when reading the result file)
    agreement_pecentage = pairs_agreement[first_pair_name][second_pair_name]
    #Calculate what the computer voted for
    computer_vote = 1 if first_pair_score <= second_pair_score else 2
    human_vote = 1 if vote == 1 else 2
    agreement_output.write(",".join([first_pair_name, second_pair_name, first_pair_first_image_name, first_pair_second_image_name, second_pair_first_image_name, second_pair_second_image_name, str(first_pair_score), str(second_pair_score), str(agreement_pecentage), str(computer_vote), str(human_vote), "agree" if computer_vote==human_vote else "disagree"]))
    agreement_output.write("\n")
