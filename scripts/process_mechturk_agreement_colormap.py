#
# Uses the results from the mechanical turk experiment to plot a graph of the number
# of tests that yieded that the row pair is more similar than the column pair ignoring
# tests with low agreement among voters
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

#computes the normalized distance between two numpy vectors
def getDistance(id1,id2):
  return distsq[id1][id2]

##
## Load distance matrix and calculate its expanded form
##

file_name = "all_unique_feret_names.csv"
threshold = 80 # only consider pairs with confidence level above this number

dist=np.loadtxt("../distance_mat/" + file_name.replace(".","_distmat_train-all-contrastive_."))
print('Loaded ' + "../distance_mat/" + file_name.replace(".","_distmat_train-all-contrastive_."))
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + file_name.replace(".","_distmat_train-all-contrastive_."))

##
## Calculates the binning that we used for the dataset using the force-13 method
##
print dist
hist, bins = np.histogram(dist, 13)
print "Bins 13 :"
print bins
print hist
## Threshold the bins (eliminate bins having less than 100 elements)
bins=np.delete(bins,np.argwhere(hist<100))
hist=np.delete(hist,np.argwhere(hist<100))
print "Bins after thresholding : "
print bins
print hist
print bins.shape
print hist.shape
# Create empty result array to store the data
# The result array is a 10 by 10 by 1000 array (10 bins x 10 bins x (10 voters x 100 pairs))
results = []
for i in range(10) :
  ar = []
  for j in range(10):
    ar.append([])
  results.append(ar)

# Create empty agreement array to store the agreement percentage
# The result array is a 10 by 10 by 100 array (10 bins x 10 bins x (100 pairs))
agreement_percentages = []
for i in range(10) :
  ar = []
  for j in range(10):
    ar.append([])
  agreement_percentages.append(ar)

# Create empty map and confidence arrays to use for the graph 
results_2d = np.array([[0.0 for j in xrange(10)] for i in xrange(10)])
confidence = np.array([[0.0 for j in xrange(10)] for i in xrange(10)])
agreement = np.array([[0.0 for j in xrange(10)] for i in xrange(10)])
pairs_bins = {}

# The pairs_votes array is a 450 by 450 by 10
pairs_votes = {}

# The pairs_agreement array is a 450 by 450 by 10
pairs_agreement = {}

# The pairs_agreement array is a 450 by 450 by 10
pairs_results = {}

# The pairs whose agreement score is lower than 80%
pairs_to_ignore = []

# Open the Mechanical Turk result and start reading data
with open('../input/external_hit.results', 'rb') as f:
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
            # get the scores to determine the bins
            first_pair_score = float(row[header.index("Answer.scores_" + str(2*i))])
            second_pair_score = float(row[header.index("Answer.scores_" + str(2*i + 1))])
            # generate a unique name for each pair
            first_pair_first_image_name = row[header.index("Answer.imagename_" + str(4*i))]
            first_pair_second_image_name = row[header.index("Answer.imagename_" + str(4*i+1))]
            second_pair_first_image_name = row[header.index("Answer.imagename_" + str(4*i+2))]
            second_pair_second_image_name = row[header.index("Answer.imagename_" + str(4*i+3))]
            first_pair_name = min(first_pair_first_image_name, first_pair_second_image_name) + max(first_pair_first_image_name, first_pair_second_image_name)
            second_pair_name = min(second_pair_first_image_name, second_pair_second_image_name) + max(second_pair_first_image_name, second_pair_second_image_name)
            # calculate the bins for each of the two pairs
            first_pair_bin = bisect.bisect_left(bins, first_pair_score)
            second_pair_bin = bisect.bisect_left(bins, second_pair_score) 
            # save the bins for each pairs
            pairs_bins[first_pair_name] = first_pair_bin
            pairs_bins[second_pair_name] = second_pair_bin
            # if the bins are valid
            if first_pair_bin>=0 and second_pair_bin>=0 :
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
results = np.array([[0.0 for j in xrange(10)] for i in xrange(10)])
# Append the found percentages to the bins of each of the pairs
for first_pair_name, second_pairs in pairs_results.iteritems():
  for second_pair_name, vote in second_pairs.iteritems():
    first_pair_bin = pairs_bins[first_pair_name]
    second_pair_bin = pairs_bins[second_pair_name]
    agreement_pecentage = pairs_agreement[first_pair_name][second_pair_name]
    if vote == 1 and agreement_pecentage >= threshold :
      results[first_pair_bin][second_pair_bin] = results[first_pair_bin][second_pair_bin] + 1
    
print results
count=0
for first_pair_name, second_pairs in pairs_results.iteritems():
  for second_pair_name, vote in second_pairs.iteritems():
    if pairs_bins[first_pair_name]== 2 or pairs_bins[second_pair_name] == 2 :
      count = count + 1
print "Result : " + str(count)
#mark the diagonal (same bin comparison) to be ignored
#print results_2d
fig, ax = plt.subplots()
cmap = plt.cm.jet
cmap.set_bad('w',1.)
# shows the bins as labels for the axis
plt.xticks(range(10), np.round(bins, 2))
plt.yticks(range(10), np.round(bins, 2))
# offset the labels of the axis
ax.set_xticks(np.arange(10) + 0.5, minor=False)
ax.set_yticks(np.arange(10) + 0.5, minor=False)
# place the axis on the top and on the left
ax.xaxis.tick_top()
ax.invert_yaxis()
# generate the graph
plt.pcolor(results, vmin=0, vmax=100)
plt.colorbar()
# # show the agreement percentage on each of the cells
if (threshold == 0):
  plt.title('Number of tests concluding the row is more similar than the column', y=1.08)
else :
  plt.title('Number of tests concluding the row is more similar than the\n column (ignoring tests with < ' + str(threshold) + '% agreement)', y=1.03)
for y in range(10):
    for x in range(10):
        if x != y :
          plt.text(x + 0.5, y + 0.5, '%d' % results[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

plt.show()
