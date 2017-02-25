#
# Uses the results from the mechanical turk experiment to plot a graph
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
dist=np.loadtxt("../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
print('Loaded ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))

##
## Calculates the binning that we used for the dataset using the force-13 method
##

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

# The pairs_agreement array is a 450 by 450 by 10
pairs_agreement = {}

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
              # append 1 to the chosen pair's bin and 0 to the other pair's bin
              results[first_pair_bin][second_pair_bin].append(0 if chosen==2 else 1)
              results[second_pair_bin][first_pair_bin].append(1 if chosen==2 else 0)
              # append 1 to the chosen pair in the agreement percentages array
              pairs_agreement.setdefault(first_pair_name, {})
              pairs_agreement[first_pair_name].setdefault(second_pair_name, [])
              pairs_agreement[first_pair_name][second_pair_name].append(0 if chosen==2 else 1)
              # append 1 to the chosen pair in the agreement percentages array
              pairs_agreement.setdefault(second_pair_name, {})
              pairs_agreement[second_pair_name].setdefault(first_pair_name, [])
              pairs_agreement[second_pair_name][first_pair_name].append(0 if chosen==1 else 1)
              #print "first_pair_score : " + str(first_pair_score) + " bin : " + str(first_pair_bin) + "  = " + str(bins[first_pair_bin])
              #print "first_pair_score : " + str(second_pair_score) + " bin : " + str(second_pair_bin) + "  = " + str(bins[second_pair_bin])

# Compute agreement for each of the "pair of pairs" by counting the number of votes that got the majority
for first_pair_name, second_pairs in pairs_agreement.iteritems():
  for second_pair_name, votes in second_pairs.iteritems():
    pairs_agreement[first_pair_name][second_pair_name] = 100*float(np.bincount(pairs_agreement[first_pair_name][second_pair_name]).max())/len(pairs_agreement[first_pair_name][second_pair_name])
    if pairs_agreement[first_pair_name][second_pair_name] < 80:
      # add the unique name of the "pair of pairs" to the pairs to ignore since there was no agreement on it
      pairs_to_ignore.append(min(first_pair_name, second_pair_name) + max(first_pair_name, second_pair_name))

## Re-compute results ignoring pairs that weren't agreed upon
# Open the Mechanical Turk result and start reading data

# Empty the result array to store the data
# The result array is a 10 by 10 by 1000 array (10 bins x 10 bins x (10 voters x 100 pairs))
results = []
for i in range(10) :
  ar = []
  for j in range(10):
    ar.append([])
  results.append(ar)

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
            # if the bins are valid
            if first_pair_bin>=0 and second_pair_bin>=0 and (min(first_pair_name, second_pair_name) + max(first_pair_name, second_pair_name)) not in pairs_to_ignore :
              # read the chosen pair (1 or 2)
              chosen = int(row[header.index("Answer.chosen_" + str(i))])
              # append 1 to the chosen pair's bin and 0 to the other pair's bin
              results[first_pair_bin][second_pair_bin].append(0 if chosen==2 else 1)
              results[second_pair_bin][first_pair_bin].append(0 if chosen==1 else 1)
              #print "first_pair_score : " + str(first_pair_score) + " bin : " + str(first_pair_bin) + "  = " + str(bins[first_pair_bin])
              #print "first_pair_score : " + str(second_pair_score) + " bin : " + str(second_pair_bin) + "  = " + str(bins[second_pair_bin])

      
# Append the found percentages to the bins of each of the pairs
for first_pair_name, second_pairs in pairs_agreement.iteritems():
  for second_pair_name, votes in second_pairs.iteritems():
    first_pair_bin = pairs_bins[first_pair_name]
    second_pair_bin = pairs_bins[second_pair_name]
    agreement_percentages[first_pair_bin][second_pair_bin].append(pairs_agreement[first_pair_name][second_pair_name])
    
print agreement
## Calculate percentages
for i in range(10):
  for j in range(10):
    if i != j :
      #result_2d will have the sum of all the 1s (when i is chosen as the more similar pair) divided by the total number of votes
      results_2d[i][j]=float(np.bincount(results[i][j])[1])
      #confidence will have the same value except when 0s are dominant, we calculate the percentage of 0s rather than 1s
      #meaning that confidence will calculate the percentage of the winning choice
      #confidence[i][j]=100*float(np.bincount(results[i][j]).max())/len(results[i][j])
      #Average the agreement percentage
      agreement[i][j]= np.average(agreement_percentages[i][j])
#mark the diagonal (same bin comparison) to be ignored
results_2d[results_2d == 0.0] = np.nan
#print results_2d
fig, ax = plt.subplots()
#ignore the diagonal (same bin comparison)
masked_array = np.ma.array (results_2d, mask=np.isnan(results_2d))
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
plt.pcolor(masked_array, vmin=0, vmax=100)
plt.colorbar()
# # show the confidence percentage on each of the cells
plt.title('Confidence percentages (ignoring pairs with < 80% agreement)', y=1.08)
for y in range(10):
    for x in range(10):
        if x != y :
          plt.text(x + 0.5, y + 0.5, '%s' % np.round(results_2d[y, x],1) + '%',
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

# show the agreement percentage on each of the cells
# plt.title('Agreement percentage', y=1.08)
# for y in range(10):
#     for x in range(10):
#         if x != y :
#           plt.text(x + 0.5, y + 0.5, '%s' % np.round(agreement[y, x],1) + '%',
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  )
plt.show()
