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

file_name = "all_unique_feret_names.csv"
threshold = int(sys.argv[1]) # only consider pairs with agreement level above this number
more_or_less = sys.argv[2] # consider pairs with agreement higher than or less than the threshold
output_file = "../results/mech_turk_page/tests_" + ("high" if more_or_less == "more" else "low") + "_agreement_" + str(threshold) + ".html"

####################
##     Methods    ##
####################

def getDistance(id1,id2):
  return distsq[id1][id2]

def getImageInfo(imgid):
  result = {}
  imgid = imgid.replace(".png", ".txt")
  imgid_person = imgid.split('_')[0]
  with open('../../feret_info/name_value/' + imgid_person + '/' + imgid_person + '.txt', 'rb') as csvfile:
     info = csv.reader(csvfile, delimiter='=')
     for row in info:
         result[row[0]] = row[1]
  with open('../../feret_info/name_value/' + imgid_person + '/' + imgid, 'rb') as csvfile:
     info = csv.reader(csvfile, delimiter='=')
     for row in info:
         result[row[0]] = row[1]
  return result
        
def printDict(dictionary):
  result = ""
  interesting = ['gender', 'race', 'beard', 'glasses', 'mustache']
  for key,value in dictionary.iteritems():
    if key in interesting : 
        result = result + str(key) + " : " + str(value) + "<br/>"
  return result

def initStatistics(container):
    container.setdefault('all_same', 0)
    container.setdefault('same_to_same', 0)
    container.setdefault('different_to_different', 0)
    container.setdefault('different_to_same', 0)
    
def addStatistics(container, info_string, first_pair_first_image_info, first_pair_second_image_info, second_pair_first_image_info, second_pair_second_image_info):
  if first_pair_first_image_info[info_string] == first_pair_second_image_info[info_string] == second_pair_first_image_info[info_string] == second_pair_second_image_info[info_string]:
    container['all_same'] += 1
  elif (first_pair_first_image_info[info_string] == first_pair_second_image_info[info_string]) and (second_pair_first_image_info[info_string] == second_pair_second_image_info[info_string]):
    container['same_to_same']  += 1
  elif (first_pair_first_image_info[info_string] != first_pair_second_image_info[info_string]) and (second_pair_first_image_info[info_string] != second_pair_second_image_info[info_string]):
    container['different_to_different']  += 1
  elif (((first_pair_first_image_info[info_string] == first_pair_second_image_info[info_string]) and (second_pair_first_image_info[info_string] != second_pair_second_image_info[info_string])) or ((first_pair_first_image_info[info_string] != first_pair_second_image_info[info_string]) and (second_pair_first_image_info[info_string] == second_pair_second_image_info[info_string]))):
    container['different_to_same'] += 1

def generateDescriptionImages(first_value, second_value, case) :
  return {
        'all_same': first_value + " " + first_value + " vs " + first_value + " " + first_value + " OR " + second_value + " " + second_value + " vs " + second_value + " " + second_value,
        'same_to_same': first_value + " " + first_value + " vs " + second_value + " " + second_value,
        'different_to_different': first_value + " " + second_value + " vs " + first_value + " " + second_value,
        'different_to_same': first_value + " " + second_value + " vs " + first_value + " " + first_value + " OR " + first_value + " " + second_value + " vs " + second_value + " " + second_value,
    }[case]

def generateDescription(container, first_value, second_value, total):
  result = ""
  for key, value in container.iteritems():
    result += generateDescriptionImages(first_value, second_value, key) + " : " + str(value) + "/" + str(total) + " : " + str(round(100*value/float(total),2)) + "%<br> \n"
  return result

##
## Load distance matrix and calculate its expanded form
##

dist=np.loadtxt("../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
print('Loaded ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + file_name.replace(".","_euclidean_distmat."))

##
## Calculates the binning that we used for the dataset using the force-13 method
##

hist, bins = np.histogram(dist, 13)

## Threshold the bins (eliminate bins having less than 100 elements)
bins=np.delete(bins,np.argwhere(hist<100))
hist=np.delete(hist,np.argwhere(hist<100))

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
            first_pair_name = min(first_pair_first_image_name, first_pair_second_image_name) + "#" + max(first_pair_first_image_name, first_pair_second_image_name)
            second_pair_name = min(second_pair_first_image_name, second_pair_second_image_name) + "#" + max(second_pair_first_image_name, second_pair_second_image_name)
            # calculate the bins for each of the two pairs
            first_pair_bin = bisect.bisect_left(bins, first_pair_score)
            second_pair_bin = bisect.bisect_left(bins, second_pair_score) 
            # save the bins for each pair
            pairs_bins[first_pair_name] = first_pair_bin
            pairs_bins[second_pair_name] = second_pair_bin
            #store the score for each pair
            pairs_scores[first_pair_name] = first_pair_score
            pairs_scores[second_pair_name] = second_pair_score
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
agreement_page= open(output_file, 'w')
agreement_page.write("""
<style>
.test {
    display:inline-block;
    background: #e8e8e8;
    border-radius: 4px;
    border: 1px solid #c7c7c7;
    padding: 10px;
    text-align:center
}

.container:first-child {
  margin-right:10px;
}
.votedfor {
    background: #7ade7a !important;
    padding: 10px;
    border: 1px solid green;
    border-radius: 5px;
    text-align:center;
    float:left;
}

.notvotedfor {
    background: white;
    padding: 10px;
    border: 1px solid gray;
    border-radius: 5px;
    text-align:center;
    float:left;
}
.clear {
float:none;
clear:both
}
.info {
display:inline-block;
padding:10px;
}
</style>
""")
# Initialize arrays to store statistics data
total_tests = 0
race_stat = {}
gender_stat = {}
glasses_stat = {}
mustache_stat = {}
beard_stat = {}
# Create keys for every statistics array
initStatistics(race_stat)
initStatistics(gender_stat)
initStatistics(glasses_stat)
initStatistics(mustache_stat)
initStatistics(beard_stat)
# Initialize other counters
asian_pair_exists = 0
asian_pair_was_chosen = 0
counter = 0
#Computer statistics
for first_pair_name, second_pairs in pairs_results.iteritems():
  for second_pair_name, vote in second_pairs.iteritems():
    #Get the images' names
    first_pair_first_image_name = first_pair_name.split("#")[0] + ".png"
    first_pair_second_image_name = first_pair_name.split("#")[1] + ".png"
    second_pair_first_image_name = second_pair_name.split("#")[0] + ".png"
    second_pair_second_image_name = second_pair_name.split("#")[1] + ".png"
    #Get the bins for every pair (already stored when reading the result file)
    first_pair_bin = pairs_bins[first_pair_name]
    second_pair_bin = pairs_bins[second_pair_name]
    #Get the scores for every pair (already stored when reading the result file)
    first_pair_score = pairs_scores[first_pair_name]
    second_pair_score = pairs_scores[second_pair_name]
    #Get the agreement percentage among voters for this comparison (already stored when reading the result file)
    agreement_pecentage = pairs_agreement[first_pair_name][second_pair_name]
    #Read each of the images' information (gender, race, glasses, beard, mustache, etc.)
    first_pair_first_image_info = getImageInfo(first_pair_first_image_name)
    first_pair_second_image_info = getImageInfo(first_pair_second_image_name)
    second_pair_first_image_info = getImageInfo(second_pair_first_image_name)
    second_pair_second_image_info = getImageInfo(second_pair_second_image_name)
    #Calculate what the computer voted for
    computer_vote = 1 if first_pair_bin <= second_pair_bin else 0
    # Compute statistics and show 100 example images that fit the criteria (less or more than a certain agreement)
    if (more_or_less=="more" and vote == 1 and agreement_pecentage >= threshold) or (more_or_less=="less" and vote == 1 and agreement_pecentage < threshold) :
      #Count number of tests to consider
      total_tests = total_tests + 1
      #Gender
      addStatistics(gender_stat, 'gender', first_pair_first_image_info, first_pair_second_image_info, second_pair_first_image_info, second_pair_second_image_info)
      #Race
      addStatistics(race_stat, 'race', first_pair_first_image_info, first_pair_second_image_info, second_pair_first_image_info, second_pair_second_image_info)
      if ((first_pair_first_image_info['race'] == first_pair_second_image_info['race'] == "Asian") or (second_pair_first_image_info['race'] == second_pair_second_image_info['race'] == "Asian")):
        asian_pair_exists += 1
      if (((first_pair_first_image_info['race'] == first_pair_second_image_info['race'] == "Asian") and vote == 1) or ((second_pair_first_image_info['race'] == second_pair_second_image_info['race'] == "Asian") and vote == 0)):
        asian_pair_was_chosen += 1
      #Glasses
      addStatistics(glasses_stat, 'glasses', first_pair_first_image_info, first_pair_second_image_info, second_pair_first_image_info, second_pair_second_image_info)
      #Beard
      addStatistics(beard_stat, 'beard', first_pair_first_image_info, first_pair_second_image_info, second_pair_first_image_info, second_pair_second_image_info)
      #Mustache
      addStatistics(mustache_stat, 'mustache', first_pair_first_image_info, first_pair_second_image_info, second_pair_first_image_info, second_pair_second_image_info)
      # Only show 100 examples to reduce page size  
      counter += 1
      if counter > 100:
        continue
      agreement_page.write("<div class='test'>Agreement : " + str(agreement_pecentage) + "%<br>\n")
      agreement_page.write("""<div class='container """ + ("votedfor" if vote == 1 else "notvotedfor") + """'>
                           """ + str(first_pair_score) + """ <br>
                           <img height='150px' src='faces_png/""" + first_pair_first_image_name + """'/> vs <img height='150px' src='faces_png/""" + first_pair_second_image_name + """'/><br>
                           <div class='info'>""" + printDict(first_pair_first_image_info) + """</div>&nbsp; &nbsp;\n
                           <div class='info'>""" + printDict(first_pair_second_image_info) + """</div>\n
                           </div>&nbsp; &nbsp;\n""")
      agreement_page.write("""<div class='container """ + ("votedfor" if vote == 0 else "notvotedfor") + """'>
                           """ + str(second_pair_score) + """ <br>
                           <img height='150px' src='faces_png/""" + second_pair_first_image_name + """'/> vs <img height='150px' src='faces_png/""" + second_pair_second_image_name + """'/><br>
                           <div class='info'>""" + printDict(second_pair_first_image_info) + """</div>&nbsp; &nbsp;\n
                           <div class='info'>""" + printDict(second_pair_second_image_info) + """</div>\n
                           </div>\n""")
      agreement_page.write("<div class='clear'></div>\n")
      agreement_page.write("</div><br><br>")

# Icons :

male = "<img height='24px' src='http://i.imgur.com/Evo5WmL.png'>"
female = "<img height='24px' src='http://i.imgur.com/hTx4d7G.png'>"

race1 = "<img height='24px' src='http://i.imgur.com/QLjKifo.png'>"
race2 = "<img height='24px' src='http://i.imgur.com/LhB8w6f.png'>"

glasses = "<img width='24px' src='http://i.imgur.com/uLFOKUw.png'>"
no_glasses = "<img width='24px' src='http://i.imgur.com/ZBLLsvP.png'>"

beard = "<img width='24px' src='http://i.imgur.com/vrbJh8j.png'>"
no_beard = "<img width='24px' src='http://i.imgur.com/TsaLPEy.png'>"

mustache = "<img width='24px' src='http://i.imgur.com/ugAXHJB.png'>"
no_mustache = "<img width='24px' src='http://i.imgur.com/iFJtjQD.png'>"

# Prepend the title and statistics data

with file(output_file, 'r') as original: data = original.read()
with file(output_file, 'w') as modified: 
  modified.write("<h2>Tests where there was " + more_or_less + " than " + str(threshold) + " agreement among voters (" + str(total_tests) + " tests)</h2>\n")
  modified.write("<br><h3>Gender statistics : </h3>\n")
  modified.write(generateDescription(gender_stat, male, female, total_tests))
  modified.write("<br><h3>Race statistics : </h3>\n")
  modified.write(generateDescription(race_stat, race1, race2, total_tests))
  modified.write(str(asian_pair_was_chosen) + " out of " + str(asian_pair_exists) + " times the asian pair was chosen as more similar (" + str(round(100*asian_pair_was_chosen/float(asian_pair_exists),2)) + "%  of times) <br><br>\n")
  modified.write("<br><h3>Glasses statistics : </h3>\n")
  modified.write(generateDescription(glasses_stat, glasses, no_glasses, total_tests))
  modified.write("<br><h3>Beard statistics : </h3>\n")
  modified.write(generateDescription(beard_stat, beard, no_beard, total_tests))
  modified.write("<br><h3>Mustache statistics : </h3>\n")
  modified.write(generateDescription(mustache_stat, mustache, no_mustache, total_tests))
  modified.write("<br><h2>100 examples of comparisons : </h2>\n")
  modified.write("<br>\n")
  modified.write("The <span style='background:#7ade7a'>green pair</span> is the one voters chose as more similar <br> \n")
  modified.write("<br>\n" + data)
