'''

This script visualizes improvement between two trained models
It shows the pairs that the computer can now say are similar
It also shows statistics about those pairs

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

DATASET = "feret"
file_name = "all_unique_" + DATASET + "_names.csv"
threshold = 80 # only consider pairs with agreement level above this number
FIRST_MODEL_NAME = "initial.ckpt"
SECOND_MODEL_NAME = "2nd_continue_norm_all.ckpt"
FIRST_MODEL_NAME_NO_EXT = FIRST_MODEL_NAME.replace(".ckpt", "")
SECOND_MODEL_NAME_NO_EXT = SECOND_MODEL_NAME.replace(".ckpt", "")
HIT_RESULTS = "external_hit" + ("_10k" if DATASET == "10k" else "") + ".results"
SHOW_INFO = True
output_file = "../results/second_experiment_results/" + DATASET + "_improvement_" + str(threshold) + ".html"

####################
##     Methods    ##
####################

def getFirstDistance(name1,name2):
  return distsq1[getIndex(name1)][getIndex(name2)]

def getSecondDistance(name1,name2):
  return distsq2[getIndex(name1)][getIndex(name2)]

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
      if key == "gender":
        result = result + str(key) + " : <b style='color:white;padding:3px;background:" + ("blue" if value=="Male" else "red") + "'>" + str(value) + "</b><br>"
      else :
        result = result + str(key) + " : " + str(value) + "<br>"
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

dist=np.loadtxt("../distance_mat/" + file_name.replace(".","_distmat_" + FIRST_MODEL_NAME_NO_EXT + "_."))
print('Loaded ' + "../distance_mat/" + file_name.replace(".","_distmat_" + FIRST_MODEL_NAME_NO_EXT + "_."))
distsq1=squareform(dist)
print('Finished processing ' + "../distance_mat/" + file_name.replace(".","_distmat_" + FIRST_MODEL_NAME_NO_EXT + "_."))

dist=np.loadtxt("../distance_mat/" + file_name.replace(".","_distmat_" + SECOND_MODEL_NAME_NO_EXT + "_."))
print('Loaded ' + "../distance_mat/" + file_name.replace(".","_distmat_" + SECOND_MODEL_NAME_NO_EXT + "_."))
distsq2=squareform(dist)
print('Finished processing ' + "../distance_mat/" + file_name.replace(".","_distmat_" + SECOND_MODEL_NAME_NO_EXT + "_."))

#Initialize arrays 
pairs_scores1 = {}
pairs_scores2 = {}
pairs_votes = {}
pairs_agreement = {}
pairs_results = {}

# Open the Mechanical Turk result and start reading data
with open('../input/' + HIT_RESULTS, 'rb') as f:
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
            # get the scores to determine the bins
            first_pair_score1 = getFirstDistance(first_pair_first_image_name, first_pair_second_image_name)
            second_pair_score1 = getFirstDistance(second_pair_first_image_name, second_pair_second_image_name)
            first_pair_score2 = getSecondDistance(first_pair_first_image_name, first_pair_second_image_name)
            second_pair_score2 = getSecondDistance(second_pair_first_image_name, second_pair_second_image_name)
              #first_pair_score = float(row[header.index("Answer.scores_" + str(2*i))])
              #second_pair_score = float(row[header.index("Answer.scores_" + str(2*i + 1))])
            #store the score for each pair
            pairs_scores1[first_pair_name] = first_pair_score1
            pairs_scores1[second_pair_name] = second_pair_score1
            pairs_scores2[first_pair_name] = first_pair_score2
            pairs_scores2[second_pair_name] = second_pair_score2
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
    #Get the scores for every pair (already stored when reading the result file)
    first_pair_score1 = pairs_scores1[first_pair_name]
    second_pair_score1 = pairs_scores1[second_pair_name]
    first_pair_score2 = pairs_scores2[first_pair_name]
    second_pair_score2 = pairs_scores2[second_pair_name]
    #Get the agreement percentage among voters for this comparison (already stored when reading the result file)
    agreement_percentage = pairs_agreement[first_pair_name][second_pair_name]
    if SHOW_INFO:
      #Read each of the images' information (gender, race, glasses, beard, mustache, etc.)
      first_pair_first_image_info = getImageInfo(first_pair_first_image_name)
      first_pair_second_image_info = getImageInfo(first_pair_second_image_name)
      second_pair_first_image_info = getImageInfo(second_pair_first_image_name)
      second_pair_second_image_info = getImageInfo(second_pair_second_image_name)
    #Calculate what the computer voted for
    computer_vote1 = 1 if first_pair_score1 <= second_pair_score1 else 0
    computer_vote2 = 1 if first_pair_score2 <= second_pair_score2 else 0
    # Compute statistics and show example images that fit the criteria (less or more than a certain agreement)  
    if vote == 1 and computer_vote1 == 0  and computer_vote2 == 1 and agreement_percentage >= threshold :
      #Count number of tests to consider
      total_tests = total_tests + 1
      if SHOW_INFO:
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
      agreement_page.write("<div class='test'>Agreement : " + str(agreement_percentage) + "%<br>\n")
      agreement_page.write("""<div class='container """ + ("votedfor" if vote == 1 else "notvotedfor") + """'>
                           Old score : """ + str(first_pair_score1) + """ <br>
                           New score : """ + str(first_pair_score2) + """ <br>
                           <img height='150px' src='faces_png/""" + first_pair_first_image_name + """'/> vs <img height='150px' src='faces_png/""" + first_pair_second_image_name + """'/><br>"""
                           + ("""<div class='info'>""" + printDict(first_pair_first_image_info) + """</div>&nbsp; &nbsp;\n
                           <div class='info'>""" + printDict(first_pair_second_image_info) + """</div>\n""" if SHOW_INFO else "") + """
                           </div>&nbsp; &nbsp;\n""")
      agreement_page.write("""<div class='container """ + ("votedfor" if vote == 0 else "notvotedfor") + """'>
                           Old score : """ + str(second_pair_score1) + """ <br>
                           New score : """ + str(second_pair_score2) + """ <br>
                           <img height='150px' src='faces_png/""" + second_pair_first_image_name + """'/> vs <img height='150px' src='faces_png/""" + second_pair_second_image_name + """'/><br>"""
                           + ("""<div class='info'>""" + printDict(second_pair_first_image_info) + """</div>&nbsp; &nbsp;\n
                           <div class='info'>""" + printDict(second_pair_second_image_info) + """</div>\n""" if SHOW_INFO else "") + """</div>\n
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
  modified.write("<h2>Tests with more than " + str(threshold) + " agreement that improved after training (" + str(total_tests) + " tests)</h2>\n")
  if SHOW_INFO:
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
  modified.write("<br>\n")
  modified.write("The <span style='background:#7ade7a'>green pair</span> is the one the trained algorithm and the voters chose as more similar <br> \n")
  modified.write("The <span style='background:#ffffff'>white pair</span> is the one the untrained algorithm chose as more similar <br> \n" + data)
