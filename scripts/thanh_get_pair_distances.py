#
# Uses the computed description vector to create an output file for Mechanical Turk (does not need to re-process the images)
# The script uses multiple processes :
# 1 . Reads the vector file and computes the distances between every combination of pairs (stored in dist and distsq)
# 2 . Calculates partitions (bins) using multiple algorithms to determine the best partitioning (/distances/*_bins.txt as well
# as /histograms/*_method.png)
# 3 . After determining the suitable bin algorithm (manual input), the script thresholds the bins to 100 elements per bin 
# then creates tests and saves the result in distances/*_result.txt
#
# @arg unique names csv file (ex all_unique_feret_names.csv)
# @returns distances/*_bins.txt
# @returns histograms/*_method.png
# @returns distances/*_result.txt

import csv
# order=[7,1,2,3,4,5,6,0,8,9,10];
# import sys
# sys.path=[sys.path[i] for i in order]
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

#finds and returns the index at unique_feret_names corresponding to the image's file name
def getIndex(filename):
  f=open("../input/" +sys.argv[1],'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (img[0] == filename) :
      break
    counter = counter + 1
  return counter

#finds and returns the index at unique_feret_names corresponding to the image's file name
def getFilename(index):
  f=open("../input/" +sys.argv[1],'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (counter == index) :
      return img[0]
    counter = counter + 1

##
## Load distance matrix and calculate its expanded form
##

dist=np.loadtxt("../distance_mat/" + sys.argv[1].replace(".","_euclidean_distmat."))
print('Loaded ' + "../distance_mat/" + sys.argv[1].replace(".","_euclidean_distmat."))
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + sys.argv[1].replace(".","_euclidean_distmat."))
#make necessary folder for results
if not os.path.exists('../results/mech_turk_page'):
    os.makedirs('../results/mech_turk_page')
    os.makedirs('../results/mech_turk_page/faces_png')
    os.makedirs('../results/mech_turk_page/faces_ppm')
    os.makedirs('../results/mech_turk_page/faces_original_png')
    os.makedirs('../results/mech_turk_page/faces_original_ppm')

##
## Calculate the partitions (binning) using various estimator algorithms and save the figures for comparison
##

# print('Saving statistics about the files')
# output = open("../distances/"+ sys.argv[1].replace(".csv","_bins.txt"), 'w')
# output.write("Number of elements : " + str(dist.shape))
# output.write("\n")
# output.write("Max : " + str(np.max(dist)))
# output.write("\n")
# output.write("Min : " + str(np.min(dist)))
# output.write("\n")
# print('Estimating bins using various partitioning algorithms')
# estimators=[10, 11, 12, 13, 'fd', 'auto', 'doane', 'scott', 'rice', 'sqrt', 'sturges']
# for estimator in estimators:
#   output.write("\n")
#   output.write("Using " + str(estimator))
#   output.write("\n")
#   hist, bin_edges = np.histogram(dist, estimator)
#   output.write("Hist : ")
#   output.write("\n")
#   output.write(','.join(str(x) for x in hist))
#   output.write("\n")
#   output.write("bin_edges")
#   output.write("\n")
#   output.write(','.join(str(x) for x in bin_edges))
#   output.write("\n")
#   ## Save a plot of the estimator to /histograms
#   fig, ax = plt.subplots()
#   plt.hist(dist, estimator)
#   plt.axhline(y=1000, color='red')
#   fig.savefig("../histograms/"+ sys.argv[1].replace(".","_" + str(estimator) + ".png"))

##
## Best binning is "force-13", create array that has a list of pairs corresponding to every bin and save it
##

hist, bins = np.histogram(dist, 13)
print bins
print hist

# Threshold bins (eliminate bins having less than 100 elements)
bins=np.delete(bins,np.argwhere(hist<100))
hist=np.delete(hist,np.argwhere(hist<100))
print bins
print hist
# Fill array with image ids for every bin
allbins=[]
lastbin=bins[0]

# Store unique pairs in their bins
for bin in bins[1:]:
  all_bin_images = np.argwhere((distsq>lastbin) & (distsq<bin))
  allbins.append(list(set(tuple(sorted(l)) for l in all_bin_images)))
  lastbin=bin

# take 100 random pairs from every bin
for bin, bin_value in enumerate(bins[:-1]) :
  np.random.shuffle(allbins[bin])
  allbins[bin] = allbins[bin][:100]

# store the binned pairs in a text file (100 per bin) and copy the images to the page's folder
binned_pairs_out= open("../results/mech_turk_page/"+  sys.argv[1].replace(".csv","_binned_pairs.txt"), 'w')
used_faces = []
for bin, bin_value in enumerate(bins[:-1]) :
  binned_pairs_out.write("\n\n" + str(bin_value) + " to " + str(bins[bin+1]) + ": (" + str(len(allbins[bin])) + " faces)\n\n")
  for first_image, second_image in allbins[bin]:
    first_image_name = getFilename(first_image)
    second_image_name = getFilename(second_image)
    pair_score = distsq[first_image][second_image]
    binned_pairs_out.write(str(first_image_name) + ", " + str(second_image_name) + "," + str(pair_score) + "\n")
    if first_image_name not in used_faces:
      used_faces.append(first_image_name)
      copyfile('../../feret/' + first_image_name, '../results/mech_turk_page/faces_original_ppm/' + first_image_name) # original ppm
      copyfile('../../feret/' + first_image_name, '../results/mech_turk_page/faces_original_png/' + first_image_name.replace(".ppm", ".png")) # original png
      copyfile('../../feret_cropped_opencv/' + first_image_name.replace(".ppm", ".png"), '../results/mech_turk_page/faces_ppm/' + first_image_name) # cropped ppm
      copyfile('../../feret_cropped_opencv/' + first_image_name.replace(".ppm", ".png"), '../results/mech_turk_page/faces_png/' + first_image_name.replace(".ppm", ".png")) # cropped png
    if second_image_name not in used_faces:
      used_faces.append(second_image_name)
      copyfile('../../feret/' + second_image_name, '../results/mech_turk_page/faces_original_ppm/' + second_image_name) # original ppm
      copyfile('../../feret/' + second_image_name, '../results/mech_turk_page/faces_original_png/' + second_image_name.replace(".ppm", ".png")) # original png
      copyfile('../../feret_cropped_opencv/' + second_image_name.replace(".ppm", ".png"), '../results/mech_turk_page/faces_ppm/' + second_image_name) # cropped ppm
      copyfile('../../feret_cropped_opencv/' + second_image_name.replace(".ppm", ".png"), '../results/mech_turk_page/faces_png/' + second_image_name.replace(".ppm", ".png")) # cropped png
print('Finished writing ' + "../results/mech_turk_page/"+ sys.argv[1].replace(".csv","_binned_pairs.txt"))

# store the unique names of all used faces
np.savetxt("../results/mech_turk_page/"+  sys.argv[1].replace(".csv","_used_faces.txt"), used_faces, fmt='%s')
print('Finished writing ' + "../results/mech_turk_page/"+ sys.argv[1].replace(".csv","_used_faces.txt"))

## Create the Mechanical Turk result file and visualization for the Mech Turk file and the binned pairs

print('Generating mechanical turk file & page')

text_file = open("../results/mech_turk_page/"+ sys.argv[1].replace(".csv","_mech_turk_result.txt"), 'w')
index_page = open("../results/mech_turk_page/index.html", 'w')

index_page.write('<h1>Feret Faces Database (used ' + str(len(used_faces)) + ' images)</h1>')
index_page.write('<a href="all_unique_feret_names_mech_turk_result.txt">Download result text file</a><br>')
index_page.write('<a href="all_unique_feret_names_binned_pairs.txt">Download list of binned pairs</a><br>')
index_page.write('<a href="all_unique_feret_names_used_faces.txt">Download list of used faces</a><br>')
index_page.write('<a href="faces_png.zip">Download used images (cropped, PNG files)</a><br>')
index_page.write('<a href="faces_ppm.zip">Download used images (cropped, PPM files)</a><br>')
index_page.write('<a href="faces_original_ppm.zip">Download used images (original, PNG files)</a><br>')
index_page.write('<a href="faces_original_ppm.zip">Download used images (original, PPM files)</a><br><br>')

index_page.write('<h2>Selected Binned Images Visualization</h2>')
for bin, bin_value in enumerate(bins[:-1]) :
  index_page.write(str(bin_value) + " to " + str(bins[bin+1]) + ": (" + str(len(allbins[bin])) + " faces) <a href='from_" + str(bin_value) + "_to_" + str(bins[bin+1]) + ".html'>See Faces</a><br>")
  bins_page = open("../results/mech_turk_page/from_" + str(bin_value) + "_to_" + str(bins[bin+1]) + ".html", 'w')
  bins_page.write("<h2>" + str(bin_value) + " to " + str(bins[bin+1]) + "</h2>")
  for first_image, second_image in allbins[bin]:
    first_image_name = getFilename(first_image)
    second_image_name = getFilename(second_image)
    pair_score = distsq[first_image][second_image]
    bins_page.write(first_image_name + ' vs ' + second_image_name + ' score : ' + str(pair_score) + '<br>')
    bins_page.write('Cropped input : <br>\n')
    bins_page.write('<img src="faces_png/' + first_image_name.replace(".ppm", ".png") + '"> vs <img src="faces_png/' + second_image_name.replace(".ppm", ".png") + '">\n<br><br>')

index_page.write('<br><h2>Mechanical Turk Job Visualization</h2>')

for bin1, bin1_value in enumerate(bins[:-1]):
  if(bin1==0): continue
  bin1_images = allbins[bin1]
  np.random.shuffle(bin1_images)
  bin1_images=bin1_images[:100]
  for bin2, bin2_value in enumerate(bins[:bin1]):
    text_file.write('high_range_low_range_' + str(bin1_value) + '_' + str(bin2_value) + '\n')
    bin2_images = allbins[bin2]
    np.random.shuffle(bin2_images)
    bin2_images=bin2_images[:100]
    index_page.write("Scores : " + str(bin1_value) + " vs " + str(bin2_value) + ": <a href='high_range_low_range_" + str(bin1_value) + "_" + str(bin2_value) + ".html'>See faces</a><br/>")
    faces_page = open("../results/mech_turk_page/high_range_low_range_" + str(bin1_value) + "_" + str(bin2_value) + ".html", 'w')
    faces_page.write("<h2>Scores : " + str(bin1_value) + " vs " + str(bin2_value) + "</h2>")
    for first_pair, second_pair in zip(bin1_images, bin2_images):
      first_pair_first_image = getFilename(first_pair[0])
      first_pair_second_image = getFilename(first_pair[1])
      second_pair_first_image = getFilename(second_pair[0])
      second_pair_second_image = getFilename(second_pair[1])
      first_pair_score = str(distsq[first_pair[0]][first_pair[1]])
      second_pair_score = str(distsq[second_pair[0]][second_pair[1]])
      text_file.write(first_pair_first_image + ',' + first_pair_second_image + ',' + second_pair_first_image + ',' + second_pair_second_image + ',' + first_pair_score + ',' + second_pair_score + '\n');
      faces_page.write('<img src="faces_png/' + first_pair_first_image.replace(".ppm", ".png") + '"> vs <img src="faces_png/' + first_pair_second_image.replace(".ppm", ".png") + '">  (' + first_pair_score + ') &nbsp; &nbsp &nbsp;')
      faces_page.write('<img src="faces_png/' + second_pair_first_image.replace(".ppm", ".png") + '"> vs <img src="faces_png/' + second_pair_second_image.replace(".ppm", ".png") + '"> (' + second_pair_score + ')  <br><br>')
