
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

DATASET_DIR = "../../faces_5k_cropped/"
NAME_FILE = "faces_5k_AZ_names.csv"
OUTPUT_DIR = "../results/"
OUTPUT_HTML = "results_5k_AZ.html"

# DATASET_DIR = os.path.abspath(DATASET_DIR)
# OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

# Finds and returns the index at unique_feret_names corresponding to the image's file name
def getFilename(index):
  f=open("../input/" +NAME_FILE,'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (counter == index) :
      return img[0]
    counter = counter + 1

##
## ============================== Find similar faces ==============================
##

# Load distance matrix
print('Loading ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_.") + " ...")
dist=np.loadtxt("../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
print('Loaded ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
# print(dist)

# Calculate matrix's expanded form
distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
print(distsq)

# Find n most similar faces
mostSim = {}
NUM_MOST_SIM = 6  
for i in range(len(distsq)):
  srcImg = getFilename(i)
  dist_i = distsq[i]
  
	# sort dist_i and find the n most similar
  most_sim_idx = []
  for j in np.argsort(dist_i):
    srcName = srcImg.split("_")[0]
    simName = getFilename(j).split("_")[0]
    if simName != srcName: # discard images of people of same first name
	    most_sim_idx.append(j)
    if len(most_sim_idx) == NUM_MOST_SIM:
      break
  
  # add to output map
  print (srcImg)
  mostSim[srcImg] = [(getFilename(j),round(dist_i[j],4)) for j in most_sim_idx]
  mostSim[srcImg] = sorted(mostSim[srcImg], key=lambda x: x[1])

	
  
##
## ============================== Display output webpage ==============================
##

def save_css(dir):
  f = open(dir + "style.css","w")
  contents = """
	<style >
	table {
	    font-family: arial, sans-serif;
	    border-collapse: collapse;
	    width: 100%;
	}

	td, th {
		white-space: nowrap;
	    border: 1px solid #dddddd;
	    text-align: left;
	    padding: 8px;
	    font-size: 100%;
	}

	tr:nth-child(even) {
	    background-color: #dddddd;
	}
	</style>"""
  f.write(contents)
  f.close()

def save_html(contents, dir):
  f = open(dir + OUTPUT_HTML,"w")

  display = """
  <!DOCTYPE html>
  <html>
  <head>
    <link rel="stylesheet" href="style.css">
  </head>
  <body>
  """ + contents + "</body>\n</html>\n"

  f.write(display)
  f.close()

def htmlImg(img, borderColor):
  h = 100
  b = 5
  return "<img src=\""+ img + "\" alt=\"Class " + img + "\" style=\"height:"+`h`+"px; border:"+`b`+"px solid "+ borderColor +";\">"

def print_html(simFaces, dir):
  resultTable = """
  <table>
    <tr>
      <th>Image Idx</th>
      <th>Source Face</th>
      <th>Similar Faces</th>
    </tr>
  """

  for i, srcface in enumerate(simFaces):
		resultTable += """
		<tr>
			<th>"""+ str(i) +"""</th>
			<th>"""+ htmlImg(DATASET_DIR + srcface,"transparent") + "<br>"+ srcface +"""</th>
		"""
		for simface in simFaces[srcface]:
			img = simface[0]
			dist = simface[1]
			resultTable += "<th>" +htmlImg(DATASET_DIR + img,"transparent") + "<br>"+ img + "<br> Dist: "+ str(dist) +"</th>"
		resultTable += "</tr>"
    
  resultTable += "</table>\n"

  contents = """<h2>Face Similarity</h2> <br> <br>""" +resultTable
	
  save_css(dir)
  save_html(contents,dir)

# Generate output webpage
print_html(mostSim, OUTPUT_DIR)
