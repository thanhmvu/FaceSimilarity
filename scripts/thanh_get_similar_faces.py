
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

DATASET_DIR = "../../faces_5k_crop/"
NAME_FILE = "faces_5k_names.csv"
OUTPUT_DIR = "../results/"
OUTPUT_HTML = "results_5k.html"

# DATASET_DIR = os.path.abspath(DATASET_DIR)
# OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

#finds and returns the index at unique_feret_names corresponding to the image's file name
def getFilename(index):
  f=open("../input/" +NAME_FILE,'rU')
  reader=csv.reader(f,delimiter='\n')
  counter = 0
  for img in reader:
    if (counter == index) :
      return img[0]
    counter = counter + 1

##
## Load distance matrix and calculate its expanded form
##
print('Loading ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_.") + " ...")
dist=np.loadtxt("../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
print('Loaded ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
# print(dist)

distsq=squareform(dist)
print('Finished processing ' + "../distance_mat/" + NAME_FILE.replace(".","_distmat_initial_."))
print(distsq)


mostSim = {}
NUM_MOST_SIM = 6  
for i in range(len(distsq)):
  dist_i = distsq[i]
  
  dist_i[i] = float("inf")
  most_sim_idx = np.argpartition(dist_i,NUM_MOST_SIM)[:NUM_MOST_SIM]
  dist_i[i] = 0
  
  # add to output map
  imgURL = getFilename(i)
  print (imgURL)
  mostSim[imgURL] = [(getFilename(j),round(dist_i[j],4)) for j in most_sim_idx]
  mostSim[imgURL] = sorted(mostSim[imgURL], key=lambda x: x[1])
#   print mostSim[imgURL]

# for i,face in enumerate(mostSim):
#   print i, face


  
##
## Display result.html
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
			resultTable += "<th>" +htmlImg(DATASET_DIR + img,"transparent") + "<br>"+ srcface + "<br> Dist: "+ str(dist) +"</th>"
		resultTable += "</tr>"
    
  resultTable += "</table>\n"

  contents = """<h2>Face Similarity</h2> <br> <br>""" +resultTable
	
  save_css(dir)
  save_html(contents,dir)

# ============================== Main ==============================
print_html(mostSim, OUTPUT_DIR)
