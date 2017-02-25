#Makes an input file that has names of all the files in the database

import cv2
import numpy as np
import csv
import sys
import os as os

filenames = next(os.walk('../../' + sys.argv[1]))[2]
output=open("../input/all_unique_" + sys.argv[1] + "_names.csv",'w')
for file in filenames:
  output.write(file + '\n')
output.close()