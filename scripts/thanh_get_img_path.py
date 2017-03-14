import os
import glob


def getImgDir():
  file = open("../input/faces_5k_crop_images.csv",'w')
  for img in sorted(glob.glob("/home/vut/FaceSimilarity/faces_5k_crop/*.png")):
    file.write(img + '\n')
  file.close()

getImgDir()