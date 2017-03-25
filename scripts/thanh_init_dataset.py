import numpy as np
import cv2
import glob
import shutil
import os
# from matplotlib import pyplot as plt


# ================================= METHODS ================================ #
def detectAndDisplay3(imgName):
	face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')

	img = cv2.imread(imgName)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# cv2.equalizeHist(gray, gray)  

	# return 3 vectors of objects, rejectLevels, levelWeights
	results = face_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
	faces = results[0]
	print("Results:")
	print(results)
	print("Number of faces: %d" % (len(faces)))

	for i,face in enumerate(faces):
		print("detectAndDisplay2 "+str(i)+": ")
		print(face)
		(x,y,w,h) = face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def detectAndDisplay2(imgName):
	face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')

	img = cv2.imread(imgName)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# cv2.equalizeHist(gray, gray)  

	# return a vector of objects and a vector of corresponding numDetections
	# An object’s number of detections is the number of neighboring positively classified rectangles that were joined together to form the object.
	results = face_cascade.detectMultiScale2(gray, 1.05, 5)
	faces = results[0]
	print("Results:")
	print(results)
	print("Number of faces: %d" % (len(faces)))

	for i,face in enumerate(faces):
		print("detectAndDisplay2 "+str(i)+": ")
		print(face)
		(x,y,w,h) = face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def detectAndDisplay(imgName):
	face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')

	img = cv2.imread(imgName)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray, gray) # helps improve the detection

	# return a list of detection boxes
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print("Number of faces: %d" % (len(faces)))

	for i,face in enumerate(faces):
		print("detectAndDisplay "+str(i)+": ")
		print(face)
		(x,y,w,h) = face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def detect(imgName):
	face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')

	img = cv2.imread(imgName)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray, gray) # helps improve the detection

	# return a list of detection boxes
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	return len(faces)

# =================================== MAIN ================================== #
SRC_NAMES = "../../database/Names100Dataset/names100.txt"
DB_DIR = "../../database/"
SRC_DIR = "Names100Dataset/Names100_Images/"
DST_DIR = "storage/good_faces/"
BAD_DIR = "storage/bad_faces/"

# import names from text file in Names100Dataset
# each name correspond to 800 images
all_names = [line.strip() for line in open(SRC_NAMES,"r+")]
all_names = sorted(all_names)
print(all_names)

# ======== Test individual images ======== #
# imgName = "../test/face_sim_test.jpg"
# print(imgName)
# detectAndDisplay(imgName)

# ======== Detect and move images ======== #
used_names = len(all_names)
image_per_name = 100 #50
# for i,imgName in enumerate(imgNames):
for name in all_names[21:used_names]:
	print(name)
	good_images = 0
	for imgName in glob.glob(DB_DIR+SRC_DIR+name+"*.png"):
		print(imgName)
		faces = detect(imgName)
		if faces != 0:
			good_images += 1
			rname = imgName.replace(SRC_DIR,DST_DIR + name + "/")
			print(os.path.dirname(rname))
			if not os.path.exists(os.path.dirname(rname)): 
				os.makedirs(os.path.dirname(rname))
			shutil.move(imgName,rname)
		else:
			rname = imgName.replace(SRC_DIR,BAD_DIR)
			print(rname)
			shutil.move(imgName,rname)

		print("Number of good images so far %d" % (good_images))
		if good_images == image_per_name:
			break

