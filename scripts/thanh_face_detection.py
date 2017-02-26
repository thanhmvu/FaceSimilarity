import numpy as np
import cv2
# from matplotlib import pyplot as plt

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

	faces = face_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
	if len(faces) == 0:
		print(imgName)

def detectAndReturn(imgName):
	face_cascade = cv2.CascadeClassifier('openCVDetector/haarcascade_frontalface_default.xml')

	img = cv2.imread(imgName)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(len(faces))
	return len(faces)

# =================================== MAIN ================================== #
src = "/Users/thanhvu/Desktop/FaceSimilarity/faces/"
# out = open("bad_imgs.txt","w+")

# cnt_bad = 0
# for i in range(26):	
# 	imgName = src + str(i).zfill(6) + ".png"
# 	faces = detectAndReturn(imgName)
# 	print(imgName)
# 	if faces == 0:
# 		out.write(imgName+"\n")
# 		cnt_bad += 1
# print("cnt bad: %d" % (cnt_bad))

# for i in range(0,1):	
	# imgName = src + str(i).zfill(6) + ".png"
	# detectAndDisplay2(imgName)
	# print(imgName)
	# if faces == 0:
		# out.write(imgName+"\n")
# 		cnt_bad += 1
# print("cnt bad: %d" % (cnt_bad))

imgName = "../test/face_sim_test.jpg"
print(imgName)
detectAndDisplay(imgName)
