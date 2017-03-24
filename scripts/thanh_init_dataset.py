import numpy as np
import cv2
import glob
import shutil
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
	cv2.equalizeHist(gray, gray) # helps improve the detection

	# return a list of detection boxes
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	return len(faces)

# =================================== MAIN ================================== #
src = "/Users/thanhvu/Desktop/FaceSimilarity/faces/"
# out = open("bad_imgs.txt","w+")

# all names from Names100Dataset. Each have 800 images
# names = ['Aaron', 'Abby', 'Amanda', 'Andrea', 'Angela', 'Ann', 'Anna', 'Annie', 'Anthony', 'Barbara', 'Brandon', 'Brian', 'Chris', 'Christina', 'Christine', 'Cindy', 'Danielle', 'Danny', 'David', 'Dylan', 'Elena', 'Elizabeth', 'Emily', 'Emma', 'Eric', 'Erin', 'Eva', 'Evan', 'Gabriel', 'Gary', 'Grace', 'Greg', 'Helen', 'Ian', 'Ivan', 'Jackie', 'Jacob', 'Jake', 'James', 'Jamie', 'Jane', 'Jason', 'Jeff', 'Jesse', 'Jim', 'Jo', 'Joel', 'Joey', 'Jon', 'Jonathan', 'Joseph', 'Julie', 'Kate', 'Katie', 'Kelly', 'Kevin', 'Kim', 'Kyle', 'Lauren', 'Leah', 'Linda', 'Lisa', 'Lucas', 'Maggie', 'Marc', 'Matthew', 'Melissa', 'Michelle', 'Mike', 'Monica', 'Nancy', 'Natalia', 'Nathan', 'Nick', 'Nicolas', 'Noah', 'Oliver', 'Olivia', 'Patricia', 'Patrick', 'Paula', 'Rachel', 'Rebecca', 'Rick', 'Samantha', 'Sarah', 'Sergio', 'Sofia', 'Stephanie', 'Stephen', 'Steve', 'Steven', 'Sue', 'Thomas', 'Tina', 'Tony', 'Tyler', 'Vanessa', 'Victor', 'Zoe']

# imgNames = []
# for name in names:
# 	for j in range(1,801):
# 		imgName = "../../Names100Dataset/Names100_Images/"+name+"_"+ str(j)+".png"
# 		imgNames.append(imgName)
# print(len(imgNames))

# ======== Test individual images ======== #
# imgName = "../test/face_sim_test.jpg"
# print(imgName)
# detectAndDisplay(imgName)

# ======== Detect and move images ======== #
good = 0
# for i,imgName in enumerate(imgNames):
for i,imgName in enumerate(sorted(glob.glob("../../Names100Dataset/Names100_Images/*"))):
	print (imgName)
	faces = detect(imgName)
	if faces != 0:
		good+= 1
		shutil.move(imgName,imgName.replace("Names100_Images","good"))
	else:
		shutil.move(imgName,imgName.replace("Names100_Images","bad"))

	print("Current index: "+str(i))
	print("Number of good images: "+str(good))
	if good == 100:
		break
print("total good images: %d" % (good))

