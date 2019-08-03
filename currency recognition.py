from matplotlib import pyplot as plt
import cv2

max_val = 10
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()

#test_img = cv2.imread('files/test_100_1.jpg')
#test_img = cv2.imread('files/test_50_2.jpg')
#test_img = cv2.imread('files/test_20_2.jpg')
#test_img = cv2.imread('files/500.jpg')
test_img = cv2.imread('files/50.jpg')

original = test_img
cv2.imshow('original', original)

# keypoints and descriptors

(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []

	# if good then append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.8 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good),'pi')

if max_val != 10:
	print(training_set[max_pt])
	print('good matches ', max_val,'pi')

	train_img = cv2.imread(training_set[max_pt])  
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected Note: Rs. ', note)

	
	(plt.imshow(img3), plt.show())
else:
	print('No Matches')
