import numpy as np
import cv2
from pygame import mixer

def state_machine(m, index):
	yes = m > (width * height * 0.8)

	if yes and index == 0:
		A.play()
		print("A_note")
	if yes and index == 1:
		B.play()
		print("B_note")
	if yes and index == 2:
		C.play()
		print("C_note")
	if yes and index == 3:
		D.play()
		print("D_note")
	if yes and index == 4:
		E.play()
		print("E_note")
	if yes and index == 5:
		F.play()
		print("F_note")
	if yes and index == 6:
		G.play()
		print("G_note")

def ROI_analysis(roi_arr):
	max = 0
	c = 0
	for i in range(len(roi_arr)):
		hsv = cv2.cvtColor(roi_arr[i], cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, blueLower, blueUpper)
		sumation = np.sum(mask)
		if sumation >= max:
			max = sumation
			c = i

	state_machine(max, c)

def create_keys(frame1):
	# visual black & white 'piano keys'
	img = np.copy(frame1)
	img[0:height, 0:width] = (0, 0, 0)
	img[0:height, width:width * 2] = (225, 225, 225)
	img[0:height, width * 2:width * 3] = (0, 0, 0)
	img[0:height, width * 3:width * 4] = (225, 225, 225)
	img[0:height, width * 4:width * 5] = (0, 0, 0)
	img[0:height, width * 5:width * 6] = (225, 225, 225)
	img[0:height, width * 6:W] = (0, 0, 0)
	return img

mixer.init()
A = mixer.Sound('extras/audios/A_piano.wav')
B = mixer.Sound('extras/audios/B_piano.wav')
C = mixer.Sound('extras/audios/C_piano.wav')
D = mixer.Sound('extras/audios/D_piano.wav')
E = mixer.Sound('extras/audios/E_piano.wav')
F = mixer.Sound('extras/audios/F_piano.wav')
G = mixer.Sound('extras/audios/G_piano.wav')

blueLower = (80, 150, 10)
blueUpper = (120, 255, 255)

camera = cv2.VideoCapture(0)
ret, frame = camera.read()
H, W = frame.shape[:2]

height = H // 3
width = W // 7

while True:
	ret, frame = camera.read()
	frame = cv2.flip(frame, 1)
	if not(ret):
		break

	image = create_keys(frame)
	ROI_arr = []

	key1_ROI = np.copy(frame[0:height, 0:width])
	ROI_arr.append(key1_ROI)

	key2_ROI = np.copy(frame[0:height, width:width * 2])
	ROI_arr.append(key2_ROI)

	key3_ROI = np.copy(frame[0:height, width * 2:width * 3])
	ROI_arr.append(key3_ROI)

	key4_ROI = np.copy(frame[0:height, width * 3:width * 4])
	ROI_arr.append(key4_ROI)

	key5_ROI = np.copy(frame[0:height, width * 4:width * 5])
	ROI_arr.append(key5_ROI)

	key6_ROI = np.copy(frame[0:height, width * 5:width * 6])
	ROI_arr.append(key6_ROI)

	key7_ROI = np.copy(frame[0:height, width * 6:W])
	ROI_arr.append(key7_ROI)

	ROI_analysis(ROI_arr)

	cv2.imshow('Air Piano', image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
