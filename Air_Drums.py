import numpy as np
import time
import cv2
from pygame import mixer

def state_machine(sumation, sound):
	yes = sumation > Hatt_thickness[0]*Hatt_thickness[1]*0.8

	if yes and sound == 1:
		drum_clap.play()
	elif yes and sound == 2:
		drum_snare.play()

	time.sleep(0.001)

def ROI_analysis(frame, sound):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, blueLower, blueUpper)
	sumation = np.sum(mask)

	state_machine(sumation, sound)
	return mask

Verbsoe = True
mixer.init()
drum_clap = mixer.Sound('extras/audios/batterrm.wav')
drum_snare = mixer.Sound('extras/audios/button-2.ogg')

blueLower = (80, 150, 10)
blueUpper = (120, 255, 255)

camera = cv2.VideoCapture(0)
ret, frame = camera.read()
H, W = frame.shape[:2]

Hatt = cv2.resize(cv2.imread('extras/images/Hatt.png'), (100, 50), interpolation=cv2.INTER_CUBIC)
Snare = cv2.resize(cv2.imread('extras/images/Snare.png'), (100, 50), interpolation=cv2.INTER_CUBIC)

Hatt_center = [np.shape(frame)[1]*2//8, np.shape(frame)[0]*6//8] # (160, 120)
Snare_center = [np.shape(frame)[1]*6//8, np.shape(frame)[0]*6//8]
Hatt_thickness = [100, 50]

Hatt_top = [Hatt_center[0] - Hatt_thickness[0]//2, Hatt_center[1] - Hatt_thickness[1]//2]
Hatt_btm = [Hatt_center[0] + Hatt_thickness[0]//2, Hatt_center[1] + Hatt_thickness[1]//2]

Snare_thickness = [100, 50]
Snare_top = [Snare_center[0] - Snare_thickness[0]//2, Snare_center[1] - Snare_thickness[1]//2]
Snare_btm = [Snare_center[0] + Snare_thickness[0]//2, Snare_center[1] + Snare_thickness[1]//2]

time.sleep(1)
while True:
	ret, frame = camera.read()
	frame = cv2.flip(frame, 1)
	if not(ret):
		break

	snare_ROI = np.copy(frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]])
	mask1 = ROI_analysis(snare_ROI, 1)

	hatt_ROI = np.copy(frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]])
	mask2 = ROI_analysis(hatt_ROI, 2)

	if Verbsoe:
		frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]] = cv2.addWeighted(Snare, 1, frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]], 1, 0)
		frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]] = cv2.addWeighted(Hatt, 1, frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]], 1, 0)

	cv2.imshow('Air Drums', frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
