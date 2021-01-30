import numpy as np
import cv2
#from background_segment import segment

icons = cv2.imread("extras/images/all.jpeg")
icons = cv2.resize(icons, (640, 480))

blueLower = (80, 150, 10)
blueUpper = (120, 255, 255)

def state_machine(m, index):
    yes = m > (100 * 100 * 0.8)
    if yes:
        if index == 0:
            img = "piano.jpeg"
        if index == 1:
            img = "guitar.jpeg"
        if index == 2:
            img = "drum.jpeg"
        if index == 3:
            img = "mic.jpeg"
        
        cap.release()
        cv2.destroyAllWindows()
        #segment(img)

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
    
    print(c)
    state_machine(max, c)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    image = np.copy(frame)
    image[icons != 255] = 0
    ROI_arr = []

    key1_ROI = np.copy(frame[0:100, 0:100])
    ROI_arr.append(key1_ROI)

    key2_ROI = np.copy(frame[0:100, 100:200])
    ROI_arr.append(key2_ROI)

    key3_ROI = np.copy(frame[0:100, 200:300])
    ROI_arr.append(key3_ROI)

    key4_ROI = np.copy(frame[0:100, 300:400])
    ROI_arr.append(key4_ROI)

    ROI_analysis(ROI_arr)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
