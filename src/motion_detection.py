import numpy as np
import cv2

CONT_AR = 10  # TODO change sensitivity
DEBUG = False  # TODO debug mode
USE_CUSTOM_AREA = True

cap = cv2.VideoCapture("test.avi")  # TODO different video sources


ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# select area of interest
roi = cv2.selectROI("Select area or interest", frame1)
cv2.destroyWindow("Select area or interest")
black = np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
black1 = cv2.rectangle(black,(roi[0], roi[1]),(roi[0]+roi[2], roi[1]+roi[3]),(255, 255, 255), -1)
gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
ret,b_mask = cv2.threshold(gray,127,255, 0)

while cap.isOpened():
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    frame_diff = cv2.absdiff(blur1, blur2)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    if USE_CUSTOM_AREA: thresh = cv2.bitwise_and(thresh, thresh, mask=b_mask)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < CONT_AR:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Motion detector", frame2)  # TODO text on screen, for instance "Motion detected"
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

    blur1 = blur2

cv2.destroyAllWindows()
cap.release()