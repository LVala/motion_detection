from copyreg import constructor
import cv2

CONT_AR = 1000  # TODO change sensitivity
DEBUG = True  # TODO debug mode

cap = cv2.VideoCapture("test.avi")  # TODO different video sources


ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while cap.isOpened():
    # TODO select area of image
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    frame_diff = cv2.absdiff(blur1, blur2)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
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