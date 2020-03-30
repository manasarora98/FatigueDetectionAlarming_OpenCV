import cv2
import winsound
from winsound import *
from playsound import playsound
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
from datetime import datetime


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#Main function
thresh = 0.25
frame_check = 20
freq=2500
duration=3000
count=0
list=[]
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    "C:\\Users\\Manas Arora\\PycharmProjects\\FatigueDetection\\venv\\util\\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame, "                DO NOT SLEEP!", (10, 30),
     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (4, 4, 98), 2)
    cv2.putText(frame, "           STAY ALERT!", (100, 440),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (4, 4, 98), 2)
    subjects = detect(gray, 0)
    cv2.putText(frame, str(datetime.now()), (415, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 2)
        if ear < thresh:
            flag += 1
            #print(flag)

            if flag >= frame_check:
                count+=1
                list.append(datetime.now())

                #cv2.putText(frame,str(DateTime.now()),(10,180),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (4, 4, 98), 2)

                winsound.Beep(freq,duration)
                cv2.imwrite('images.png', frame)
                #playsound("C:\\Users\\suvan\\Desktop\\abc.mp3")

        else:
            flag = 0
    cv2.imshow("DO NOT SLEEP!", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("\n\nNumber of times you slept ",count,"\n")
        for x in list:
            print(x,"\n")
        break
cv2.destroyAllWindows()
#cap.stop()