import cv2
import winsound

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


thresh = 0.25
frame_check = 20
freq = 2500
duration = 3000
count = 0
list = []
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    "C:\\Users\\suvan\\Downloads\\shape_predictor_face_landmarks.dat")

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
            # print(flag)

            if flag >= frame_check:
                count += 1
                list.append(datetime.now())

                # cv2.putText(frame,str(DateTime.now()),(10,180),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (4, 4, 98), 2)

                winsound.Beep(freq, duration)
                cv2.imwrite('images.png', frame)
                # playsound("C:\\Users\\suvan\\Desktop\\abc.mp3")

        else:
            flag = 0
    cv2.imshow("DO NOT SLEEP!", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("\n\nNumber of times you slept ", count, "\n")
        for x in list:
            print(x, "\n")
        break
cv2.destroyAllWindows()
# cap.stop()


from tkinter import *
import os
from datetime import datetime;

root = Tk()

root.configure(background="white")


def function2():
    os.system("py Mytrainer.py")


def function3():
    os.system("py Mytester.py")


def function5():
    os.startfile("Myattendance.txt")


def function6():
    root.destroy()


def attend():
    os.startfile("Myattendance.txt")


# stting title for the window
root.title("Attendance Management Using Face Recognition")

# creating a text label
Label(root, text="Attendance Management System", font=("times new roman", 20), fg="white", bg="#E80816", height=2).grid(
    row=0, rowspan=2, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)

# creating second button
Button(root, text="Train Dataset", font=("times new roman", 20), bg="#0827E8", fg='white', command=function2).grid(
    row=4, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)

# creating third button
Button(root, text="Recognize", font=('times new roman', 20), bg="#0827E8", fg="white", command=function3).grid(row=5,
                                                                                                               columnspan=2,
                                                                                                               sticky=N + E + W + S,
                                                                                                               padx=5,
                                                                                                               pady=5)

# creating attendance button
Button(root, text="Attendance List", font=('times new roman', 20), bg="#0827E8", fg="white", command=attend).grid(row=6,
                                                                                                                  columnspan=2,
                                                                                                                  sticky=N + E + W + S,
                                                                                                                  padx=5,
                                                                                                                  pady=5)

# Button(root,text="Developers",font=('times new roman',20),bg="#0D47A1",fg="white",command=function5).grid(row=8,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root, text="Exit", font=('times new roman', 20), bg="#E80816", fg="white", command=function6).grid(row=9,
                                                                                                          columnspan=2,
                                                                                                          sticky=N + E + W + S,
                                                                                                          padx=5,
                                                                                                          pady=5)

root.mainloop()
