# This file is for the asanas data capturing and labelling of asanas
# and it generates a numpy file of array of arrays on which training of the DNN model is done
# So the output of this file acts as the input layer in the DNN model formed with keras

import cv2 
import mediapipe as mp 
import numpy as np

# This function checks if the landmarks are in the current frame or not
def checkInFrame(landmarks):
	if landmarks[27].visibility > 0.6 and landmarks[28].visibility > 0.6 and landmarks[15].visibility>0.6 and landmarks[16].visibility>0.6:
		return True 
	return False
 
cap = cv2.VideoCapture(0)

# Taking the name of asana
name = input("Enter the name of the Asana : ")

# Going in the pose solutions of the mediapipe library
holistic = mp.solutions.pose
holis = holistic.Pose()

# It helps in drawing on the frame
drawing = mp.solutions.drawing_utils

X_in = []
frame_cnt = 0

while True:
	landmarks = []
	# Going through each frame
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	# mediapipe process only works with RGB images so converting it in the process
	resultant = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

	# Storing the relative position of the landmarks with respect
	# to landmark[0] i.e., the "nose" landmark if it satisfies checkInFrame
	if resultant.pose_landmarks and checkInFrame(resultant.pose_landmarks.landmark):
		for i in resultant.pose_landmarks.landmark:
			landmarks.append(i.x - resultant.pose_landmarks.landmark[0].x)
			landmarks.append(i.y - resultant.pose_landmarks.landmark[0].y)
			break

		# Appending landmark list to X_input
		X_in.append(landmarks)
		frame_cnt = frame_cnt+1

	else:
		cv2.putText(frame, "Make sure full body is visible!!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

	# We draw the landmarks and their connections with each other on the frame
	drawing.draw_landmarks(frame, resultant.pose_landmarks, holistic.POSE_CONNECTIONS)
	# Printing number on the image
	cv2.putText(frame, str(frame_cnt), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	cv2.imshow("window", frame)

	if cv2.waitKey(1) == 27 or frame_cnt>100:
		cv2.destroyAllWindows()
		cap.release()
		break

# Saving in the form of numpy file so as to use it in the next file
np.save(f"{name}.npy", np.array(X_in))
# print(np.array(X_in).shape)
