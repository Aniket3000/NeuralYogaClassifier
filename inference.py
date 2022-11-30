import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 


def inFrame(landmarks):
	if landmarks[28].visibility > 0.6 and landmarks[27].visibility > 0.6 and landmarks[15].visibility>0.6 and landmarks[16].visibility>0.6:
		return True 
	return False

# Loading the trained model from pervious file
model = load_model("model.h5")
label = np.load("labels.npy")



holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


while True:
	lst = []

	ret, frm = cap.read()
	window = np.zeros((480,640,3), dtype="uint8")
	frm = cv2.flip(frm, 1)
	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
	frm = cv2.blur(frm, (4,4))
	if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
		for i in res.pose_landmarks.landmark:
			lst.append(i.x - res.pose_landmarks.landmark[0].x)
			lst.append(i.y - res.pose_landmarks.landmark[0].y)

		lst = np.array(lst).reshape(1,-1)

		p = model.predict(lst)
		pred = label[np.argmax(p)]

		if p[0][np.argmax(p)] > 0.75:
			cv2.putText(frm, pred , (100,450),cv2.FONT_ITALIC, 0.8, (0,255,0),2)

		else:
			cv2.putText(frm, "Asana is either wrong or not trained properly" , (100,180),cv2.FONT_ITALIC, 1.8, (0,0,255),3)

	else: 
		cv2.putText(frm, "Make sure full body is visible!!", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

		
	drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
							connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=1),
							 landmark_drawing_spec=drawing.DrawingSpec(color=(255,0,0), circle_radius=4, thickness=2))


	window[0:480, 0:640, :] = cv2.resize(frm, (640, 480))

	cv2.imshow("window", window)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break

