import pickle 
import cv2
import mediapipe as mp
import numpy as np

model_dict=pickle.load(open('model.pickle','rb'))
model=model_dict['model']
camera=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

hands=mp_hands.Hands(min_detection_confidence=0.3,static_image_mode=True)

while True:
    try:
        success,img=camera.read()
        H,W,C=img.shape
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(img_rgb)
        landmarks=[]
        x_=[]
        y_=[]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            for hand_landmarks in results.multi_hand_landmarks:            
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    landmarks.append(x-min(x_))
                    landmarks.append(y-min(y_))
        x1=int(min(x_)*W)-10
        y1=int(min(y_)*H)-10
        x2=int(max(x_)*W)-10
        y2=int(max(y_)*H)-10
        pred=model.predict([np.asarray(landmarks)])
        pred_ind=int(pred[0])
        pred_char=chr(pred_ind+65)
        cv2.putText(img,pred_char,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2,cv2.LINE_AA)
    except:
        pass
    cv2.imshow("Predicting Window",img)    
    key=cv2.waitKey(25)
    if key==ord('q'):
        break
    

cv2.destroyAllWindows()
camera.release()