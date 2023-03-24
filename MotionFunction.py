import cv2
import time
import mediapipe as mp
import cvzone 
from time import sleep
from pynput.keyboard import Controller
import math 
from numpy import ndarray

portion =0 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
 
left_s =  ndarray((2,),int)
right_s = ndarray((2,),int)
left_h =  ndarray((2,),int)
right_h = ndarray((2,),int)

def CoOrdinate(id, x,y):

    if(id==12):
     right_s[0]=x
     right_s[1]=y
    if(id==11):
     left_s[0]=x
     left_s[1]=y
    if(id==16):
     right_h[0]=x
     right_h[1]=y
    if(id==15):
     left_h[0]=x
     left_h[1]=y

mp_pose=mp.solutions.pose
pose=mp_pose.Pose()

def dist(soulder,hand):
  x1, y1= soulder
  x2, y2= hand 
  
#    Modified Euclidean distance equation
  d = (abs(x1-x2)*abs(x1-x2)) - (abs(y1-y2)*abs(abs(y1-y2)))  
  #d= math.sqrt(d)
  return d 
 
 

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Stream on 

  while True:
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
       
      continue
 
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    result=pose.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    

    for id,lm in enumerate (result.pose_landmarks.landmark):
         x=int(lm.x*640)
         y=int(lm.y*480)
         CoOrdinate(id, x, y) 

    cnt =0
    #  Hand Landmark detection
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Get hand index to check label (left or right)
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        # Set variable to keep landmarks positions (x and y)
        handLandmarks = []

        # Fill list with x and y positions of each landmark
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])
      if(handLabel=="Right"):
        cnt+=1
      if(handLabel=="Left"):
        cnt+=1
#      
        # Moving Both hand
      if(dist(right_s,right_h)> dist(left_s, right_h) and dist(right_s,left_h)< dist(left_s, left_h)):
        cv2.putText(image, "Moving Both hand", (60, 310),cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)
        

        #   Detect Left hand and Left move
      elif( handLabel=='Right' and dist(right_s,left_h) > dist(left_s, left_h) and dist(right_s,right_h)> dist(left_s, right_h)):
        cv2.putText(image, "Detect Left hand & Moving Left", (60, 510),cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)
        

        #  Detect Right hand and Right move 

      elif( handLabel=='Left' and dist(right_s,right_h)< dist(left_s, right_h) and  dist(right_s,left_h)< dist(left_s, left_h)):
        cv2.putText(image, "Detect Right hand & Moving RIght", (60, 510),cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)
         
      
      #   Left Move
      elif(handLabel!="Right" and dist(right_s,left_h) > dist(left_s, left_h) and dist(right_s,right_h) > dist(left_s, right_h)):
        cv2.putText(image, "Moving Left", (60, 310),cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)
     

        #  Right move
      elif(handLabel!="Left" and dist(right_s,left_h) < dist(left_s, left_h) and dist(right_s,right_h) < dist(left_s, right_h)):
        cv2.putText(image, "Moving Right", (60, 310),cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)
      
      #Detect Left Hand
          
      elif(handLabel == "Right" and dist(right_s,left_h)> dist(left_s, left_h) and dist(right_s,right_h)< dist(left_s, right_h)):
        cv2.putText(image, "Detect Left hand", (60, 310),cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)
         
      #    Detect Right Hand
      elif( handLabel == "Left" and dist(right_s,right_h)< dist(left_s, right_h)and dist(right_s,left_h)> dist(left_s, left_h)):
        cv2.putText(image, "Detect Right hand", (60, 310),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)    
        
         

         #  Move Up 
      if( right_s[1]- right_h[1] >200):
             
             cv2.putText(image, "Move Up", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3) 
         
    # Display image
    cv2.imshow('Motion Function', image)
      
    flag=False
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()