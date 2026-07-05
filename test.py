import cv2 
import mediapipe as mp
import time

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Specifies running in single frame mode
    num_hands=2
)

cam = cv2.VideoCapture(2)#Change accordingly try 1 or 2 if 0 didnt work(or howevermany camera soruces you have)
cam.set(3, 1280)#setting width
cam.set(4, 720)#setting height

bananas = []
is_drawing = False
i = 0
#For i from 1 to length(your_list):
 #   pt1 = your_list[i-1]
  #  pt2 = your_list[i]
   # draw_line(pt1, pt2)
with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened(): #Making sure the camera is open
        success, frame = cam.read()
        if not success: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        if result.hand_world_landmarks:#makes sure our code only works when we actually get the data
            
                jointA = result.hand_world_landmarks[0][8] #Getting the point 08 which is the index tip
                jointB = result.hand_world_landmarks[0][4] #Getting the point 04 which is the thumb tip
                x1,y1,z1 = jointA.x,jointA.y,jointA.z#assigning the values we get to shorter variables
                x2,y2,z2 = jointB.x,jointB.y,jointB.z#assigning the values we get to shorter variables
                print(abs(x1-x2),abs(y1-y2))

                jAnormal = result.hand_landmarks[0][8]
                jBnormal = result.hand_landmarks[0][4]
                
                h,w,_=frame.shape
                bx = int(jBnormal.x*w)
                by = int(jBnormal.y*h)
                cx = int(jAnormal.x*w)
                cy = int(jAnormal.y*h)
                dx = int((bx + cx)/2)
                dy = int((by + cy)/2)
                   
                if abs(x1 - x2) < 0.025 and abs(y1 - y2) < 0.018:
                    if not is_drawing:
                        bananas.append([])# adding a sublist(a list in a list)
                        is_drawing = True
                    bananas[-1].append((dx, dy))
                else:
                    is_drawing = False
                    cv2.line((frame),pt1=(bx,by),pt2=(cx,cy),color=(0,255,0),thickness=10)
                      
        for banana in bananas:
            for i in range(1, len(banana)):
                cv2.line((frame),pt1=(banana[i-1]),pt2=(banana[i]),color=(0,255,0),thickness=10)
                      
        if cv2.waitKey(1) & 0xFF == ord('c'):
            bananas.clear()
            print(bananas)
     
        cv2.imshow("Show Video", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv2.destroyAllWindows()