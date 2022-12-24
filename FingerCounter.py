import cv2
import mediapipe as mp

video = cv2.VideoCapture(0) # Where image will be captured from

hand = mp.solutions.hands # mediapipe hand configurations
#Hand = hand.Hands(max_num_hands=1) # Number of hands used
Hand = hand.Hands() # Responsible for detecting hand in video
mpDraw = mp.solutions.drawing_utils # draw connections between points in hand
    
while True:
    success, img = video.read() # Boolean if image is being recorded or not
    img = cv2.flip(img,1) # mirror img making video normal
    #image recieved in BGR format
    # convert to RGB so it can be processed with mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB) # process with mp
    handsPoints = results.multi_hand_landmarks # extract points from hands and its coordinates
    h,w,_ = img.shape # get video img size
    pts = [] # list with points coordinates
    
    if handsPoints:
        for points in handsPoints:
            #print(points) # print points coordinates
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS) # show points on screen
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x*w), int(cord.y*h) # attribute width to x coordinate an height to y
                cv2.putText(img, str(id), (cx,cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)
                #pts.append((str(id),(cx,cy))) # append point coord and its id to pts list
                pts.append((cx,cy))
                #print(pts)

        fingers = [8, 12, 16, 20] # highest point of all fingers exept thumb 
        counter = 0
        
        if points:
            if pts[4][0] < pts[17][0]: # if left hand
                if pts[4][0] < pts[2][0]:
                    counter += 1
                    #print("RIGHT")
            if pts[4][0] > pts[17][0]: # if right hand
                if pts[4][0] > pts[2][0]:
                    counter += 1
                    #print("LEFT")    
                
            for x in fingers: # for each finger
                if pts[x][1] < pts[x-2][1]: # if highest point is lower than 3rd point
                    counter += 1
            
        cv2.putText(img,str(counter),(100,100), cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),5) # print on screen
        
    if success: # If image is being recorded
        cv2.imshow("WEBCAM", img) # Display the image
        
              
    if cv2.waitKey(1) & 0xFF==ord('q'): 
        break # If 'q' key is press stop recording
    
video.release() # Stop using the webcam
cv2.destroyAllWindows() # Close open windows   
