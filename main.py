import cv2
from sklearn.feature_selection import VarianceThreshold
from tracker import *

#create tracker object
tracker = EuclideanDistTracker()

#all bounding boxes are stored into one array

cap = cv2.VideoCapture("Highway.mp4")

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history =100, varThreshold = 50)


#capturing each frames one after another
while True:
    ret, frame = cap.read()

    #extract region of interest
    height, width, _ = frame.shape
    print(height, width)
    roi = frame[100:720, 140:1200]


    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []

    for cnt in contours:
        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area>100:
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)

            detections.append([x,y,w,h])

#Object tracking
    boxes_ids = tracker.update(detections) 

    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)

    print(detections)        
    cv2.imshow("ROI", roi)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)

    key = cv2.waitKey(50)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()