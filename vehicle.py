import cv2
import numpy as np 

#WEB camera
vid = cv2.VideoCapture("video.mp4") 
count_line_position = 550
#Initialize subtractor
algo = cv2.createBackgroundSubtractorMOG2()
min_width_rect = 80
min_height_rect = 80

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6 # ALLOWABBLE ERROR BETWEEN PIXEL
counter = 0

while(True):  
    ret, frame = vid.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #operations based on morphology of images
    dilat2 = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilat2 = cv2.morphologyEx(dilat2,cv2.MORPH_CLOSE,kernel)

    #counting of vehicles
    counterSahpe,h = cv2.findContours(dilat2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #find contours in binary images
    # to show how algorithm works in black and white
    # cv2.imshow('Detector1', dilat2)

    #to draw counting line in cv2
    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,127,0),3) 
    
    for (i,c) in enumerate(counterSahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4,(0,255,255),-1)

        for(x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
            cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,120,0),3) 
            detect.remove((x,y))
            print("Vehicle Counter:" +str(counter))
        # Display the resulting frame 
        cv2.imshow('Video Original', frame) 
    cv2.putText(frame,"Vehicle Counter :"+str(counter),(x,y-29),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),5)

    if cv2.waitKey(1) == 13: 
        break
  
# After the loop release the cap object 

# Destroy all the windows 
cv2.destroyAllWindows() 
vid.release() 