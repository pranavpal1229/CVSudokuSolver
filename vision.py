import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model("digits_model.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(1)
    diff = np.diff(points, axis=1)
    
    new_points[0] = points[np.argmin(add)]       # Top-left
    new_points[2] = points[np.argmax(add)]       # Bottom-right
    new_points[1] = points[np.argmin(diff)]      # Top-right
    new_points[3] = points[np.argmax(diff)]      # Bottom-left
    
    return new_points

def split_boxes(img):
    rows = np.vsplit(img, 9)
    
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

def get_prediction(boxes):
    results = []
    for img in boxes:
        total_pixels = cv2.countNonZero(img)
        if total_pixels < 50:  # tweak this threshold based on your lighting
            results.append(0)
            continue
        
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)
        pred = model.predict(img, verbose=0)
        class_index = np.argmax(pred)
        prob = np.max(pred)
        
        if prob > 0.8:
            results.append(class_index)
        else:
            results.append(0)
    return results

# Flag to track if we've saved the boxes
boxes_saved = False

while True:
    ret, img = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', img)
    
    #first we want to convert to black and white.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grey scale
    imgThresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #actually makes it black and white
    cv2.imshow("Black and White Image", imgThresh)
    
    #now my goal is to simply get the grid and nothing else
    #imgCanny = cv2.Canny(imgThresh, 50, 150)
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgCont = img.copy()
    
    maxArea = 0
    biggestContour = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True) #calculating perimeter and making sure it is closed
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > maxArea:
                maxArea = area
                biggestContour = approx
    
    if biggestContour is not None:
        cv2.drawContours(imgCont, [biggestContour], -1, (0,255,0), 2)
        #I now want to get a bird's eye view of this grid
        orderedPoints = reorder(biggestContour)
        target_corners = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(orderedPoints, target_corners)
        warped = cv2.warpPerspective(img, transform, (450, 450))
        
        if warped is not None:
            #first...convert back to black and white
            bwImg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((3,3), np.uint8)
            bwThresh = cv2.adaptiveThreshold(bwImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            # Fix: You had a line using bwThresh before it was defined
            bwThresh = cv2.morphologyEx(bwThresh, cv2.MORPH_OPEN, kernel)
            
            cv2.imshow("imgCont", bwThresh)
            
            boxes = split_boxes(bwThresh)
            predictions = get_prediction(boxes)
            print(predictions)
            cv2.imshow("One Box", boxes[0])
            
            # Save the first 5 boxes if they haven't been saved yet
            if not boxes_saved and len(boxes) >= 5:
                for i in range(5):
                    # Create a directory if it doesn't exist
                    import os
                    if not os.path.exists('saved_boxes'):
                        os.makedirs('saved_boxes')
                    
                    # Save the box
                    cv2.imwrite(f'saved_boxes/box_{i}.png', boxes[i])
                
                print("Saved the first 5 boxes to the 'saved_boxes' directory!")
                boxes_saved = True
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()