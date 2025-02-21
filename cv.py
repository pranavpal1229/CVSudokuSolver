import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_grid = None  #needing this for constant flickering



def getContours(img, original_img):
    global last_grid

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) ##just trying to get outer points

    largest_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15000:
            perimeter = cv2.arcLength(cnt, True) #num sides
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area #updating area
            
            if largest_contour is not None:
                last_grid = largest_contour
            else:
                last_grid = None
            
            if last_grid is not None:
                cv2.drawContours(original_img, [last_grid], -1, (0, 255, 0), 3)
                return last_grid
            return None
    



