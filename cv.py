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
    

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 3) #smooting out the values

    imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel, iterations=2) #not really sure what it does but GPT recommended it and it's working

    imgCanny = cv2.Canny(imgMorph, 50, 150) #detects edges with lower and upper bound...might need higher upper bound

    img_copy = img.copy()
    grid_corners = getContours(imgCanny, img_copy)


