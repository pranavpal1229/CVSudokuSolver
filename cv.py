import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        if area > 10000:
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

def reorder_points(pts):
    pts = pts.reshape((4,2))
    new_pts = np.zeros((4,2), dtype = np.float32)

    s = pts.sum(axis = 1)
    new_pts[0] = pts[np.argmin(s)]
    new_pts[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    new_pts[1] = pts[np.argmin(diff)]  # Fixed from overwriting `new_pts`
    new_pts[3] = pts[np.argmax(diff)]  # Fixed from overwriting `new_pts`

    return new_pts

def warped_perspective(img, corners):
    if corners is None:
        return None
    ordered_corners = reorder_points(corners)
    target_corners = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(ordered_corners, target_corners)
    warped = cv2.warpPerspective(img, transform, (450, 450))

    return warped

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

    warped_grid = None
    if grid_corners is not None:
        warped_grid = warped_perspective(img, grid_corners)

    cv2.imshow('Webcam', img_copy)

    if warped_grid is not None:
        cv2.imshow('Bird Eye View', warped_grid)
    
    if warped_grid is not None:
        warpedGray = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
        warpedBlur = cv2.GaussianBlur(warpedGray, (5,5), 3)

        warpedThresh = cv2.adaptiveThreshold(warpedBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        warpedThresh = cv2.bitwise_not(warpedThresh)
        #cv2.imshow("Warp Canny", warpedThresh)
        vertKern = np.ones((25,1), np.uint8)
        horizonKern = np.ones((1,10), np.uint8)
        # Extract vertical lines
        verticalLines = cv2.erode(warpedThresh, vertKern, iterations=2)
        verticalLines = cv2.dilate(verticalLines, vertKern, iterations=2)
        horizontalLines = cv2.erode(warpedThresh, horizonKern, iterations=2)
        horizontalLines = cv2.dilate(horizontalLines, horizonKern, iterations=2)
        gridOverlay = cv2.bitwise_or(verticalLines, horizontalLines)
        lines = cv2.HoughLinesP(gridOverlay, 1, np.pi/180, threshold=150, minLineLength=40, maxLineGap=5)

        # Create a blank white image for the structured grid
        structured_grid = np.ones_like(gridOverlay) * 255  # White background

        # Draw detected lines in black
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(structured_grid, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Draw black lines

        structured_grid_dilated = cv2.dilate(structured_grid, np.ones((3,3), np.uint8), iterations=2)
        invertedGrid = cv2.bitwise_not(structured_grid_dilated)
        cleanDigits = cv2.bitwise_and(warpedThresh, cv2.bitwise_not(invertedGrid))

        cv2.imshow("Masked Image", cleanDigits)
        grid_size = cleanDigits.shape[0]
        cell_size = grid_size // 9

        sudoku_cells = []

        for row in range(9):
            for col in range(9):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                cell = cleanDigits[y1:y2, x1:x2]
                cell = cv2.resize(cell, (28, 28))
                sudoku_cells.append(cell)


        fig, axes = plt.subplots(9, 9, figsize=(10, 10))

        for i, cell in enumerate(sudoku_cells):
            row, col = divmod(i, 9)  # Convert index to row, column
            axes[row, col].imshow(cell, cmap='gray')  # Display cell
            axes[row, col].axis('off')  # Hide axes for better visualization

        plt.show()




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
