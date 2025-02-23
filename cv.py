import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("digits.keras")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_grid = None  #needing this for constant flickering


def preprocess_digit(cell):
    """Ensure the digit is centered and cleaned before prediction."""
    # Resize to a larger square
    cell = cv2.resize(cell, (50, 50))

    # Apply thresholding to clean up the image
    _, cell_bin = cv2.threshold(cell, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours and extract the digit
    contours, _ = cv2.findContours(cell_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cell = cell[y:y+h, x:x+w]  # Crop the digit

    # Resize cropped digit to 24x24
    cell = cv2.resize(cell, (24, 24))

    # Add padding to make it 28x28
    cell = cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # Normalize
    cell = cell.astype('float32') / 255.0
    cell = np.expand_dims(cell, axis=[0, -1])  # Shape: (1, 28, 28, 1)

    return cell


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
    new_pts[1] = pts[np.argmin(diff)] 
    new_pts[3] = pts[np.argmax(diff)] 

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
        print('hi')
        warpedGray = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
        warpedBlur = cv2.GaussianBlur(warpedGray, (5,5), 3)

        # Initial threshold to get white digits on black background
        # Convert to binary (ensure digits are white on black)
        warpedThresh = cv2.adaptiveThreshold(warpedBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Create kernels for extracting grid lines
        vertKern = np.ones((25, 1), np.uint8)
        horizKern = np.ones((1, 25), np.uint8)

        # Extract vertical and horizontal lines
        verticalLines = cv2.morphologyEx(warpedThresh, cv2.MORPH_OPEN, vertKern, iterations=2)
        horizontalLines = cv2.morphologyEx(warpedThresh, cv2.MORPH_OPEN, horizKern, iterations=2)

        # Combine grid lines
        gridLines = cv2.bitwise_or(verticalLines, horizontalLines)

        # Remove grid lines from the thresholded image (digitsOnly is now properly defined)
        digitsOnly = cv2.bitwise_and(warpedThresh, cv2.bitwise_not(gridLines))

        # Further clean the digits by removing any remaining noise
        kernel = np.ones((3, 3), np.uint8)

        # Enhance the digits by dilating them slightly
        digitsOnly = cv2.dilate(digitsOnly, kernel, iterations=1)

        # Remove any noise that might still remain
        digitsOnly = cv2.erode(digitsOnly, kernel, iterations=1)

        # Apply a stronger threshold to remove any remaining weak pixels
        _, digitsOnly = cv2.threshold(digitsOnly, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # Ensure digits are **white on black**
        digitsOnly = cv2.bitwise_not(digitsOnly)

        # Show final processed image
        cv2.imshow("Masked Image", digitsOnly)


        grid_size = digitsOnly.shape[0]
        cell_size = grid_size // 9

        sudoku_cells = []

        for row in range(9):
            for col in range(9):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                cell = digitsOnly[y1:y2, x1:x2]
                cell = cv2.resize(cell, (28, 28))
                #cell = cv2.bitwise_not(cell)
                cell = cell[5:25, 5:25]
                sudoku_cells.append(cell)

        fig, axes = plt.subplots(9, 9, figsize=(10, 10))

        for i, cell in enumerate(sudoku_cells):
            row, col = divmod(i, 9)  # Convert index to row, column
            axes[row, col].imshow(cell, cmap='gray')  # Display cell
            axes[row, col].axis('off')  # Hide axes for better visualization

        plt.show()
        digits = []
        for i, test_cell in enumerate(sudoku_cells):
            if np.sum(test_cell) < 50:
                digits.append(0)
                continue
            
            test_cell = cv2.resize(test_cell, (28, 28))  # Resize for model input
            test_cell = test_cell.astype('float32') / 255.0  
            test_cell = np.expand_dims(test_cell, axis=[0, -1])


            # Visualize each digit before prediction
            if i % 5 == 0:
                plt.imshow(test_cell.squeeze(), cmap="gray")
                plt.title(f"Test Cell {i}")
                plt.axis("off")
                plt.show()

            prediction = model.predict(test_cell)
            digit = np.argmax(prediction)

            digits.append(digit)

        print(f"Final Predictions: {digits}")



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()