import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

try:
    model = load_model("digits.keras")
except Exception:
    model = load_model("digits_model.h5")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_grid = None  #needing this for constant flickering
frame_count = 0
UPDATE_EVERY = 5  # run heavy pipeline every N frames
last_overlay = None
last_warped = None
MIN_CONFIDENCE = 0.8
REQUIRED_STREAK = 2
stable_digits = [0] * 81
stable_confidences = [0.0] * 81
candidate_digits = [-1] * 81
candidate_confidences = [0.0] * 81
candidate_streaks = [0] * 81
empty_streaks = [0] * 81


def preprocess_digit(cell):
    """Center crop and normalize a single cell assumed to be white digit on black background.
    Returns a 28x28 float32 array in [0,1] (no batch dims) for fast batching.
    """
    cell = cv2.resize(cell, (50, 50))

    # Ensure clean binary (keep white digits on black)
    _, cell_bin = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background appears white, invert to make digit white on black
    if np.mean(cell_bin) > 127:
        cell_bin = cv2.bitwise_not(cell_bin)

    contours, _ = cv2.findContours(cell_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = cell_bin[y:y+h, x:x+w]
    else:
        digit = cell_bin

    h, w = digit.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = digit

    cell = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    cell = cell.astype('float32') / 255.0
    return cell


def prepare_batch(imgs28):
    """Stack a list of 28x28 images into the model's expected input shape."""
    if not imgs28:
        return None
    input_shape = getattr(model, 'input_shape', None)
    arr = np.stack(imgs28, axis=0)
    if input_shape is not None and len(input_shape) == 4:
        # Expecting (None, 28, 28, 1)
        return np.expand_dims(arr, axis=-1)
    # Default: (None, 28, 28)
    return arr


def to_probabilities(logits_or_probs):
    """Convert model outputs to probabilities if needed (softmax)."""
    arr = np.array(logits_or_probs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sums = np.sum(arr, axis=1, keepdims=True)
    if np.all(arr >= 0) and np.all(arr <= 1) and np.all(np.abs(row_sums - 1.0) < 1e-3):
        return arr
    arr = arr - np.max(arr, axis=1, keepdims=True)
    expx = np.exp(arr)
    return expx / np.sum(expx, axis=1, keepdims=True)


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
    target_corners = np.array([[0, 0], [540, 0], [540, 540], [0, 540]], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(ordered_corners, target_corners)
    warped = cv2.warpPerspective(img, transform, (540, 540))

    return warped

while True:
    frame_count += 1
    ret, img = cap.read()
    if not ret:
        break
    
    img_copy = img.copy()
    update_now = (frame_count % UPDATE_EVERY == 0) or (last_grid is None)
    grid_corners = None
    if update_now:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5,5), 3)
        imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        imgCanny = cv2.Canny(imgMorph, 50, 150)
        grid_corners = getContours(imgCanny, img_copy)
    else:
        grid_corners = last_grid
        if last_grid is not None:
            cv2.drawContours(img_copy, [last_grid], -1, (0, 255, 0), 3)

    warped_grid = None
    if grid_corners is not None:
        warped_grid = warped_perspective(img, grid_corners)

    cv2.imshow('Webcam', img_copy)

    if last_warped is not None:
        cv2.imshow('Bird Eye View', last_warped)
    
    if warped_grid is not None and update_now:
        warpedGray = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
        warpedBlur = cv2.GaussianBlur(warpedGray, (5,5), 3)

        warpedThresh = cv2.adaptiveThreshold(warpedBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        vertKern = np.ones((25, 1), np.uint8)
        horizKern = np.ones((1, 25), np.uint8)
        verticalLines = cv2.morphologyEx(warpedThresh, cv2.MORPH_OPEN, vertKern, iterations=1)
        horizontalLines = cv2.morphologyEx(warpedThresh, cv2.MORPH_OPEN, horizKern, iterations=1)

        # Combine grid lines
        gridLines = cv2.bitwise_or(verticalLines, horizontalLines)

        # Remove grid lines from the thresholded image
        digitsOnly = cv2.bitwise_and(warpedThresh, cv2.bitwise_not(gridLines))

        # Further clean the digits by removing any remaining noise
        kernel = np.ones((3, 3), np.uint8)

        # Enhance the digits by dilating them slightly
        digitsOnly = cv2.dilate(digitsOnly, kernel, iterations=1)

        # Remove any noise that might still remain
        digitsOnly = cv2.erode(digitsOnly, kernel, iterations=0)
        _, digitsOnly = cv2.threshold(digitsOnly, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("Masked Image", digitsOnly)

        grid_size = digitsOnly.shape[0]
        cell_size = grid_size // 9

        sudoku_cells = []
        is_empty_cell = []
        for row in range(9):
            for col in range(9):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                cell = digitsOnly[y1:y2, x1:x2]
                cell = cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                cell = cv2.resize(cell, (28, 28))
                sudoku_cells.append(cell)
                is_empty_cell.append(cv2.countNonZero(cell) < 0.02 * cell.size)

        digits = [0] * 81
        batch_imgs = []
        batch_indices = []
        for i, raw_cell in enumerate(sudoku_cells):
            if is_empty_cell[i]:
                continue
            img28 = preprocess_digit(raw_cell)
            batch_imgs.append(img28)
            batch_indices.append(i)

        if batch_imgs:
            batch_input = prepare_batch(batch_imgs)
            raw_preds = model.predict(batch_input, verbose=0)
            probs = to_probabilities(raw_preds)
            pred_digits = np.argmax(probs, axis=1)
            pred_conf = probs[np.arange(len(pred_digits)), pred_digits]
            conf_by_index = {idx: float(c) for idx, c in zip(batch_indices, pred_conf)}
            for idx, d in zip(batch_indices, pred_digits):
                digits[idx] = int(d)

        # Temporal smoothing and confidence gating
        for i in range(81):
            if is_empty_cell[i]:
                empty_streaks[i] += 1
                if empty_streaks[i] >= 2:
                    stable_digits[i] = 0
                    stable_confidences[i] = 1.0
                    candidate_digits[i] = -1
                    candidate_confidences[i] = 0.0
                    candidate_streaks[i] = 0
                continue
            else:
                empty_streaks[i] = 0

            new_digit = digits[i]
            new_conf = 0.0
            if batch_imgs:
                new_conf = conf_by_index.get(i, 0.0)

            if candidate_digits[i] == new_digit:
                candidate_streaks[i] += 1
                if new_conf > candidate_confidences[i]:
                    candidate_confidences[i] = new_conf
            else:
                candidate_digits[i] = new_digit
                candidate_confidences[i] = new_conf
                candidate_streaks[i] = 1

            if candidate_streaks[i] >= REQUIRED_STREAK and candidate_confidences[i] >= MIN_CONFIDENCE:
                stable_digits[i] = candidate_digits[i]
                stable_confidences[i] = candidate_confidences[i]

        # --- OVERLAY PREDICTED DIGITS ON WARPED GRID ---
        overlay_grid = warped_grid.copy()
        for i, digit in enumerate(stable_digits):
            if digit == 0:
                continue
            row, col = divmod(i, 9)
            x = col * cell_size + cell_size // 4
            y = row * cell_size + 3 * cell_size // 4
            cv2.putText(overlay_grid, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        last_overlay = overlay_grid
        last_warped = warped_grid

    if last_overlay is not None:
        cv2.imshow("Overlay", last_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
