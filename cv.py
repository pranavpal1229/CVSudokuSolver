import cv2 #importing the OpenCV Module here

cap = cv2.VideoCapture(0)

while True:
    #just gonna run until user presses q...might quit after sudoku is solved or voice commmand?
    ret, frame = cap.read()

    cv2.imshow("Camera Feed", frame) #going to show the camera what we are capturing

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() #need to see effectiveness