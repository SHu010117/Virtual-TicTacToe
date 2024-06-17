import cv2
import numpy as np

def detect_tic_tac_toe_grid(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny Edge Detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a copy of the original image to draw lines on
    line_image = np.copy(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Detected Tic Tac Toe Grid", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
parent_dir = "C:/Users/Siwei Hu/Desktop/AILab/Virtual-TicTacToe"
grid_image_path = parent_dir + "/assets/images/background/tic2.png"
detect_tic_tac_toe_grid(grid_image_path)
