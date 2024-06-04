import cv2
import numpy as np
import mediapipe as mp
import math

# Define the size and properties of the Tic Tac Toe grid
line_thickness = 2
line_color = (100, 100, 100) 

class HandControlTicTacToe:
    def __init__(self):
        # initialize medialpipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.fingers = []

        # activate draw function
        self.draw_active = False
        self.image = None

        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.xp = 0
        self.yp = 0
    
    def fingers_up(self, hand_landmarks):
        """
        Determine which fingers are up.
        
        Arguments:
        hand_landmarks -- list of hand landmarks

        Returns:
        list of booleans representing whether each finger is up (True) or down (False)
        """
        fingers = []

        # Thumb (landmark 4 is the tip, landmark 3 is the knuckle)
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(True)
        else:
            fingers.append(False)

        # Fingers (landmark 8 is the tip, landmark 6 is the pip joint)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y)

        return fingers

    # 主函数
    def recognize(self):
        

        # OpenCV video capture
        cap = cv2.VideoCapture(0)
        # sizes
        resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with self.mp_hands.Hands(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            
            while cap.isOpened():

                # initial image
                success, self.image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                self.image = cv2.resize(self.image, (resize_w, resize_h))
                
                # Calculate positions for the grid lines
                third_width = resize_w // 3
                third_height = resize_h // 3

                # Draw the vertical lines
                cv2.line(self.image, (third_width, 0), (third_width, resize_h), line_color, line_thickness)
                cv2.line(self.image, (2 * third_width, 0), (2 * third_width, resize_h), line_color, line_thickness)

                # Draw the horizontal lines
                cv2.line(self.image, (0, third_height), (resize_w, third_height), line_color, line_thickness)
                cv2.line(self.image, (0, 2 * third_height), (resize_w, 2 * third_height), line_color, line_thickness)
                
                # improve performance
                self.image.flags.writeable = False
                # cenvert to RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # flip the image
                self.image = cv2.flip(self.image, 1)
                # mediapipe model process
                results = hands.process(self.image)

                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                # check if there a hand to be captured
                if results.multi_hand_landmarks:
                    # for the whole hand
                    for hand_landmarks in results.multi_hand_landmarks:
                        # detect the fingers
                        self.mp_drawing.draw_landmarks(
                            self.image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS)
                        
                        self.fingers = self.fingers_up(hand_landmarks)


                        # analys the fingers and save their position, but we actuallt need only the index finger tip to draw
                            
                        
                        # Conversion to pixel coordinates
                        ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                        ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)

                        # Index finger tip position
                        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_finger_tip_x = ratio_x_to_pixel(index_finger_tip.x)
                        index_finger_tip_y = ratio_y_to_pixel(index_finger_tip.y)
                        index_finger_point = (index_finger_tip_x, index_finger_tip_y)
                        
                        
                    if self.fingers[1] and not self.fingers[2]:
                        # we can draw
                        # draw function to be implemented correctly
                        if (self.xp == 0) and (self.yp == 0):
                            self.xp, self.yp = index_finger_point
                        start_point = (int(self.xp), int(self.yp))                                
                        end_point = (int(index_finger_tip_x), int(index_finger_tip_y))
                        cv2.line(self.canvas, start_point, end_point, line_color, 5)
                    
                    self.xp, self.yp = index_finger_tip_x, index_finger_tip_y
                
                # Prepare the canvas for overlay
                grayscale_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, canvas_inv = cv2.threshold(grayscale_canvas, 50, 255, cv2.THRESH_BINARY_INV)
                canvas_inv = cv2.cvtColor(canvas_inv, cv2.COLOR_GRAY2BGR)
                self.image = cv2.bitwise_and(self.image, canvas_inv)
                self.image = cv2.bitwise_or(self.image, self.canvas)
                
                cv2.imshow('TIC_TAC_TOE', self.image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()


# run
control = HandControlTicTacToe()
control.recognize()