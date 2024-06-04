import cv2
import numpy as np
import mediapipe as mp
import math

# Define the size and properties of the Tic Tac Toe grid
line_thickness = 2
line_color = (100, 100, 100)  # Black color

class HandControlTicTacToe:
    def __init__(self):
        # initialize medialpipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # activate draw function
        self.draw_active = False
        self.image = None

        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.xp = 0
        self.yp = 0

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
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # analys the fingers and save their position
                        landmark_list = []

                        # hands' position matrix
                        paw_x_list = []
                        paw_y_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)
                        if landmark_list:
                            # convertion to pixel coordinates
                            ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                            ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)

                            # middle fingers tip position
                            middle_finger_tip = landmark_list[12]
                            middle_finger_tip_x = ratio_x_to_pixel(middle_finger_tip[1])
                            middle_finger_tip_y = ratio_y_to_pixel(middle_finger_tip[2])

                            # index finger tip position
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[1])
                            index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[2])
                            # middle point between 2 finger tips
                            between_finger_tip = (middle_finger_tip_x + index_finger_tip_x) // 2, (
                                        middle_finger_tip_y + index_finger_tip_y) // 2
                            # print(middle_finger_tip_x)
                            thumb_finger_point = (middle_finger_tip_x, middle_finger_tip_y)
                            index_finger_point = (index_finger_tip_x, index_finger_tip_y)
                            # draw the two finger tips point
                            circle_func = lambda point: cv2.circle(self.image, point, 10, (255, 0, 255), -1)
                            self.image = circle_func(thumb_finger_point)
                            self.image = circle_func(index_finger_point)
                            self.image = circle_func(between_finger_tip)
                            # draw the line between finger tips
                            self.image = cv2.line(self.image, thumb_finger_point, index_finger_point, (255, 0, 255), 3)
                            # get the line segment's length
                            line_len = math.hypot((index_finger_tip_x - middle_finger_tip_x),
                                                  (index_finger_tip_y - middle_finger_tip_y))


                    if line_len < 60:
                        # we can draw
                        # draw function to be implemented correctly
                        if (self.xp == 0) and (self.yp == 0):
                            self.xp, self.yp = between_finger_tip
                        start_point = (int(self.xp), int(self.yp))                                
                        end_point = (int(between_finger_tip[0]), int(between_finger_tip[1]))
                        cv2.line(self.canvas, start_point, end_point, line_color, 5)

                    self.xp, self.yp = between_finger_tip[0], between_finger_tip[1]

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
