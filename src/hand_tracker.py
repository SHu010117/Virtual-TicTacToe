import cv2
import mediapipe as mp
import math
import constants
import numpy as np

'''
Here's the file which contains the class HandTracker
By using mediapipe library we keep track of user hand through the webcam
 and uses the hand landmarks to recognize the gesture
Main functions:
    - get_hand_position
    - is_thumb_up_or_down_and fist_closed
    - fingers_up
'''

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        

    def get_hand_position(self, frame):
        
        '''
        This function detect the user's hand position and returns
         following parameters:
            - index_pos: position of index tip
            - hand_detected: booolean which tells if the webcam catches a hand or not
            - thumb_up: boolean which tells is the thumb up gesture is detected
            - thumb_down
            - fingers: an array of 5 booleans each telling if the corresponding finger 
                        is up or down
        '''
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        index_pos = None
        WIDTH, HEIGHT = constants.WIDTH, constants.HEIGHT
        hand_detected = False
        thumb_up, thumb_down = None, None
        fingers = None
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = self.fingers_up(hand_landmarks)
                thumb_up, thumb_down = self.is_thumb_up_or_down_and_fist_closed(hand_landmarks)                
                ratio_x_to_pixel = lambda x: math.ceil(x * WIDTH)
                ratio_y_to_pixel = lambda y: math.ceil(y * HEIGHT)
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx = float(np.interp(int(ratio_x_to_pixel(index_tip.x)), [150, WIDTH - 150], [0, WIDTH]))
                cy = float(np.interp(int(ratio_y_to_pixel(index_tip.y)), [150, HEIGHT - 150], [0, HEIGHT]))
                index_pos = (cx, cy)
        return index_pos, hand_detected, thumb_up, thumb_down, fingers

    def release(self):
        self.cap.release()

    def is_thumb_up_or_down_and_fist_closed(self, hand_landmarks):
        
        '''
        This functions analizes the hand_land marks and compares the key
         points to detect each fingers position and then individuate if
         the user is keeping the fist closed and with the thumb finger
         up or down (so two boolean variables are returned: thumb_up, thumb_down)
        '''
        
        # hand tips positions
        EPSILON = 0.03
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # check whether thumb's up or down
        is_thumb_up = thumb_tip.y < thumb_mcp.y
        is_thumb_down = thumb_tip.y > thumb_mcp.y

        # Check if the fist is closed:
        is_fist_closed = (
                (min(thumb_mcp.x, index_mcp.x) - EPSILON <= index_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
                (min(thumb_mcp.x, index_mcp.x) - EPSILON <= middle_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
                (min(thumb_mcp.x, index_mcp.x) - EPSILON <= ring_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
                (min(thumb_mcp.x, index_mcp.x) - EPSILON <= pinky_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON)
        )

        return is_thumb_up and is_fist_closed, is_thumb_down and is_fist_closed
    
    def fingers_up(self, hand_landmarks):
        
        '''
        Another key function which detects each finger's position and by comparing
         the key points of landmarks estabilishes if the finger's up or down
        The returning value is an array of 5 booleans in the order [thumb, index, middle, ring, little]
         and each variable is set to True if the correspoding finger is considered up, False otherwise
        We use this array to recognize different gestures and then perform the corresponding function
        '''
        
        fingers = []
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[
            self.mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(True)
        else:
            fingers.append(False)

        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
            self.mp_hands.HandLandmark.RING_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
            self.mp_hands.HandLandmark.PINKY_PIP].y)

        return fingers