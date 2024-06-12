import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
EPSILON = 0

def is_thumb_up_and_fist_closed(hand_landmarks):
    # Punti chiave dita
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]


    # Verificare se il pollice è alzato
    is_thumb_up = thumb_tip.y < thumb_mcp.y

    # Verificare se il pugno è chiusto:
    is_fist_closed = (
        (min(thumb_mcp.x, index_mcp.x) - EPSILON <= index_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
        (min(thumb_mcp.x, index_mcp.x) - EPSILON <= middle_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
        (min(thumb_mcp.x, index_mcp.x) - EPSILON <= ring_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
        (min(thumb_mcp.x, index_mcp.x) - EPSILON <= pinky_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON)
    )

    return is_thumb_up and is_fist_closed

# Aprire la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convertire l'immagine in RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rilevare le mani
    results = hands.process(image)

    # Convertire l'immagine di nuovo in BGR per OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Disegnare i punti chiave delle mani
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Controllare se il segno è OK
            if is_thumb_up_and_fist_closed(hand_landmarks):
                cv2.putText(image, 'OK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print("Pollice alzato")

    # Mostrare l'immagine
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()