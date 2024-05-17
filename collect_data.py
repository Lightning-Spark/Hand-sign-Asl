import cv2
import mediapipe as mp
import numpy as np
import os

# Create a folder to store the collected data
if not os.path.exists('data'):
    os.makedirs('data')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize variables
counter = 0
label = input("Enter the label for the hand sign: ")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB and process it with Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Get the first hand landmark
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the landmarks as a flat array of (x, y, z) coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
        landmarks = np.array(landmarks).flatten()

        # Save the data as a .npy file
        np.save(f"data/{label}_{counter}.npy", landmarks)
        counter += 1
        print(counter)

    # Draw hand landmarks on the frame
    annotated_frame = frame.copy()
    mp_drawing = mp.solutions.drawing_utils
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Sign Detection', annotated_frame)
    cv2.setWindowProperty('Hand Sign Detection', cv2.WND_PROP_TOPMOST, 1)

    # Exit the program when the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
