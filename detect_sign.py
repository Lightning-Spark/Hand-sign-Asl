# import cv2
# import mediapipe as mp
# import numpy as np
# import os

# # Load the KNN model
# model = np.load('models/knn_model.npy', allow_pickle=True)

# # Initialize Mediapipe Hand class
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize variables
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# prev_label = None
# label_count = 0

# # Start capturing video
# cap = cv2.VideoCapture(0)

# while True:
#     # Read frame from camera
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for a mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert the frame to RGB and process it with Mediapipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     # If hand landmarks are detected
#     if results.multi_hand_landmarks:
#         # Get the first hand landmark
#         hand_landmarks = results.multi_hand_landmarks[0]

#         # Get the landmarks as a flat array of (x, y, z) coordinates
#         landmarks = []
#         for lm in hand_landmarks.landmark:
#             landmarks.append((lm.x, lm.y, lm.z))
#         landmarks = np.array(landmarks).flatten()

#         # Use the KNN model to predict the label
#         label = model.predict([landmarks])[0]

#         # Check if the predicted label is the same as the previous label
#         if label == prev_label:
#             label_count += 1
#         else:
#             prev_label = label
#             label_count = 0

#         # If the same label has been predicted for a certain number of frames, display it on the screen
#     if label_count == 10:
#         if label == 'A':
#             cv2.putText(frame, 'A', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'B':
#             cv2.putText(frame, 'B', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'C':
#             cv2.putText(frame, 'C', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'D':
#             cv2.putText(frame, 'D', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'E':
#             cv2.putText(frame, 'E', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'F':
#             cv2.putText(frame, 'F', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'G':
#             cv2.putText(frame, 'G', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'H':
#             cv2.putText(frame, 'H', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'I':
#             cv2.putText(frame, 'I', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'J':
#             cv2.putText(frame, 'J', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'K':
#             cv2.putText(frame, 'K', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'L':
#             cv2.putText(frame, 'L', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'M':
#             cv2.putText(frame, 'M', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'N':
#             cv2.putText(frame, 'N', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'O':
#             cv2.putText(frame, 'O', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'P':
#             cv2.putText(frame, 'P', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'Q':
#             cv2.putText(frame, 'Q', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'R':
#             cv2.putText(frame, 'R', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'S':
#             cv2.putText(frame, 'S', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'T':
#             cv2.putText(frame, 'T', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'U':
#             cv2.putText(frame, 'U', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'V':
#             cv2.putText(frame, 'V', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'W':
#             cv2.putText(frame, 'W', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'X':
#             cv2.putText(frame, 'X', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'Y':
#             cv2.putText(frame, 'Y', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         elif label == 'Z':
#             cv2.putText(frame, 'Z', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
#         # Display the frame
#         cv2.imshow('Hand Sign Detection', frame)

#         # Exit the program when the 'q' key is pressed
#         if cv2.waitKey(1) == ord('q'):
#             break

# # Release the video capture and destroy the window
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the KNN model
filename = 'models/knn_model_k5.joblib'
knn = joblib.load(filename)

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

        # Use the KNN model to predict the hand sign
        label = knn.predict([landmarks])[0]

        # Draw the predicted label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (200, 200), font, 3, (0, 0, 255), 2)

    # Draw hand landmarks on the frame
    annotated_frame = frame.copy()
    mp_drawing = mp.solutions.drawing_utils
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Sign Detection', annotated_frame)

    # Exit the program when the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
