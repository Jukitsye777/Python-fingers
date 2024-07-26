import cv2
import mediapipe as mp
import math
import serial
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2)


# Initialize serial port (adjust 'COM3' and 9600 to match your configuration)
ser = serial.Serial('COM16', 9600, timeout=1)

# Allow time for the serial connection to establish
time.sleep(2)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image)

    # Convert RGB image back to BGR for OpenCV
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and calculate distances
            landmarks = hand_landmarks.landmark

            # Landmark indices and labels
            landmark_pairs = [
                (4, 2),  # Thumb: (idx4, idx2)
                (6, 5),  # Index: (idx6, idx5)
                (12, 9),  # Middle: (idx12, idx9)
                (16, 13),  # Ring: (idx16, idx13)
                (20, 7)  # Pinky: (idx20, idx7)
            ]

            distances = [0] * len(landmark_pairs)  # List to store distances

            for idx, (idx1, idx2) in enumerate(landmark_pairs):
                landmark1 = landmarks[idx1]
                landmark2 = landmarks[idx2]
                distance = calculate_distance(landmark1, landmark2)

                # Convert normalized coordinates to pixel coordinates
                x1, y1 = int(landmark1.x * frame.shape[1]), int(landmark1.y * frame.shape[0])
                x2, y2 = int(landmark2.x * frame.shape[1]), int(landmark2.y * frame.shape[0])

                # Draw the distance on the frame
                distance_text = f"{distance:.2f}"
                cv2.putText(frame, distance_text, ((x1 + x2) // 2, (y1 + y2) // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)

                # Draw lines between landmarks
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Store the distance in the list
                distances[idx] = f"{distance:.2f}"

            # Convert distances list to a formatted string and send via serial
            distances_str = ",".join(distances)
            ser.write(f"{distances_str}\n".encode())

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the image
    cv2.imshow('MediaPipe Hands', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Close the serial connection
ser.close()
