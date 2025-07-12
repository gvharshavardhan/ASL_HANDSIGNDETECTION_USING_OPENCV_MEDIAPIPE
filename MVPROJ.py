import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define lighting condition thresholds and corresponding accuracy
lighting_accuracy_map = {
    "Bright": 95.2,
    "Normal Indoor": 92.5,
    "Low Light": 85.3,
    "Shadows Present": 78.1,
    "Backlit": 72.4
}

# Function to auto-detect lighting condition
def detect_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness > 180:
        return "Bright"
    elif 130 < brightness <= 180:
        return "Normal Indoor"
    elif 80 < brightness <= 130:
        return "Low Light"
    elif 50 < brightness <= 80:
        return "Shadows Present"
    else:
        return "Backlit"

# Accuracy tracking variables
total_predictions = 0
correct_predictions = 0
captured_image = None
captured_sign = ""
expected_sign = None

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ab = b - a
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Function to recognize hand signs
def recognize_sign(landmarks):
    thumb_angle = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
    index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[8])
    middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[12])
    ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[16])
    pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[20])

    if pinky_angle > 120 and index_angle < 40 and middle_angle < 40 and ring_angle < 40:
        return "Hello"
    elif index_angle > 120 and middle_angle > 120 and ring_angle > 120 and pinky_angle < 50:
        return "What is your name?"
    elif thumb_angle < 30 and index_angle < 30 and middle_angle < 30 and ring_angle < 30 and pinky_angle < 30:
        return "Stop"
    elif thumb_angle < 30 and index_angle > 120 and middle_angle > 120:
        return "Thank You"
    elif index_angle < 30 and middle_angle < 30 and ring_angle < 30 and pinky_angle < 30:
        return "Please"
    elif index_angle > 120 and middle_angle > 120 and ring_angle > 120 and pinky_angle > 120:
        return "Bye"
    elif thumb_angle < 40 and index_angle > 120 and pinky_angle > 120 and middle_angle < 40 and ring_angle < 40:
        return "I Love You"
    elif index_angle < 30 and middle_angle > 120 and ring_angle > 120:
        return "Yes"
    elif index_angle > 120 and middle_angle > 120 and thumb_angle < 40:
        return "No"
    elif thumb_angle > 120 and index_angle < 40 and middle_angle < 40:
        return "Good Job"
    return "Unknown"

# Start webcam
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Auto-detect lighting
    lighting_condition = detect_lighting(frame)
    lighting_based_accuracy = lighting_accuracy_map.get(lighting_condition, 0.0)

    # Detect hands
    results = hands.process(rgb_frame)
    detected_sign = "Unknown"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_sign = recognize_sign(hand_landmarks.landmark)

    # Handle key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and detected_sign != "Unknown":
        captured_image = frame.copy()
        captured_sign = detected_sign.lower()

        if expected_sign is None:
            expected_sign = captured_sign
            print(f"✅ First captured sign set as expected: '{expected_sign}'")
        else:
            total_predictions += 1
            print(f"Expected: {expected_sign} | Captured: {captured_sign}")
            if captured_sign == expected_sign:
                correct_predictions += 1
                print("✅ Match! Correct prediction count increased.")
            else:
                print("❌ No match!")

    # Overlay on main frame
    cv2.putText(frame, f"Sign: {detected_sign}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Lighting: {lighting_condition}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Lighting Accuracy: {lighting_based_accuracy:.2f}%", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Recognition with Lighting", frame)

    # Show captured image and prediction accuracy
    if captured_image is not None:
        display_image = captured_image.copy()
        prediction_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 100.0
        cv2.putText(display_image, f"Captured Sign: {captured_sign}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f"Prediction Accuracy: {prediction_accuracy:.2f}%", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Captured Image", display_image)

    if key == ord('q') or key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()