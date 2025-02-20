import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
from tensorflow.keras.models import load_model

# Load the improved CNN model
model = load_model("improved_digit_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

# Initialize OpenCV
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Set window to half-screen size
cv2.namedWindow("Air Draw - Number Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Draw - Number Recognition", 720, 1000)

def preprocess_image(canvas):
    """Convert drawing to a clean 28x28 image for CNN."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, (28, 28))
    return resized / 255.0

def predict_number():
    """Predict the drawn number and provide voice output."""
    global canvas
    img = preprocess_image(canvas).reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    number = np.argmax(prediction)

    print(f"Predicted Number: {number}")
    
    # Speak the number
    tts_engine.say(f"You drew the number {number}")
    tts_engine.runAndWait()
    
    return number

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape

    # Ensure canvas size matches frame
    if canvas.shape[:2] != (h, w):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert frame to RGB & process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)  # Draw green path

    # Overlay drawing on frame
    output_frame = cv2.addWeighted(frame, 1, canvas, 0.6, 0)

    cv2.imshow("Air Draw - Number Recognition", output_frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Predict Number
        predict_number()
        time.sleep(2)  # Pause
    elif key == ord('c'):  # Clear Canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
