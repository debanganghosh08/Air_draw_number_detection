# 🎨 AI-Powered Air Draw - Hand Gesture Digit Recognition

## 📌 Overview
This project allows users to draw numbers in the air using their fingers, which are then recognized using a CNN model trained on the MNIST dataset. The predicted number is spoken aloud using a Text-to-Speech engine. The system uses OpenCV, MediaPipe, and TensorFlow to track hand gestures and predict numbers accurately.

---

## ✋ Hand Detection Logic

To detect the hand and track finger movements, we use **MediaPipe Hands**:

```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
```

- `min_detection_confidence=0.8`: Ensures reliable hand detection.
- `min_tracking_confidence=0.8`: Maintains smooth tracking even with motion.
- The detected hand landmarks help locate the index finger tip to enable drawing.

---

## ✏️ Drawing and Predicting Numbers

### 🖐️ Finger Detection & Drawing Logic

The index finger tip coordinates are extracted and used to draw on the screen:

```python
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[8]
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)  # Draw green path
```

- The **index finger tip (landmark[8])** coordinates are converted to pixel values.
- `cv2.circle()` draws a green dot at each point, forming a continuous path.

### 🔢 Predicting the Drawn Number

Once the user presses `p`, the drawn digit is preprocessed and fed into the trained CNN model:

```python
def preprocess_image(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, (28, 28))
    return resized / 255.0
```

```python
def predict_number():
    img = preprocess_image(canvas).reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    number = np.argmax(prediction)
    print(f"Predicted Number: {number}")
    tts_engine.say(f"You drew the number {number}")
    tts_engine.runAndWait()
    return number
```

- **Preprocessing Steps:** Convert the image to grayscale, apply Gaussian Blur and thresholding, resize to (28x28), and normalize pixel values.
- The processed image is reshaped and passed into the CNN model.
- The predicted number is announced using **pyttsx3** Text-to-Speech.

### 🏋️ Training the CNN Model with MNIST Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

- **MNIST Dataset:** 60,000 training images, 10,000 test images.
- Images are reshaped to match the CNN’s input shape (28x28x1).
- Labels are one-hot encoded for categorical classification.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')  # Output layer for 10 digits
])
```

- Two convolutional layers extract features.
- **MaxPooling** reduces dimensions to improve efficiency.
- **Dropout** prevents overfitting.
- **Softmax activation** predicts the probability for each digit (0-9).

---

## 🖥️ Code Breakdown

### 📷 Initializing Camera & Setting Up OpenCV
```python
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow("Air Draw - Number Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Draw - Number Recognition", 720, 1000)
```
- **Opens webcam feed** and creates a drawing canvas.
- **Resizes the window** for better visualization.

### 🎮 Keyboard Controls
```python
if key == ord('p'):  # Predict Number
    predict_number()
    time.sleep(2)  # Pause
elif key == ord('c'):  # Clear Canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
elif key == ord('q'):  # Quit
    break
```
- `p`: Predict the drawn number.
- `c`: Clear the screen.
- `q`: Quit the application.

---

## 📌 How to Run the Project

### ⚙️ Requirements
- Python 3.8+
- OpenCV
- NumPy
- TensorFlow
- MediaPipe
- pyttsx3

### 📥 Installation
```bash
pip install opencv-python numpy tensorflow mediapipe pyttsx3
```

### 🚀 Running the Project
```bash
python main.py
```

---

## 📢 Conclusion
This AI-powered "Air Draw" project provides an interactive and intuitive way to draw and recognize numbers using hand gestures. It utilizes deep learning with CNNs, real-time hand tracking, and speech synthesis for an engaging experience. 🚀✨

