# 🚀 Installation Guide for Air Draw - Number Recognition

## 📥 Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- Pip (Python package manager)

## 📌 Step 1: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/debanganghosh08/Air_draw_number_detection.git
cd Air_draw_number_detection
```

## 📌 Step 2: Create a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to manage dependencies:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 📌 Step 3: Install Dependencies
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is missing, install manually:
```bash
pip install opencv-python numpy tensorflow mediapipe pyttsx3
```

## 📌 Step 4: Run the Project
Start the application using:
```bash
python main.py
```

## 🛠️ Troubleshooting
- If `tensorflow` installation fails, try:
  ```bash
  pip install tensorflow --upgrade
  ```
- If `pyttsx3` doesn’t work properly on Linux, install additional dependencies:
  ```bash
  sudo apt install espeak ffmpeg libespeak1
  ```

## 🎉 You're all set!
Now you can start drawing numbers in the air and let AI recognize them! ✨

