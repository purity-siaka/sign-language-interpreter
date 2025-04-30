# 🤟 Sign Language Interpreter (Python + Mediapipe + TensorFlow)

This is a real-time Sign Language Interpreter using a Convolutional Neural Network (CNN) and MediaPipe for hand tracking. It uses your webcam to detect and interpret ASL signs.

---

## 📦 Features

- Real-time hand gesture detection using MediaPipe
- CNN model trained on ASL alphabet images
- Live prediction and display
- Python + OpenCV + TensorFlow

---

## 🚀 Getting Started

### 1. Clone the Repository

If using Git Bash or Terminal:

```bash
git clone https://github.com/your-username/sign-language-interpreter-py310.git
cd sign-language-interpreter-py310
```

### 2. Set Up Virtual Environment
   
```bash
python -m venv venv
venv\Scripts\activate  
```

### 3. Install Dependencies
If you already have a requirements.txt file:

```bash
pip install -r requirements.txt
```
Or manually install packages:

```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy pandas matplotlib
```
### 4. Train the Model
Ensure your dataset folder (asl_alphabet_train) is in the project directory.
Then run:

```bash
python train_model.py
```
This will create a asl_model.h5 file after training.

### 5. Run the Live Sign Language Detector
Make sure:
You’re inside your virtual environment
The webcam is on
asl_model.h5 exists
Then run:

```bash
python predict_live.py
```

### 📂 Folder Structure

```bash
sign-language-interpreter-py310/
│
├── train_model.py          # Trains the CNN model
├── predict_live.py         # Runs webcam interpreter
├── asl_model.h5            # Saved model (after training)
├── .gitignore
├── README.md
└── asl_alphabet_train/     # ASL image dataset
```

### ✅ TODO
 Improve accuracy with more training data
 Add GUI for easier interaction
 Support full sentence detection (not just letters)

### 🧠 Credits
Dataset: ASL Alphabet Dataset by M. Massey
Libraries: OpenCV, TensorFlow, MediaPipe

### 📄 License
MIT — free for personal or commercial use.

---









