🖐️ Hand Gesture Recognition using Deep Learning
📌 Project Overview
This project is a Hand Gesture Recognition system built using Convolutional Neural Networks (CNN).
It classifies hand gestures from the LeapGestRecog dataset into 10 categories.
The system learns patterns from hand images and predicts the gesture shown.
🎯 Objective
Recognize hand gestures from images
Build a CNN-based image classification model
Enable gesture-based human-computer interaction
📂 Dataset
Name: LeapGestRecog
Source: Kaggle
Classes: 10 gesture categories (00–09)
Type: Grayscale hand gesture images
🛠️ Technologies Used
Python 🐍
OpenCV 👁️
NumPy 🔢
Scikit-learn ⚙️
TensorFlow / Keras 🤖
CNN (Deep Learning)
🧠 Model Architecture
Conv2D (32 filters) + MaxPooling
Conv2D (64 filters) + MaxPooling
Flatten Layer
Dense Layer (128 neurons)
Dropout (0.5)
Output Layer (Softmax)
⚙️ Installation
1️⃣ Install dependencies
Bash
pip install numpy opencv-python tensorflow scikit-learn
2️⃣ Run the project
Bash
python task4.py
📁 Project Structure
Plain text
pro4/
│
├── task4.py
├── gesture_model.h5
├── leapGestRecog/
└── README.md
📊 Output
After running the program:
Plain text
🚀 Loading dataset...
✅ Images loaded: 10000+
🧠 Data ready
🔥 Training started...
Epoch 1/5 ...
Epoch 5/5 ...
🎉 Task 4 Completed Successfully!
🎯 Result
Model successfully classifies hand gestures
CNN trained on LeapGestRecog dataset
Model saved as gesture_model.h5
🚀 Future Improvements
Real-time webcam gesture detection
Mobile app integration
Improved accuracy using augmentation
Gesture-based smart control systems
