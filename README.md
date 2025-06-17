# violence-detection-app  
A deep learning-based system that detects violence in short video clips using Convolutional Neural Networks (CNN) and a Streamlit web interface. Ideal for public surveillance and real-time safety applications.  
# Violence Detection in Real-Life Videos 🎥🧠

This project implements a deep learning-based solution to detect violent activities in short video clips. It uses a Convolutional Neural Network (CNN) for frame-wise classification and provides a simple and interactive web interface using Streamlit.

## 🔍 Features
- Accepts short videos (≤10 seconds)
- Extracts 20 frames per video
- Classifies each frame as Violent or Non-Violent using a trained CNN
- Aggregates frame predictions for final video classification
- Fast, responsive Streamlit web interface
- Easy to deploy and extend

----------------------------------------------------------------------------

## 🧠 Model Architecture

- *Input*: 128x128 RGB frames
- *Model*: Convolutional Neural Network (CNN)
- *Output*: Binary classification (Violent / Non-Violent)
- *Accuracy*: ~92% on test data

---------------------------------------------------------------------------

## 🛠 Tech Stack

- *Language*: Python 3.10  
- *Libraries*: TensorFlow, Keras, OpenCV, NumPy, Streamlit  
- *Tools*: VS Code, Git, virtualenv  
- *Hardware Used*: Intel i5, 16 GB RAM, NVIDIA RTX 3060

-----------------------------------------------------------------------

## 🚀 How to Run Locally

step 1 . : download dataset fron kaggle using the link(https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) .  
step 2 . : extract the datset .  
step 3 . : create a folder structure like this  
           violent_detection_project  /  
	         ├── data/  
	         │   ├── Violence/(copy the violence video from dataset to this folder)  
	         │   └── NonViolence/(copy the non-violence video from dataset to this folder)  
	         ├── frames/(automatically created after the frames are extracted)  
	         ├── scripts/  
	         │   ├── extract_frames.py  
	         │   ├── train_model.py  
	         │   └── utils.py  
	         ├── app.py  
	         ├── violence_detection_model.h5(automatically created after training the model)  
	         ├── venv/(atomatically create after activating virtual environment)  
           ├── background.jpg  
           
step 4 . : open the terminal and change the folder to **violent_detection_project** .
           create the virtual environment by using the command **py -3.10 -m venv venv** .
           activate the virtual environment by using the command **venv\Scripts\activate** .  

step 5 . : install the reqirements using following code .  
            **pip install tensorflow opencv-python numpy scikit-learn matplotlib pillow streamlit**  

step 6 . : navigate to **extract_frames.py** paste and run the code using ** python extract_frames.py ** .  
           navigate to **train_model.py** paste and run the code using ** python train_model.py ** .                                   (violence_detection_model.h5 model is created)  

step 7 . : Then move to app.py paste and run the code using **streamlit run app.py** .  
           Then server start to run navigate to the web app using link .  

