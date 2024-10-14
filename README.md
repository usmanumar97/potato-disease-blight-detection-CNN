# Potato Disease Blight Detection Using CNN 🌿🤖

This project is focused on detecting late blight disease in potato plants using Convolutional Neural Networks (CNN). The dataset used is from the **PlantVillage** repository, which contains images of healthy and infected potato plants.

## 🔍 Project Overview
- **Objective**: Build a CNN model to classify potato leaf images as healthy or infected with late blight.
- **Dataset**: PlantVillage dataset containing healthy and blight-affected potato leaf images.
- **Deep Learning Framework**: TensorFlow and Keras were used to build and train the model.
- **Image Preprocessing**: Images were normalized by rescaling them between `[0,1]`.

## 🛠️ Installation and Usage

### Clone the Repository

```bash
git clone https://github.com/usmanumar97/potato-disease-blight-detection-CNN.git
cd potato-disease-blight-detection-CNN 
```
### 🛠️ Installation Dependencies
```
pip install -r requirements.txt
```
### 🛠️ Run the FASTAPI Server
```
uvicorn api.main:app --reload
```

🏅 Results
The model achieved high accuracy on both the training and validation datasets. Below is a plot of the accuracy and loss over 50 epochs.

🚀 Prediction Example
Here’s an example of a prediction made by the model:

Actual Class: Late Blight
Predicted Class: Late Blight
Confidence: 95%

✨ Journey to Learning CNN
This project was a great learning experience in understanding the core workings of Convolutional Neural Networks (CNN). Below are some key concepts learned:

CNN Layers:  
Understanding how convolution, max pooling, and fully connected layers help in feature extraction and classification.  
Data Augmentation: Learning how augmenting data helps prevent overfitting and improves the model's robustness.  
Model Optimization: Tuning hyperparameters like learning rate, batch size, and epochs to improve model performance.  

📈 Future Enhancements  
Hyperparameter Tuning: Experiment with different optimizers and learning rates to further enhance accuracy.  
Transfer Learning: Explore pre-trained models like VGG16 or ResNet to improve prediction performance.  
Deployment: Deploy the model using Docker and host it on cloud platforms like AWS or Google Cloud.  
