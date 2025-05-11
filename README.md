# FaceGuard: Deepfake Face Detection System

**FaceGuard** is an intelligent deepfake detection system designed to classify facial images as **Real** or **Fake** using a pretrained deep learning model. Built with Streamlit, TensorFlow, and OpenCV, this application delivers real-time results through an intuitive web interface.

##  Features
-  **EfficientNetB7 Model**  
  Leverages the power of a pretrained EfficientNetB7 CNN architecture (trained on ImageNet) for robust feature extraction and classification.

-  **Real-Time Image Analysis**  
  Upload a facial image and get instant classification along with a confidence score.

-  **Confidence Score Display**  
  Shows how confident the model is about its prediction.

-  **Visual Feedback**  
  Output image is annotated with a colored border — green (Real), red (Fake) — based on detection.

-  **Streamlit-Based UI**  
  Lightweight, responsive, and user-friendly interface for easy interaction.

##  Tech Stack
- Python 3.10.9  
- [TensorFlow / Keras](https://www.tensorflow.org/)  
- [EfficientNetB7](https://keras.io/api/applications/efficientnet/#efficientnetb7-function) (Pretrained on ImageNet)  
- [OpenCV](https://opencv.org/)  
- [NumPy](https://numpy.org/)  
- [Streamlit](https://streamlit.io/)  
- Base64 (for image handling)

##  How It Works
1. User uploads an image through the Streamlit UI.
2. The image is preprocessed (resized, normalized).
3. It is passed through the EfficientNetB7 model.
4. The model outputs:
   - Classification (`Real` or `Fake`)
   - Confidence score (probability)
   - Annotated image (with color-coded result)
5. Results are displayed instantly on the interface.

