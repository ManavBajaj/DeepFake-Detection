import streamlit as st
import cv2
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import base64

# Function to preprocess input image
def preprocess_image(img):
    img = cv2.resize(img, (600, 600))  # EfficientNetB7 input size
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Function to perform deepfake detection
def detect_deepfake(img):
    img = preprocess_image(img)
    preds = model.predict(img)
    if preds[0][0] > preds[0][1]:  # Assuming index 0 corresponds to 'real' and index 1 corresponds to 'fake'
        return "Real", preds[0][0]
    else:
        return "Fake", preds[0][1]

# Streamlit app
st.title('Deepfake Detection')

# Sidebar buttons
about_button = st.sidebar.checkbox('About the Project', key='about')
code_button = st.sidebar.checkbox('Code', key='code')
Home_button = st.sidebar.checkbox('Home', key='Home')

if about_button:
    st.write("""Welcome to the Deepfake Detection project!
             

 In today's digital age, the rise of deepfake technology poses significant challenges to the authenticity of visual media. Deepfakes, which are realistic but manipulated videos created using artificial intelligence techniques, have the potential to deceive and manipulate viewers.
            
 Our project aims to address this problem by developing a deep learning-based solution for detecting and mitigating deepfake content.
Our primary objective is to create a robust and accurate deepfake detection system that can identify manipulated videos with high confidence. By leveraging advanced machine learning algorithms and image processing techniques, we aim to develop a tool that can distinguish between real and fake videos, helping to combat the spread of misinformation and protect the integrity of digital media.""")
    

if code_button:
    st.code("""
import cv2
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load pre-trained EfficientNetB7 model without the top classification layer
base_model = EfficientNetB7(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer with 128 units
x = Dense(128, activation='relu')(x)

# Add a classification layer with 2 units (for real and fake classes)
predictions = Dense(2, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess input image
def preprocess_image(img):
    img = cv2.resize(img, (600, 600))  # EfficientNetB7 input size
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Function to perform deepfake detection
def detect_deepfake(img):
    img = preprocess_image(img)
    preds = model.predict(img)
    if preds[0][0] > preds[0][1]:  # Assuming index 0 corresponds to 'real' and index 1 corresponds to 'fake'
        return "Real", preds[0][0]
    else:
        return "Fake", preds[0][1]

# Load the image
image_path = "path_to_your_image.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

# Perform deepfake detection
result, confidence = detect_deepfake(image)

# Add borders around the image based on classification result
if result == "Real":
    border_color = (0, 255, 0)  # Green border for real image
else:
    border_color = (255, 0, 0)  # Red border for fake image

image_with_border = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

# Convert the image to RGB color space after adding the border
image_with_border_rgb = cv2.cvtColor(image_with_border, cv2.COLOR_BGR2RGB)

cv2.imshow("Image with Border", image_with_border_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Classification Result: {result}")
print(f"Confidence: {confidence:.2f}")
""")

if Home_button:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Load pre-trained EfficientNetB7 model without the top classification layer
        base_model = EfficientNetB7(weights='imagenet', include_top=False)

        # Add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Add a fully connected layer with 128 units
        x = Dense(128, activation='relu')(x)

        # Add a classification layer with 2 units (for real and fake classes)
        predictions = Dense(2, activation='softmax')(x)

        # Create the final model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Perform deepfake detection
        result, confidence = detect_deepfake(image)

        # Add borders around the image based on classification result
        if result == "Real":
            border_color = (0, 255, 0)  # Green border for real image
        else:
            border_color = (0, 0, 255)  # Red border for fake image

        image_with_border = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

        # Convert the image to RGB color space after adding the border
        image_with_border_rgb = cv2.cvtColor(image_with_border, cv2.COLOR_BGR2RGB)

        st.image(image_with_border_rgb, caption=f'Uploaded Image - Confidence: {confidence:.2f}', use_column_width=True)

        st.write(f"Classification Result: {result}")
