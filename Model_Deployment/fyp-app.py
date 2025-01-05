import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import torch
import cv2
import os

# Function for CLAHE transformation
def apply_clahe(pil_image):
    # Convert PIL image to a NumPy array
    image_np = np.array(pil_image)
    # Convert RGB to LAB
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge the CLAHE-enhanced L-channel back with a and b
    lab = cv2.merge((cl, a, b))
    # Convert back to RGB
    enhanced_image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Convert NumPy array back to PIL Image
    enhanced_pil_image = Image.fromarray(enhanced_image_np)
    return enhanced_pil_image

# Data transformations: Padding, Resizing
data_transforms = transforms.Compose([
    transforms.Pad(padding=20, fill=(255, 255, 255)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the model
model = models.densenet121(pretrained=False)
num_classes = 3
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Construct the model full path
file_path = os.path.join(os.path.dirname(__file__), 'best_contrast_densenet121_scratch.pth')

# Construct full paths for the sample images
sample_image_path = os.path.join(os.path.dirname(__file__), 'SampleImage.jpg')
sample_crop_path = os.path.join(os.path.dirname(__file__), 'SampleROI.jpg')


model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

model.eval()

# Predict Material
def predict_image(image):
    # Apply CLAHE to the image before processing
    image = apply_clahe(image)
    # Transform and prepare the image for the model
    image = data_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities.tolist()

# Define labels for predictions
labels = ["Cotton", "Silk", "Wool"]

# Initialize session states
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None
if "prediction_probabilities" not in st.session_state:
    st.session_state["prediction_probabilities"] = None
if "canvas_locked" not in st.session_state:
    st.session_state["canvas_locked"] = False
if "cropped_image" not in st.session_state:
    st.session_state["cropped_image"] = None
if "show_canvas" not in st.session_state:
    st.session_state["show_canvas"] = True

# Streamlit App
st.title("Image-Based Clothing Material Identification System")

# Sidebar: Help/Guide
# Sidebar: Enhanced Help/Guide
with st.sidebar:
    st.header("📖 Help & Guide")
    
    # Section 1: Overview
    st.subheader("🔍 Overview")
    st.markdown("""
    This tool helps identify the material of fabric using image processing and AI.
    Follow these steps to get accurate predictions.
    """)

    # Section 2: Step-by-Step Instructions
    st.subheader("🛠️ Steps to Use")
    st.markdown("""
    1. **Upload an Image**: Upload a clear image of the fabric. Ensure the fabric fills most of the image.
    2. **Draw ROI**: Select the region of interest (ROI) by drawing a rectangle around the fabric.
    3. **Confirm ROI**: Click the "Confirm Selection" button.
    4. **Predict**: Click "Predict Now" to identify the material.
    5. **View Results**: The material and confidence levels will be displayed.
    """)

    # Section 3: Cropping Tips
    st.subheader("✂️ Cropping Tips")
    st.image(sample_image_path, caption="Sample Full Image", use_column_width=True)
    st.image(sample_crop_path, caption="Ideal ROI Example", use_column_width=True)
    st.markdown("""
    - Ensure the fabric is **well-lit** and **in focus**.
    - Avoid background clutter. Only the fabric should be in the box.
    - Try to box the texture details for better predictions.
    """)

    # Section 4: Common Issues
    st.subheader("❓ Common Issues")
    st.markdown("""
    - **Prediction Accuracy**: Ensure the fabric texture is clearly visible.
    - **Multiple ROIs**: Only one rectangle is allowed. Reset and try again if you draw multiple ROIs.
    - **Image Upload Issues**: Supported formats are JPG, PNG, and JPEG.
    """)

    # Section 5: Support
    st.subheader("📩 Need Help?")
    st.markdown("""
    If you encounter any issues, contact us at **tp062856@mail.apu.edu.my**.
    """)

    # Section 6: Disclaimer
    st.markdown('<div class="sidebar-subheader">⚠️ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="sidebar-text">
    This tool is developed for <span class="highlight">academic purposes</span> as part of a final year project. 
    While the predictions are based on trained AI models, they may contain <span class="highlight">errors or inaccuracies</span>. 
    Users are advised not to rely solely on the results for critical decisions.
    </p>
    """,
    unsafe_allow_html=True)

# Step 1: Upload Image
st.subheader("Step 1: Upload Your Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    fixed_width = 600
    aspect_ratio = image.height / image.width
    resized_image = image.resize((fixed_width, int(fixed_width * aspect_ratio)))
    st.session_state["resized_image"] = resized_image

    # Step 2: Draw ROI
    if st.session_state["show_canvas"]:
        st.subheader("Step 2: Select Region of Interest (ROI)")

        # Ensure resized_image is valid and in RGBA format
        background_image = resized_image.convert("RGBA")
        st.write(f"Type of background_image: {type(background_image)}")
        st.write(f"Background image size: {background_image.size}")

        # Render the canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill
            stroke_width=3,
            background_image=background_image,  # Pass RGBA image
            update_streamlit=True,
            height=background_image.height,  # Match canvas height to image
            width=background_image.width,  # Match canvas width to image
            drawing_mode="rect",
            key="canvas",
        )

        # Log canvas result
        st.write("Canvas result (debug):", canvas_result)

        # Check for multiple boxes
        if canvas_result and canvas_result.json_data:
            if len(canvas_result.json_data["objects"]) > 1:
                st.warning("Only one bounding box is allowed. Click 'Undo' to revert action.")
            elif len(canvas_result.json_data["objects"]) == 1:
                confirm_button = st.button("Confirm Selection")
                if confirm_button:
                    obj = canvas_result.json_data["objects"][0]
                    scale_factor = image.width / fixed_width
                    x, y, w, h = map(lambda v: int(v * scale_factor), [obj["left"], obj["top"], obj["width"], obj["height"]])
                    cropped_image = np.array(image)[y:y + h, x:x + w]
                    cropped_image = Image.fromarray(cropped_image)

                    # Maintain padding and aspect ratio for cropped ROI display
                    cropped_image_padded = ImageOps.expand(cropped_image, border=(20, 20, 20, 20), fill="white")
                    st.session_state["cropped_image"] = cropped_image_padded
                    st.session_state["show_canvas"] = False
                    st.experimental_rerun()

    # Step 3: Prediction Results
    if st.session_state["cropped_image"]:
        st.subheader("Step 3: Crop Results")

        # Set display height to match both images
        display_height = 300  # Fixed height for both images
        aspect_ratio_original = st.session_state["resized_image"].width / st.session_state["resized_image"].height
        resized_display_image = st.session_state["resized_image"].resize((int(display_height * aspect_ratio_original), display_height))

        aspect_ratio_cropped = st.session_state["cropped_image"].width / st.session_state["cropped_image"].height
        resized_display_cropped = st.session_state["cropped_image"].resize((int(display_height * aspect_ratio_cropped), display_height))

        # Add padding between columns
        col1, padding_col, col2 = st.columns([1, 0.05, 1])  # Extra padding in the middle column
        with col1:
            st.image(
                resized_display_image,
                caption="Original Image",
                use_column_width=False,
                output_format="JPEG",
            )
        with padding_col:
            st.markdown("")  # Empty padding column to create spacing
        with col2:
            st.image(
                resized_display_cropped,
                caption="Cropped ROI",
                use_column_width=False,
                output_format="JPEG",
            )


        # Buttons and Prediction
        if not st.session_state["prediction_done"]:
            if st.button("Predict Now"):
                predicted_label, probabilities = predict_image(st.session_state["cropped_image"])
                st.session_state["prediction_results"] = labels[predicted_label]
                st.session_state["prediction_probabilities"] = probabilities
                st.session_state["prediction_done"] = True
                st.experimental_rerun()
        else:
            # Display Prediction Results
            st.subheader("Prediction Results")
            ranked_predictions = sorted(
                zip(labels, st.session_state["prediction_probabilities"]),
                key=lambda x: x[1],
                reverse=True
            )
            for rank, (label, prob) in enumerate(ranked_predictions, start=1):
                st.write(f"Rank {rank}: **{label}** with probability **{prob:.2%}**")

            # Recrop and Start Over Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Recrop"):
                    st.session_state["cropped_image"] = None
                    st.session_state["show_canvas"] = True
                    st.session_state["prediction_done"] = False
                    st.session_state["prediction_results"] = None
                    st.session_state["prediction_probabilities"] = None
                    st.experimental_rerun()
            with col2:
                if st.button("Start Over"):
                    st.session_state.clear()
                    st.markdown("""
                        <meta http-equiv="refresh" content="0; url=." />
                    """, unsafe_allow_html=True)
