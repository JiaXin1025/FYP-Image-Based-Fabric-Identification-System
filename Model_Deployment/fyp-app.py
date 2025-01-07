import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import torch
import cv2
import os

# Initialize session states
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None
if "prediction_probabilities" not in st.session_state:
    st.session_state["prediction_probabilities"] = None
if "canvas_locked" not in st.session_state:
    st.session_state["canvas_locked"] = False
if "cropped_image" not in st.session_state:  # Ensure cropped_image is initialized
    st.session_state["cropped_image"] = None
if "show_canvas" not in st.session_state:  # Ensure show_canvas is initialized
    st.session_state["show_canvas"] = True
if "last_uploaded_file" not in st.session_state:  # Ensure last_uploaded_file is initialized
    st.session_state["last_uploaded_file"] = None

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

# Construct full paths for the sample images
sample_image_path = os.path.join(os.path.dirname(__file__), 'SampleImage.jpg')
sample_crop_path = os.path.join(os.path.dirname(__file__), 'SampleROI.jpg')
sample_crop_path_2 = os.path.join(os.path.dirname(__file__), 'SampleROI2.jpg')
model_path = os.path.join(os.path.dirname(__file__), 'best_contrast_densenet121_scratch.pth')

# Load the model (only executed once!)
@st.cache(allow_output_mutation=True)  # Use allow_output_mutation to cache the model object
def load_model():
    # Load the pre-trained model
    model = models.densenet121(pretrained=False)
    num_classes = 3
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set model to evaluation mode
    model.eval()
    return model

# Load the model once
model = load_model()

# Perform a prediction (Example function)
def predict_image(image):
    # Apply image preprocessing (example)
    image = apply_clahe(image)
    image = data_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities.tolist()

# Define labels for predictions
labels = ["Cotton", "Silk", "Wool"]

# Streamlit App
st.title("Image-Based Clothing Material Identification System")

# Sidebar: Help/Guide
with st.sidebar:
    st.header("📖 Help & Guide")
    
    # Section 1: Overview
    st.subheader("🔍 Overview")
    st.markdown("""
    This tool helps identify the material of fabric using image processing and AI.
    Follow these steps to get accurate predictions.

    **Supported Materials:** Cotton, Silk, Wool  
    *Future updates will expand support to more materials.*
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
    st.image(sample_crop_path, caption="Ideal ROI Example 1", use_column_width=True)
    st.image(sample_crop_path_2, caption="Ideal ROI Example 2", use_column_width=True)

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
    st.subheader("⚠️ Disclaimer")
    st.markdown("""
    This tool is developed for academic purposes as part of a final year project. 
    While the predictions are based on trained AI models, they may contain inaccuracies.

    **Supported Materials:** Cotton, Silk, Wool  
    *We are actively working to support more fabric types in future releases.*

    Users are advised not to rely solely on the results for critical decisions.
    """)


# Step 1: Upload Image
st.subheader("Step 1: Upload Your Image")
st.markdown("""
**Supported Materials:** Cotton, Silk, Wool  
*More materials will be supported in future versions.*
""")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if uploaded_file:
    # Detect if a new file is uploaded
    if "last_uploaded_file" in st.session_state and st.session_state["last_uploaded_file"] != uploaded_file.name:
        # Reset all session states related to the previous image
        st.session_state["resized_image"] = None
        st.session_state["cropped_image"] = None
        st.session_state["prediction_done"] = False
        st.session_state["prediction_results"] = None
        st.session_state["prediction_probabilities"] = None
        st.session_state["show_canvas"] = True

    # Save the current file name to session state
    st.session_state["last_uploaded_file"] = uploaded_file.name

    # Process the uploaded file
    image = Image.open(uploaded_file)
    fixed_width = 400  # Reduced width for smaller display
    aspect_ratio = image.height / image.width
    resized_image = image.resize((fixed_width, int(fixed_width * aspect_ratio)))
    st.session_state["resized_image"] = resized_image

    # Step 2: Draw ROI
    if st.session_state["show_canvas"]:
        st.subheader("Step 2: Select Region of Interest (ROI)")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0)",  # Red stroke
            stroke_width=3,
            stroke_color="red",  # Set red color for border
            background_image=ImageOps.exif_transpose(resized_image),
            update_streamlit=True,
            height=resized_image.height,
            width=fixed_width,
            drawing_mode="rect",
            key="canvas",
        )

        # Check for ROI selection issues
        if canvas_result and canvas_result.json_data:
            if len(canvas_result.json_data["objects"]) > 1:
                st.warning("Only one bounding box is allowed. Click 'Undo' to revert action.")
            elif len(canvas_result.json_data["objects"]) == 0:
                st.error("Please select an ROI to proceed.")
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

                    # Reset prediction states after recrop
                    st.session_state["prediction_done"] = False
                    st.session_state["prediction_results"] = None
                    st.session_state["prediction_probabilities"] = None
                    st.experimental_rerun()
        else:
            st.error("Please select an ROI before proceeding.")  # If no objects are drawn


    # Step 3: Cropped Image
    if st.session_state["cropped_image"] and not st.session_state["show_canvas"]:
        st.subheader("Step 3: Crop Results")

        # Display images
        display_height = 250  # Reduced display height for a compact layout
        aspect_ratio_original = st.session_state["resized_image"].width / st.session_state["resized_image"].height
        resized_display_image = st.session_state["resized_image"].resize((int(display_height * aspect_ratio_original), display_height))

        aspect_ratio_cropped = st.session_state["cropped_image"].width / st.session_state["cropped_image"].height
        resized_display_cropped = st.session_state["cropped_image"].resize((int(display_height * aspect_ratio_cropped), display_height))

        # Add padding between columns
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                resized_display_image,
                caption="Original Image",
                use_column_width=False,
                output_format="JPEG",
            )
        with col2:
            st.image(
                resized_display_cropped,
                caption="Cropped ROI",
                use_column_width=False,
                output_format="JPEG",
            )

        # Buttons and Prediction
        if not st.session_state["prediction_done"]:
            col1, col2 = st.columns([1, 1])  # Adjust button spacing
            with col1:
                if st.button("Predict Now"):
                    predicted_label, probabilities = predict_image(st.session_state["cropped_image"])
                    st.session_state["prediction_results"] = labels[predicted_label]
                    st.session_state["prediction_probabilities"] = probabilities
                    st.session_state["prediction_done"] = True
                    st.experimental_rerun()
            with col2:
                if st.button("Recrop"):
                    st.session_state["cropped_image"] = None
                    st.session_state["show_canvas"] = True
                    st.session_state["prediction_done"] = False
                    st.experimental_rerun()
        else:
            # Display Prediction Results
            st.subheader("Prediction Results")
            st.markdown("""
            **Note:** This system currently supports Cotton, Silk, and Wool.  
            Future updates aim to support additional fabric types.
            """)
            ranked_predictions = sorted(
                zip(labels, st.session_state["prediction_probabilities"]),
                key=lambda x: x[1],
                reverse=True
            )
            for rank, (label, prob) in enumerate(ranked_predictions, start=1):
                st.markdown(f"**Rank {rank}: {label} ({prob:.2%})**")
                st.progress(prob)
    
            # Fabric Info Section
            st.subheader("Fabric Details")
            material = st.session_state["prediction_results"]
            fabric_info = {
            "Cotton": {
                "Care Tips": (
                    "- Wash in cold or warm water.\n"
                    "- Use a gentle cycle and mild detergent.\n"
                    "- Air dry or tumble dry on low."
                ),
                "Visual Characteristics": (
                    "- Soft and matte finish.\n"
                    "- Lightweight and breathable.\n"
                    "- Common in everyday clothing."
                ),
                "Sustainability Info": (
                    "- Growing cotton uses a lot of water and chemicals.\n"
                    "- Recycle or reuse cotton items to reduce waste.\n"
                    "- Support eco-friendly brands or practices."
                )
            },
            "Silk": {
                "Care Tips": (
                    "- Hand wash with gentle soap or use a delicate cycle.\n"
                    "- Air dry flat; avoid direct sunlight."
                ),
                "Visual Characteristics": (
                    "- Smooth, shiny, and elegant texture.\n"
                    "- Lightweight and drapes beautifully.\n"
                    "- Often used in formal wear."
                ),
                "Sustainability Info": (
                    "- Traditional silk production harms silkworms.\n"
                    "- Recycle or reuse silk items when possible.\n"
                    "- Consider alternatives like plant-based or synthetic fibers."
                )
            },
            "Wool": {
                "Care Tips": (
                    "- Wash with lukewarm water and wool-safe detergent.\n"
                    "- Gently press out water; do not wring.\n"
                    "- Lay flat to dry."
                ),
                "Visual Characteristics": (
                    "- Thick, soft, and slightly fluffy.\n"
                    "- Warm and insulating.\n"
                    "- Common in winter clothing like sweaters and coats."
                ),
                "Sustainability Info": (
                    "- Wool farming can impact the environment.\n"
                    "- Extend product life with proper care.\n"
                    "- Recycle or donate wool items to reduce waste."
                )
            }
        }
            if material in fabric_info:
                st.markdown("##### Care Tips")
                st.markdown(fabric_info[material]["Care Tips"])
                st.markdown("##### Looks Like")
                st.markdown(fabric_info[material]["Visual Characteristics"])
                st.markdown("##### Sustainability Info")
                st.markdown(fabric_info[material]["Sustainability Info"])

            # After Prediction Buttons
            col1, col2 = st.columns([1, 1])  # Adjust column ratios as needed
            with col1:
                if st.button("Recrop"):
                    st.session_state["cropped_image"] = None
                    st.session_state["show_canvas"] = True
                    st.session_state["prediction_done"] = False
                    st.experimental_rerun()
            with col2:
                if st.button("Start Over"):
                    st.session_state.clear()
                    st.markdown("""<meta http-equiv="refresh" content="0; url=." />""", unsafe_allow_html=True)
