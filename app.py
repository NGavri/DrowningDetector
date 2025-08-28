import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load saved model
model = joblib.load("drowning_detector.pkl")

# Categories
categories = ['drown', 'not_drown']

# Image size (must match training size)
IMG_SIZE = (64, 64)

# Streamlit UI
st.set_page_config(page_title="Drowning Detection AI", page_icon="🌊")
st.title("🌊 Drowning Detection AI")

if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("About"):
    st.session_state.page = "About"
    
if st.session_state.page == "Home":
    st.write("Upload an image and the AI will detect if the person is drowning or not.")

    # File uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img_array = np.array(image)

        try:
            img_array = cv2.resize(img_array, IMG_SIZE)  # resize
        except:
            st.error("Error processing image. Please try another one.")
            st.stop()

        img_array = img_array / 255.0  # normalize
        img_flat = img_array.reshape(1, -1)  # flatten for logistic regression

        # Prediction
        prediction = model.predict(img_flat)[0]
        probas = model.predict_proba(img_flat)[0]

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {categories[prediction].upper()}")

        st.write("**Confidence:**")
        st.write({categories[i]: f"{probas[i]*100:.2f}%" for i in range(len(categories))})

elif st.session_state.page == "About":
    st.subheader("About Drowning Detection AI")
    st.markdown("""
    Drowning Detection AI is a machine learning web app that predicts whether a person is drowning in an image. Using a Logistic Regression model trained on image data, it provides real-time predictions with confidence scores in an easy-to-use web interface.

    Whether you're a student, researcher, or developer, Drowning Detection AI demonstrates how AI can be applied to safety and computer vision in a simple and interactive way.

    This version is just the beginning. I’m actively improving the model, refining predictions, and exploring ways to make AI-powered safety tools more accessible.

    ---
    **Curious about AI beyond the usual?**  
    I also run a blog — [**Synapse and Steel**](https://synapseandsteel.wordpress.com/?_gl=1*1wk53el*_gcl_au*MTUyNDU2NzIzNy4xNzUxMTE3NzEx). It's part journal, part tech talk, and part “oops, I did that.”
    Feel free to explore and maybe chuckle once or twice.


    """)
