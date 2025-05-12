import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os


USERS = {
    "core_user": {"password": "core123", "role": "core"},
    "admin_user": {"password": "admin123", "role": "admin"},
}

# Session state to keep track of user login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None

# Login page
if not st.session_state.authenticated:
    st.title("Login to Construction Site Monitor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state.authenticated = True
            st.session_state.role = user["role"]
            st.success(f"Logged in as {username} ({user['role']})")
        else:
            st.error("Invalid username or password")
    st.stop()

# Logout button
if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.role = None
    st.experimental_rerun()

# Load helmet detection model
model = YOLO('best.pt')  # Replace with helmet-trained model

st.title("CONSTRUCTION SITE MONITORING")

# Upload image feature (for core and admin)
if st.session_state.role in ["core", "admin"]:
    st.markdown("Upload an image to detect safety helmet compliance.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)

        with st.spinner('Running detection...'):
            results = model.predict(source=img_array, save=False)
            result_img = results[0].plot()  # Render bounding boxes

        st.image(result_img, caption='Detection Result', use_column_width=True)
        st.success("Detection complete!")

# Admin-only section
if st.session_state.role == "admin":
    st.markdown("---")
    st.subheader("🛠️ Admin Panel")
    st.write("You can manage users, delete images, or view analytics here.")
    st.info("(Admin features to be implemented in next phase)")
