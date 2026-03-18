import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import dlib

from utils.model import OrthodonticModel
from utils.transforms import test_transforms
from utils.tps import warp_face_tps

import os
import gdown

def download_file(file_id, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
# -------------------------
# Download required files
# -------------------------
download_file("1hhtStOe3KoYbEtW2zrNxXSZNo4YwIvz5", "orthodontic_model_v2.pth")
download_file("1PpdGIOc6iU4WPJE-HdgPgKxtYEx1sjqx", "shape_predictor_68_face_landmarks.dat")
# -------------------------
# Load model
# -------------------------
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = OrthodonticModel().to(device)
    model.load_state_dict(
        torch.load("orthodontic_model_v2.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()
scaler = joblib.load("angle_scaler.pkl")

# -------------------------
# dlib
# -------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# -------------------------
# Landmark extraction
# -------------------------
def extract_landmarks(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    shape = predictor(gray, faces[0])

    landmarks = []
    for i in range(68):
        landmarks.append([shape.part(i).x, shape.part(i).y])

    return np.array(landmarks)

# -------------------------
# UI
# -------------------------
st.title("🦷 AI Orthodontic Simulator")

uploaded = st.file_uploader("Upload Image")

naso = st.number_input("Nasolabial Before")
mento = st.number_input("Mentolabial Before")

class_type = st.selectbox("Class", ["Class I","Class II"])
treatment = st.selectbox("Treatment", ["Tads","Conventional anchorage"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image)

    if st.button("Predict"):

        landmarks = extract_landmarks(img)

        if landmarks is None:
            st.error("Face not detected")
        else:

            h,w = img.shape[:2]

            landmarks_norm = landmarks.astype(np.float32)
            landmarks_norm[:,0] /= w
            landmarks_norm[:,1] /= h

            img_tensor = test_transforms(image).unsqueeze(0)

            tabular = scaler.transform(pd.DataFrame(
                [[naso, mento]],
                columns=["nasolabial_before","mentolabial_before"]
            ))
            tabular = torch.tensor(tabular).float()

            class_id = torch.tensor([0 if class_type=="Class I" else 1])
            treatment_id = torch.tensor([0 if treatment=="Tads" else 1])

            with torch.no_grad():
                angle_pred, landmark_pred = model(
                    img_tensor,
                    tabular,
                    class_id,
                    treatment_id
                )

                landmark_pred = torch.tanh(landmark_pred) * 0.02

            landmark_pred = landmark_pred.numpy().reshape(68,2)

            dst = landmarks_norm + landmark_pred
            src = landmarks_norm.copy()

            src[:,0] *= w
            src[:,1] *= h
            dst[:,0] *= w
            dst[:,1] *= h

            warped = warp_face_tps(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                src,
                dst
            )

            st.image(warped)

            st.success(f"""
Nasolabial Change: {angle_pred[0][0]:.2f}°
Mentolabial Change: {angle_pred[0][1]:.2f}°
""")
