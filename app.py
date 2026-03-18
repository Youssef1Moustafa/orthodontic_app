import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import face_recognition  # بدلاً من dlib مباشرة

from utils.model import OrthodonticModel
from utils.transforms import test_transforms
from utils.tps import warp_face_tps

import os
import gdown

# دالة التحميل داخل كاش
@st.cache_resource
def setup_files():
    files = {
        "orthodontic_model_v2.pth": "1hhtStOe3KoYbEtW2zrNxXSZNo4YwIvz5",
        "shape_predictor_68_face_landmarks.dat": "1PpdGIOc6iU4WPJE-HdgPgKxtYEx1sjqx"
    }
    
    for name, file_id in files.items():
        if not os.path.exists(name):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, name, quiet=False)
    
    # تحميل الـ Scaler
    scaler = joblib.load("angle_scaler.pkl")
    
    return scaler

# تنفيذ التحميل والتحضير
try:
    scaler = setup_files()
except Exception as e:
    st.error(f"Error initializing files: {e}")
    st.stop()

# تكملة كود تحميل الموديل
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = OrthodonticModel().to(device)
    if os.path.exists("orthodontic_model_v2.pth"):
        model.load_state_dict(
            torch.load("orthodontic_model_v2.pth", map_location=device)
        )
    model.eval()
    return model

model = load_model()

# -------------------------
# Landmark extraction باستخدام face_recognition
# -------------------------
def extract_landmarks(image):
    """استخراج landmarks باستخدام face_recognition"""
    try:
        # face_recognition يتوقع RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # استخراج landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        
        if not face_landmarks_list:
            return None
        
        # تحويل الـ landmarks إلى الشكل المطلوب (68 نقطة)
        # face_recognition يعطي 68 نقطة لكن بشكل مختلف
        landmarks = []
        
        # استخراج النقاط بالترتيب الصحيح
        for face_landmarks in face_landmarks_list:
            # هذا مبسط - قد تحتاج لترتيب النقاط حسب الترتيب المطلوب
            for part in ['chin', 'left_eyebrow', 'right_eyebrow', 
                        'nose_bridge', 'nose_tip', 'left_eye', 
                        'right_eye', 'top_lip', 'bottom_lip']:
                for point in face_landmarks[part]:
                    landmarks.append([point[0], point[1]])
        
        return np.array(landmarks[:68])  # نأخذ أول 68 نقطة فقط
        
    except Exception as e:
        st.error(f"Error in landmark extraction: {e}")
        return None

# -------------------------
# UI
# -------------------------
st.title("🦷 AI Orthodontic Simulator")

uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

col1, col2 = st.columns(2)
with col1:
    naso = st.number_input("Nasolabial Before", value=0.0)
with col2:
    mento = st.number_input("Mentolabial Before", value=0.0)

class_type = st.selectbox("Class", ["Class I", "Class II"])
treatment = st.selectbox("Treatment", ["Tads", "Conventional anchorage"])

if uploaded:
    # قراءة الصورة
    image = Image.open(uploaded).convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        with st.spinner("Processing..."):
            # استخراج landmarks
            landmarks = extract_landmarks(img)
            
            if landmarks is None:
                st.error("Face not detected. Please try another image.")
            else:
                # تحضير البيانات
                h, w = img.shape[:2]
                
                landmarks_norm = landmarks.astype(np.float32)
                landmarks_norm[:, 0] /= w
                landmarks_norm[:, 1] /= h
                
                # تحضير tensor الصورة
                img_tensor = test_transforms(image).unsqueeze(0)
                
                # تحضير tabular data
                tabular = scaler.transform(pd.DataFrame(
                    [[naso, mento]],
                    columns=["nasolabial_before", "mentolabial_before"]
                ))
                tabular = torch.tensor(tabular).float()
                
                class_id = torch.tensor([0 if class_type == "Class I" else 1])
                treatment_id = torch.tensor([0 if treatment == "Tads" else 1])
                
                # التنبؤ
                with torch.no_grad():
                    angle_pred, landmark_pred = model(
                        img_tensor,
                        tabular,
                        class_id,
                        treatment_id
                    )
                    
                    landmark_pred = torch.tanh(landmark_pred) * 0.02
                
                # تحويل التنبؤات
                landmark_pred = landmark_pred.numpy().reshape(68, 2)
                dst = landmarks_norm + landmark_pred
                src = landmarks_norm.copy()
                
                src[:, 0] *= w
                src[:, 1] *= h
                dst[:, 0] *= w
                dst[:, 1] *= h
                
                # تطبيق TPS warp
                warped = warp_face_tps(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    src,
                    dst
                )
                
                # عرض النتائج
                col3, col4 = st.columns(2)
                with col3:
                    st.image(image, caption="Before", use_container_width=True)
                with col4:
                    st.image(warped, caption="After Prediction", use_container_width=True)
                
                # عرض النتائج الرقمية
                st.success(f"""
                **Predicted Changes:**
                - Nasolabial Change: {angle_pred[0][0]:.2f}°
                - Mentolabial Change: {angle_pred[0][1]:.2f}°
                """)
