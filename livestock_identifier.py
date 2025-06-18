import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
import uuid
import datetime
import cv2
import pytesseract
from ultralytics import YOLO
from streamlit_geolocation import streamlit_geolocation
import qrcode
from io import BytesIO
from fpdf import FPDF

# Setup
st.set_page_config(page_title="Livestock ID", layout="centered")
os.makedirs("records/images", exist_ok=True)
os.makedirs("pdf_profiles", exist_ok=True)
DATA_LOG = "records/data.csv"
model = YOLO("yolov8n.pt")

def run_yolo_model(pil_img):
    results = model(pil_img)
    labels, boxes = [], []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            label = model.names[int(cls)]
            labels.append(label)
            boxes.append(box.tolist())
    return labels, boxes

def extract_text_from_roi(pil_img, box):
    img = np.array(pil_img.convert("RGB"))
    x1, y1, x2, y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 7')
    return text.strip()

def generate_qr_code(data: str):
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer)
    buffer.seek(0)
    return buffer

def save_record(image, metadata):
    animal_id = str(uuid.uuid4())
    metadata["Animal ID"] = animal_id
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"records/images/{animal_id}.png"
    image.save(filename)
    metadata.update({
        "Timestamp": timestamp,
        "Image": filename,
        "Status": "Active"
    })
    df = pd.DataFrame([metadata])
    if os.path.exists(DATA_LOG):
        df.to_csv(DATA_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_LOG, index=False)
    return metadata

def update_database(df):
    df.to_csv(DATA_LOG, index=False)

query_params = st.experimental_get_query_params()
if "animal_id" in query_params:
    animal_id = query_params["animal_id"][0]
    if os.path.exists(DATA_LOG):
        df = pd.read_csv(DATA_LOG)
        animal = df[df["Animal ID"] == animal_id]
        if not animal.empty:
            row = animal.iloc[0]
            st.title(f"🐄 Livestock Profile: {row['Tag ID']}")
            st.image(row["Image"], width=300)
            for field in ["Breed", "Owner", "Farm ID", "Sex", "DOB", "Sire", "Dam", "Status",
                          "Vaccination Records", "Medical Notes", "Description", "Latitude", "Longitude"]:
                st.markdown(f"**{field.replace('_', ' ')}:** {row.get(field, '')}")
            st.stop()
        else:
            st.error("Animal not found.")
            st.stop()

st.title("🐄 Livestock Identification System")

img = None
mode = st.radio("📷 Choose Input", ["Take Photo", "Upload Image"], horizontal=True)
if mode == "Take Photo":
    picture = st.camera_input("Capture Livestock")
    if picture: img = Image.open(picture)
elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded: img = Image.open(uploaded)

if img:
    st.image(img, use_column_width=True)
    labels, boxes = run_yolo_model(img)
    detected_breed = next((l for l in labels if l.lower() in ["nguni", "boran", "brahman", "holstein", "sheep"]), "Unknown")
    tag_id = "Not Detected"
    for label, box in zip(labels, boxes):
        if "tag" in label.lower():
            tag_id = extract_text_from_roi(img, box)
            break

    st.subheader("📋 Animal Details")
    with st.form("animal_form"):
        farm_id = st.text_input("Farm ID")
        owner = st.text_input("Owner")
        sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
        dob = st.date_input("Date of Birth")
        sire = st.text_input("Sire")
        dam = st.text_input("Dam")
        desc = st.text_area("Description")
        vaccines = st.text_area("Vaccination Records")
        medical = st.text_area("Medical Notes")
        location = geolocation()
        lat, lon = "", ""
        if location:
            lat, lon = location.get("latitude", ""), location.get("longitude", "")
            st.markdown(f"📍 Location: **{lat}, {lon}**")
        submit = st.form_submit_button("✅ Save Record")
    if submit:
        metadata = {
            "Breed": detected_breed,
            "Tag ID": tag_id,
            "Farm ID": farm_id,
            "Owner": owner,
            "Sex": sex,
            "DOB": dob,
            "Sire": sire,
            "Dam": dam,
            "Latitude": lat,
            "Longitude": lon,
            "Description": desc,
            "Vaccination Records": vaccines,
            "Medical Notes": medical
        }
        save_record(img, metadata)
        st.success(f"Saved record for {tag_id}")

if os.path.exists(DATA_LOG):
    df = pd.read_csv(DATA_LOG)
    search_term = st.text_input("🔍 Search records").strip().lower()
    filtered_df = df[df.apply(lambda row: search_term in str(row.values).lower(), axis=1)] if search_term else df

    for i, row in filtered_df.iterrows():
        with st.expander(f"{row['Tag ID']} - {row['Breed']} [{row['Status']}]"):
            st.image(row["Image"], width=250)
            base_url = st.secrets.get("base_url", "https://your-app-name.streamlit.app")
            animal_url = f"{base_url}?animal_id={row['Animal ID']}"
            st.markdown(f"[🔗 View Profile]({animal_url})")
            qr_img = generate_qr_code(animal_url)
            st.image(qr_img, caption="QR Code", width=100)
            with st.form(f"edit_{i}"):
                farm_id = st.text_input("Farm ID", value=row["Farm ID"])
                owner = st.text_input("Owner", value=row["Owner"])
                sex = st.selectbox("Sex", ["Male", "Female", "Unknown"], index=["Male", "Female", "Unknown"].index(row["Sex"]))
                dob = st.text_input("DOB", value=row["DOB"])
                sire = st.text_input("Sire", value=row["Sire"])
                dam = st.text_input("Dam", value=row["Dam"])
                desc = st.text_area("Description", value=row.get("Description", ""))
                vaccines = st.text_area("Vaccination Records", value=row.get("Vaccination Records", ""))
                medical = st.text_area("Medical Notes", value=row.get("Medical Notes", ""))
                lat = st.text_input("Latitude", value=str(row.get("Latitude", "")))
                lon = st.text_input("Longitude", value=str(row.get("Longitude", "")))
                status = st.selectbox("Status", ["Active", "Sold"], index=["Active", "Sold"].index(row["Status"]))
                col1, col2, col3 = st.columns(3)
                if col1.form_submit_button("💾 Update"):
                    df.at[i, "Farm ID"] = farm_id
                    df.at[i, "Owner"] = owner
                    df.at[i, "Sex"] = sex
                    df.at[i, "DOB"] = dob
                    df.at[i, "Sire"] = sire
                    df.at[i, "Dam"] = dam
                    df.at[i, "Latitude"] = lat
                    df.at[i, "Longitude"] = lon
                    df.at[i, "Status"] = status
                    df.at[i, "Description"] = desc
                    df.at[i, "Vaccination Records"] = vaccines
                    df.at[i, "Medical Notes"] = medical
                    update_database(df)
                    st.success("Record updated.")
                    st.experimental_rerun()
                if col2.form_submit_button("❌ Delete"):
                    df.drop(index=i, inplace=True)
                    update_database(df)
                    st.warning("Record deleted.")
                    st.experimental_rerun()
