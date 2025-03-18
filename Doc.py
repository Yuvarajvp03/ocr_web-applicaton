import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from io import BytesIO
import base64
import mysql.connector
from fpdf import FPDF
import time

# Set Tesseract Path (Modify if needed)
# pytesseract.pytesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="Yuvaraj@2423",
        database="ocr_db"
    )

def fetch_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

def register_user(username, password, role):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
    if cursor.fetchone()[0] > 0:
        st.warning("⚠️ Username already exists! Please choose another one.")
    else:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", (username, password, role))
        conn.commit()
        st.success("✅ Registration successful! Redirecting to login...")  
        time.sleep(2)  # Wait for 2 seconds to show the message
        st.rerun()  # Refresh page after success
    conn.close()

def save_document(username, filename, text, status, category):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (username, filename, text, status, category) VALUES (%s, %s, %s, %s, %s)",
                   (username, filename, text, status, category))
    conn.commit()
    conn.close()

def save_file(text, filename, format):
    """Save extracted text to a file and return downloadable link"""
    buffer = BytesIO()
    if format == "TXT":
        buffer.write(text.encode())
    elif format == "PDF":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(buffer, dest='S')
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("📝 OCR Web App Login Page")

# Authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

# Registration
if not st.session_state.logged_in:
    auth_option = st.radio("Choose an option", ["Login", "Register"])
    
    if auth_option == "Register":
        st.subheader("📝 Create an Account")
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        role = st.selectbox("Select Role", ["user", "admin"])
        if st.button("Register"):
            register_user(new_username, new_password, role)
    
    elif auth_option == "Login":
        st.subheader("🔑 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = fetch_users()
            user_data = next((u for u in users if u["username"] == username and u["password"] == password), None)
            if user_data:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = user_data["role"]
                st.rerun()
            else:
                st.error("Invalid credentials!")
else:
    if st.session_state.role == "user":
        st.title("User Page")
        st.subheader("📂 Upload an Image for OCR")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        lang = st.selectbox("Select OCR Language", ["eng", "hin", "tam", "fra", "spa"], index=0)
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            text = pytesseract.image_to_string(gray, lang=lang)
            st.subheader("Extracted Text:")
            st.text_area("Text Output", text, height=200)
            file_format = st.radio("Save as:", ["TXT", "PDF"])
            buffer = save_file(text, uploaded_file.name, file_format)
            st.download_button("Download", buffer, file_name=f"{uploaded_file.name}.{file_format.lower()}")
            save_document(st.session_state.username, uploaded_file.name, text, "Pending", "General")
    
    elif st.session_state.role == "admin":
        st.title(" Admin Page")
        st.subheader("📊 Admin Dashboard - Uploaded Documents")
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM documents", conn)
        conn.close()
        if df.empty:
            st.info("No documents uploaded yet!")
        else:
            st.dataframe(df)
            doc_id = st.selectbox("Select Document to Approve/Reject", df["id"])
            new_status = st.radio("Update Status:", ["Approved", "Rejected"])
            if st.button("Update Status"):
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE documents SET status=%s WHERE id=%s", (new_status, doc_id))
                conn.commit()
                conn.close()
                st.success("Status Updated!")
    
    st.sidebar.button("Logout", on_click=lambda: [st.session_state.update({"logged_in": False, "username": None, "role": None}), st.rerun()])
