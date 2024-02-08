import os
import sys
import io
import cv2
import time
import smtplib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
from email import encoders
from ultralytics import YOLO
from email.mime.text import MIMEText
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

# For Getting the absolute path of the current file
file_path = Path(__file__).resolve()

# For Getting the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())
DETECTION_MODEL = ROOT / 'best.pt'
model_path = Path(DETECTION_MODEL)

sender_email = "puyushgupta786@gmail.com"
receiver_email = "puyush.work@gmail.com"
sender_password = "ttzxwnrgzbxckndi"
smtp_port = 587
smtp_server = "smtp.gmail.com"
subject = "Accident detected"

def send_email(accident_type,image):
    body = accident_type

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    filename = "res.png"
    is_success, buffer = cv2.imencode(".jpg", image)
    attachment = buffer.tobytes()

    attachment_package = MIMEBase('application', 'octet-stream')
    attachment_package.set_payload(attachment)
    encoders.encode_base64(attachment_package)
    attachment_package.add_header('Content-Disposition', "attachment; filename=" + filename)
    msg.attach(attachment_package)

    text = msg.as_string()

    print("Connecting to server")
    gmail_server = smtplib.SMTP(smtp_server, smtp_port)
    gmail_server.starttls()
    gmail_server.login(sender_email, sender_password)
    print("Successfully Connected to Server")

    print("Sending email to ", receiver_email)
    gmail_server.sendmail(sender_email, receiver_email, text)
    print("Email sent to ", receiver_email)

    gmail_server.quit()


def check_acc(box):
    res_index_list = box.cls.tolist()
    result = ""

    for index in res_index_list:
        if index == 1:
            result = "Bike Bike Accident Detected"
            break
        elif index == 2:
            result = "Bike Object Accident Detected"
            break
        elif index == 3:
            result = "Bike Person Accident Detected"
            break
        elif index == 5:
            result = "Car Bike Accident Detected"
            break
        elif index == 6:
            result = "Car Car Accident Detected"
            break
        elif index == 7:
            result = "Car Object Accident Detected"
            break
        elif index == 8:
            result = "Car Person Accident Detected"
            break

    return result


st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(
    page_title="Accident Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Real Time Accident Detection')

# Sidebar
st.sidebar.header("Model Configuration")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

iou = float(st.sidebar.slider(
    "Select IOU", 25, 100, 40)) / 100

@st.cache_data
def load_model():
    model = YOLO(model_path)
    return model

model = load_model()

@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img

st.sidebar.header("Image/Video Options")
source_radio = st.sidebar.radio("Select Source", ['Image', 'Video'])
source_img = None
if source_radio == 'Image':
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is not None:
                st.image(load_image(source_img), caption="Uploaded Image",use_column_width=True, width=150)

        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if(source_img is not None):
            image = Image.open(io.BytesIO(source_img.getvalue()))
            results = model.predict(source=image, conf=confidence, iou=iou, imgsz=512)
            box = results[0].boxes
            res = check_acc(box)

            annotated_frame = results[0].plot()
            annotated_frame= cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_frame, caption='Detected Image',use_column_width=True)
            if res:
                st.write(res)
                send_email(res, annotated_frame)
            else:
                st.write("No Accident Detected")


elif source_radio == 'Video':
    source_video = st.sidebar.file_uploader("Choose an video...", type=["mp4"])
    temp_loc = "test.mp4"
    if source_video is not None:
        count = 0
        g = io.BytesIO(source_video.read())
        with open(temp_loc, 'wb') as out:
            out.write(g.read())
        out.close()

    col1, col2 = st.columns(2)

    with col1:
        if source_video is not None:
            video_file = open(temp_loc, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

    with col2:
        if source_video is not None:
            vidcap = cv2.VideoCapture(temp_loc)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            nof = 2
            frame_no = 0
            image_holder = st.empty()

            while vidcap.isOpened():
                res = ""
                success, image = vidcap.read()
                if success == False:
                    break

                # Check if it's time to process the frame based on the desired interval
                if (frame_no % (int(fps / nof))) == 0:
                    results = model.predict(image,conf = confidence,iou = iou,imgsz = 512)
                    box = results[0].boxes
                    res = check_acc(box)

                    annotated_frame = results[0].plot()
                    annotated_frame= cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    image_holder.image(annotated_frame, caption='Detected Image',use_column_width=True)

                    # Display the annotated frame only if an accident is detected
                    if len(res) > 0:
                        if count == 0:
                            send_email(res, annotated_frame)
                        count+=1

                frame_no += 1  

            # Release the video capture object
            vidcap.release()


else:
    st.error("Please select a valid source type!")