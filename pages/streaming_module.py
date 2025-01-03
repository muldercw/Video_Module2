import streamlit as st
import cv2
import os
import numpy as np
import threading
import time
import subprocess
from collections import deque
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.user import User
from clarifai.client.model import Model
from clarifai.client.app import App
from clarifai.modules.css import ClarifaiStreamlitCSS
from google.protobuf import json_format

def list_models():
    app_obj = App(user_id=userDataObject.user_id, app_id=userDataObject.app_id)
    all_models = list(app_obj.list_models())
    usermodels = []
    for model in all_models:
        model_url = f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/models/{model.id}"
        _umod = {"Name": model.id, "URL": model_url, "type": "User"}
        usermodels.append(_umod)
    return list_community_models() + usermodels

def list_community_models():
    return [
        {"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection", "type": "Community"},
        {"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection", "type": "Community"},
        {"Name": "Weapon Detection", "URL": "https://clarifai.com/clarifai/main/models/weapon-detection", "type": "Community"},
        {"Name": "Vehicle Detection", "URL": "https://clarifai.com/clarifai/Roundabout-Aerial-Images-for-Vehicles-Det-Kaggle/models/vehicle-detector-alpha-x", "type": "Community"},
    ]

def draw_box_corners(frame, left, top, right, bottom, color, thickness=1, corner_length=15):
    cv2.line(frame, (left, top), (left + corner_length, top), color, thickness)  # Top-left horizontal
    cv2.line(frame, (left, top), (left, top + corner_length), color, thickness)  # Top-left vertical
    cv2.line(frame, (right, top), (right - corner_length, top), color, thickness)  # Top-right horizontal
    cv2.line(frame, (right, top), (right, top + corner_length), color, thickness)  # Top-right vertical
    cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, thickness)  # Bottom-left horizontal
    cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, thickness)  # Bottom-left vertical
    cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, thickness)  # Bottom-right horizontal
    cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, thickness)  # Bottom-right vertical

def run_model_inference(det_threshold, frame, model_option, color=(0, 255, 0)):
    try:
        _frame = frame.copy()
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        model_url = model_option['URL']
        detector_model = Model(url=model_url)

        cv2.putText(_frame, model_option['Name'], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        prediction_response = detector_model.predict_by_bytes(frame_bytes, input_type="image")
        regions = prediction_response.outputs[0].data.regions

        for region in regions:
            top_row = round(region.region_info.bounding_box.top_row, 3)
            left_col = round(region.region_info.bounding_box.left_col, 3)
            bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
            right_col = round(region.region_info.bounding_box.right_col, 3)

            left = int(left_col * frame.shape[1])
            top = int(top_row * frame.shape[0])
            right = int(right_col * frame.shape[1])
            bottom = int(bottom_row * frame.shape[0])

            draw_box_corners(_frame, left, top, right, bottom, color)

            for concept in region.data.concepts:
                name = concept.name
                value = round(concept.value, 4)
                if value >= det_threshold:
                    text_position = (left + (right - left) // 4, top - 10)
                    cv2.putText(_frame, f"{name}:{value}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        return _frame, prediction_response
    except Exception as e:
        st.error(str(e))
        return frame, None

def verify_json_responses():
    if st.checkbox("Show JSON Results", value=False):
        st.subheader("Model Predictions (JSON Responses)")
        for idx, response in enumerate(json_responses):
            st.json(response)

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

st.title("Video Processing & Monitoring")
json_responses = []

video_option = st.radio("Choose Video Input:", ("Standard Video File URLs","BetaOption"), horizontal=True)

if video_option == "Standard Video File URLs":
    video_urls = st.text_area("Enter video URLs (one per line):", value="http://example.com/sample.mp4")
    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=120, value=30)
    det_threshold = st.slider("Select detection threshold:", min_value=0.01, max_value=1.00, value=0.5)

    available_models = list_models()
    url_list = [url.strip() for url in video_urls.split('\n') if url.strip()]
    model_options = [
        next(model for model in available_models if model["Name"] == st.selectbox(f"Select a model for Video {idx + 1}:", [model["Name"] for model in available_models], key=f"model_{idx}"))
        for idx, url in enumerate(url_list)
    ]

    stop_event = threading.Event()

    if st.button("Stop Processing"):
        stop_event.set()

    if st.button("Process Videos") and not stop_event.is_set():
        frame_placeholder = st.empty()
        try:
            video_buffers = [deque(maxlen=6) for _ in range(len(url_list))]
            threads = []

            def process_video(video_url, index, model_option, stop_event):
                video_capture = cv2.VideoCapture(video_url)
                if not video_capture.isOpened():
                    st.error(f"Error: Could not open video at {video_url}.")
                    return

                frame_count = 0
                while video_capture.isOpened() and not stop_event.is_set():
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    if frame_count % frame_skip == 0:
                        processed_frame, prediction_response = run_model_inference(det_threshold, frame, model_option)
                        if prediction_response:
                            json_responses.append(json_format.MessageToJson(prediction_response))
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        video_buffers[index].append(rgb_frame)

                    frame_count += 1

                video_capture.release()

            for index, (video_url, model_option) in enumerate(zip(url_list, model_options)):
                thread = threading.Thread(target=process_video, args=(video_url, index, model_option, stop_event))
                thread.start()
                threads.append(thread)

            while any(thread.is_alive() for thread in threads):
                grid_frames = [video_buffers[index][-1] for index in range(len(video_buffers)) if len(video_buffers[index]) > 0]
                if grid_frames:
                    grid_image = np.concatenate(grid_frames, axis=1) if len(grid_frames) == 1 else np.vstack(grid_frames)
                    frame_placeholder.image(grid_image, caption="Processed Video Frames")

                time.sleep(0.01)

            for thread in threads:
                thread.join()

        except Exception as e:
            st.error(e)
            json_responses.append(f"Error: {e}")

        verify_json_responses()
