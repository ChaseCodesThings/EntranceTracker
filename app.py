import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np

# --- CONFIGURATION ---
PAGE_TITLE = "Occupancy Monitor (Cloud Version)"
CONFIDENCE_THRESHOLD = 0.5 
MIN_BOX_AREA = 5000 

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- 1. LOAD MODEL ONCE ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# --- 2. THE VIDEO PROCESSOR CLASS ---
# This runs on the cloud server for every single frame of video
class OccupancyProcessor:
    def __init__(self):
        # Initialize Trackers & Tools
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5, 
            lost_track_buffer=60, 
            minimum_matching_threshold=0.8, 
            frame_rate=30
        )
        self.box_an = sv.BoxAnnotator(thickness=3)
        self.label_an = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)
        
        # State Management (Inside the Class)
        self.tracker_state = {}     # {id: "OUTSIDE" | "INSIDE"}
        self.tracker_origin = {}    # {id: "OUTSIDE" | "INSIDE"}
        self.count_in = 0
        self.count_out = 0
        
        # Sliders are hard to pass into this thread, so we set sensible defaults
        # or you can use st.session_state if configured correctly, but defaults are safer for Cloud.
        self.line_door_pct = 0.40
        self.line_room_pct = 0.85

    def recv(self, frame):
        # 1. Convert WebRTC frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # 2. Define Lines
        door_y = int(h * self.line_door_pct)
        room_y = int(h * self.line_room_pct)
        
        # 3. Draw Lines & Overlay Text
        cv2.line(img, (0, door_y), (w, door_y), (255, 255, 255), 3)
        cv2.line(img, (0, room_y), (w, room_y), (0, 255, 0), 3)
        
        # 4. Run AI
        results = model(img, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        dets = sv.Detections.from_ultralytics(results)
        
        # Filter Small Boxes
        valid_indices = []
        for i, bbox in enumerate(dets.xyxy):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > MIN_BOX_AREA:
                valid_indices.append(i)
        
        if len(valid_indices) > 0:
            dets = dets[valid_indices]
        else:
            dets = sv.Detections.empty()

        # Update Tracker
        dets = self.tracker.update_with_detections(dets)
        
        # 5. Logic Loop
        current_states = {}
        for i, (tid, cid) in enumerate(zip(dets.tracker_id, dets.class_id)):
            if cid != 0: continue # Humans Only
            
            # Get Feet
            x1, y1, x2, y2 = dets.xyxy[i]
            feet_y = int(y2)
            
            # Determine Zone
            if feet_y < door_y:
                curr_pos = "A" # Outside
            elif feet_y < room_y:
                curr_pos = "B" # Buffer
            else:
                curr_pos = "C" # Inside
            
            current_states[tid] = curr_pos
            
            # Origin Logic
            if tid not in self.tracker_origin:
                self.tracker_origin[tid] = curr_pos
            
            origin = self.tracker_origin[tid]
            prev_pos = self.tracker_state.get(tid)
            
            if prev_pos:
                # Entry (A/B -> C)
                if (prev_pos == "A" or prev_pos == "B") and curr_pos == "C":
                    if origin != "C": # Strict Origin Check
                        self.count_in += 1
                        self.tracker_state[tid] = "C" # Prevent double count
                
                # Exit (C/B -> A)
                if (prev_pos == "C" or prev_pos == "B") and curr_pos == "A":
                    if origin != "A": # Strict Origin Check
                        self.count_out += 1
                        self.tracker_state[tid] = "A"
                        
        self.tracker_state.update(current_states)
        
        # 6. Draw Boxes
        labels = []
        for tid, cid in zip(dets.tracker_id, dets.class_id):
            name = model.names[cid]
            labels.append(f"#{tid} {name}" if cid == 0 else name)

        img = self.box_an.annotate(img, dets)
        img = self.label_an.annotate(img, dets, labels=labels)
        
        # 7. Draw HUD (Heads Up Display) for Metrics
        # Since we can't update the sidebar easily from here, we draw on the image
        cv2.rectangle(img, (10, 10), (250, 90), (0, 0, 0), -1) # Background box
        cv2.putText(img, f"ENTERED: {self.count_in}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"EXITED:  {self.count_out}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 8. Return frame to browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. UI LAYOUT ---
st.title("☁️ Cloud Occupancy Monitor")
st.info("Ensure you allow camera access in your browser.")

# Webrtc Streamer is the magic component that works on the web
webrtc_streamer(
    key="occupancy-tracker",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=OccupancyProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.divider()
st.write("Note: On Cloud deployments, line calibration is set to defaults (40% and 85%) to ensure stability.")
