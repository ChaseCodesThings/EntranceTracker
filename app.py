import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import csv
import time

# --- CONFIGURATION ---
PAGE_TITLE = "Occupancy Monitor (Secure)"
CSV_FILE = "attendance_log.csv"

# 1. COOLDOWN: Prevent double-counting the same person (3 seconds)
COOLDOWN_SECONDS = 3.0
# 2. CONFIDENCE: Ignore objects unless AI is 50% sure it's a person
CONFIDENCE_THRESHOLD = 0.5
# 3. MINIMUM SIZE: Ignore tiny "ghost" boxes (width * height)
MIN_BOX_AREA = 5000

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- 1. STATE MANAGEMENT ---
if 'cam_index' not in st.session_state:
    st.session_state['cam_index'] = 0
if 'count_in' not in st.session_state:
    st.session_state['count_in'] = 0
if 'count_out' not in st.session_state:
    st.session_state['count_out'] = 0
if 'recent_events' not in st.session_state:
    st.session_state['recent_events'] = []

# TRACKER STATE
if 'tracker_state' not in st.session_state:
    st.session_state['tracker_state'] = {}
if 'tracker_origin' not in st.session_state:
    st.session_state['tracker_origin'] = {}

# TIMERS
if 'last_entry_time' not in st.session_state:
    st.session_state['last_entry_time'] = {}
if 'last_exit_time' not in st.session_state:
    st.session_state['last_exit_time'] = {}


def cycle_camera():
    st.session_state['cam_index'] = (st.session_state['cam_index'] + 1) % 3


# --- 2. CORE FUNCTIONS ---

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')


def get_feet_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def log_event(event_type, tracker_id):
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Event", "ID"])
        writer.writerow([date_str, time_str, event_type, tracker_id])

    return {"Time": time_str, "Event": event_type, "ID": f"#{tracker_id}"}


# --- 3. UI LAYOUT ---

st.title(PAGE_TITLE)

with st.sidebar:
    st.header("Control Panel")

    st.subheader("Video Source")
    st.write(f"Index: {st.session_state['cam_index']}")
    st.button("Switch Camera", on_click=cycle_camera, use_container_width=True)

    st.divider()

    st.subheader("Calibration")
    st.caption("Adjust lines to floor perspective.")
    line_door_pct = st.slider("Door Line (White)", 0.0, 1.0, 0.50, 0.01)
    line_room_pct = st.slider("Room Line (Green)", 0.0, 1.0, 0.85, 0.01)

    st.divider()

    st.subheader("System Status")
    run_ai = st.toggle("Active Monitoring", value=False)

    st.divider()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("Export Log", f, file_name="attendance_log.csv", use_container_width=True)

col_video, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("Real-Time Metrics")
    m1, m2 = st.columns(2)
    in_metric = m1.empty()
    out_metric = m2.empty()
    in_metric.metric("Total Entered", st.session_state['count_in'])
    out_metric.metric("Total Exited", st.session_state['count_out'])

    st.divider()
    st.subheader("Activity Log")
    log_table = st.empty()
    if st.session_state['recent_events']:
        df = pd.DataFrame(st.session_state['recent_events'][:8])
        log_table.dataframe(df, hide_index=True, use_container_width=True)

with col_video:
    video_placeholder = st.empty()


# --- 4. MAIN LOOP ---

def main():
    # Attempt to open camera
    cap = cv2.VideoCapture(st.session_state['cam_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        video_placeholder.error(f"Error: Video Source {st.session_state['cam_index']} unavailable.")
        return

    model = load_model()

    # SETUP TRACKER
    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )

    box_an = sv.BoxAnnotator(thickness=3)
    label_an = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)

    fail_count = 0

    while True:
        ret, frame = cap.read()

        # --- AUTO-RECONNECT LOGIC ---
        if not ret:
            fail_count += 1
            if fail_count > 5:
                video_placeholder.error("Video stream lost. Attempting to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(st.session_state['cam_index'])
                fail_count = 0
            continue

        fail_count = 0  # Reset fail count on success

        h, w = frame.shape[:2]
        door_y = int(h * line_door_pct)
        room_y = int(h * line_room_pct)

        # Draw Lines
        cv2.line(frame, (0, door_y), (w, door_y), (255, 255, 255), 3)
        cv2.line(frame, (0, room_y), (w, room_y), (0, 255, 0), 3)
        cv2.putText(frame, "DOOR ZONE (A)", (10, door_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "BUFFER ZONE (B)", (10, room_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "ROOM ZONE (C)", (10, room_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if run_ai:
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
            dets = sv.Detections.from_ultralytics(results)

            # --- FILTER: SIZE CHECK ---
            valid_indices = []
            for i, bbox in enumerate(dets.xyxy):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area > MIN_BOX_AREA:
                    valid_indices.append(i)

            if len(valid_indices) > 0:
                dets = dets[valid_indices]
            else:
                dets = sv.Detections.empty()

            dets = tracker.update_with_detections(dets)

            current_states = {}
            current_time = time.time()

            for i, (tid, cid) in enumerate(zip(dets.tracker_id, dets.class_id)):
                if cid != 0: continue

                bbox = dets.xyxy[i]
                _, feet_y = get_feet_position(bbox)

                # Zone Logic
                if feet_y < door_y:
                    curr_pos = "A"  # Outside
                elif feet_y < room_y:
                    curr_pos = "B"  # Buffer
                else:
                    curr_pos = "C"  # Inside

                current_states[tid] = curr_pos

                # Origin Logic
                if tid not in st.session_state['tracker_origin']:
                    st.session_state['tracker_origin'][tid] = curr_pos

                origin = st.session_state['tracker_origin'][tid]
                prev_pos = st.session_state['tracker_state'].get(tid)

                if prev_pos:
                    # ENTRY
                    if (prev_pos == "A" or prev_pos == "B") and curr_pos == "C":
                        if origin != "C":
                            last_time = st.session_state['last_entry_time'].get(tid, 0)
                            if current_time - last_time > COOLDOWN_SECONDS:
                                st.session_state['count_in'] += 1
                                st.session_state['recent_events'].insert(0, log_event("ENTRY", tid))
                                st.session_state['last_entry_time'][tid] = current_time

                                in_metric.metric("Total Entered", st.session_state['count_in'])
                                df = pd.DataFrame(st.session_state['recent_events'][:8])
                                log_table.dataframe(df, hide_index=True, use_container_width=True)

                    # EXIT
                    elif (prev_pos == "C" or prev_pos == "B") and curr_pos == "A":
                        if origin != "A":
                            last_time = st.session_state['last_exit_time'].get(tid, 0)
                            if current_time - last_time > COOLDOWN_SECONDS:
                                st.session_state['count_out'] += 1
                                st.session_state['recent_events'].insert(0, log_event("EXIT", tid))
                                st.session_state['last_exit_time'][tid] = current_time

                                out_metric.metric("Total Exited", st.session_state['count_out'])
                                df = pd.DataFrame(st.session_state['recent_events'][:8])
                                log_table.dataframe(df, hide_index=True, use_container_width=True)

            st.session_state['tracker_state'].update(current_states)

            # Visuals
            labels = []
            for tid, cid in zip(dets.tracker_id, dets.class_id):
                object_name = model.names[cid]
                if cid == 0:
                    labels.append(f"#{tid} Person")
                else:
                    labels.append(f"{object_name}")

            frame = box_an.annotate(frame, dets)
            frame = label_an.annotate(frame, dets, labels=labels)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


if __name__ == "__main__":
    main()