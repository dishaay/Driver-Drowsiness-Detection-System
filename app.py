import os

# Reduce TensorFlow console messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import customtkinter as ctk
from PIL import Image
import tensorflow as tf
import numpy as np
import time
import threading
import winsound
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 64

# CNN output:
# 0 = CLOSED
# 1 = OPEN
OPEN_THRESHOLD = 0.60

# Ignore normal short blinks
BLINK_GRACE_TIME = 0.40

# Eyes must remain closed this long for drowsiness
DROWSY_TIME = 2.5

# Smooth predictions
SMOOTHING_WINDOW = 4


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "drowsiness_cnn.keras"
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

print("Loading CNN model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN loaded successfully!")


# ============================================================
# LOAD HAAR CASCADES
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_eye_tree_eyeglasses.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load face Haar cascade."
    )

if eye_cascade.empty():
    raise RuntimeError(
        "Could not load eye Haar cascade."
    )


# ============================================================
# WINDOW
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title(
    "Driver Drowsiness Detection System"
)

app.geometry("1100x760")
app.resizable(False, False)


# ============================================================
# TITLE
# ============================================================

title = ctk.CTkLabel(
    app,
    text="DRIVER DROWSINESS DETECTION SYSTEM",
    font=("Arial", 30, "bold")
)

title.pack(
    pady=(20, 5)
)


subtitle = ctk.CTkLabel(
    app,
    text="CNN-Based Real-Time Driver Fatigue Monitoring",
    font=("Arial", 15)
)

subtitle.pack(
    pady=(0, 15)
)


# ============================================================
# CAMERA PANEL
# ============================================================

camera_container = ctk.CTkFrame(
    app,
    width=800,
    height=440,
    corner_radius=15
)

camera_container.pack(
    pady=10
)


camera_label = ctk.CTkLabel(
    camera_container,
    text="Starting Camera...",
    width=760,
    height=400
)

camera_label.pack(
    padx=20,
    pady=20
)


# ============================================================
# STATUS PANEL
# ============================================================

status_frame = ctk.CTkFrame(
    app,
    corner_radius=15
)

status_frame.pack(
    pady=15,
    padx=40,
    fill="x"
)


eye_label = ctk.CTkLabel(
    status_frame,
    text="Eye Status: Detecting...",
    font=("Arial", 18, "bold")
)

eye_label.grid(
    row=0,
    column=0,
    padx=30,
    pady=20
)


driver_status = ctk.CTkLabel(
    status_frame,
    text="Driver Status: AWAKE",
    font=("Arial", 18, "bold")
)

driver_status.grid(
    row=0,
    column=1,
    padx=30,
    pady=20
)


score_label = ctk.CTkLabel(
    status_frame,
    text="Drowsiness Score: 0%",
    font=("Arial", 18, "bold")
)

score_label.grid(
    row=0,
    column=2,
    padx=30,
    pady=20
)


status_frame.grid_columnconfigure(
    0,
    weight=1
)

status_frame.grid_columnconfigure(
    1,
    weight=1
)

status_frame.grid_columnconfigure(
    2,
    weight=1
)


# ============================================================
# WARNING LABEL
# ============================================================

warning_label = ctk.CTkLabel(
    app,
    text="SYSTEM ACTIVE",
    font=("Arial", 20, "bold"),
    text_color="green"
)

warning_label.pack(
    pady=5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# STATE VARIABLES
# ============================================================

running = True

eyes_closed_start = None

alarm_active = False

probability_history = deque(
    maxlen=SMOOTHING_WINDOW
)

# Stores eye locations relative to the face
last_eye_boxes = []


# ============================================================
# ALARM
# ============================================================

def play_alarm():
    """
    Plays repeated short warning beeps.

    Short beeps are used instead of one long blocking beep,
    so the webcam/UI does not freeze.
    """

    try:

        for _ in range(8):

            winsound.Beep(
                1800,
                350
            )

            time.sleep(
                0.10
            )

    except Exception as error:

        print(
            "Alarm error:",
            error
        )


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_eyes(eye_images):

    processed = []


    for eye in eye_images:

        if eye is None:
            continue

        if eye.size == 0:
            continue


        eye = cv2.resize(
            eye,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )


        eye = eye.astype(
            np.float32
        )


        # Add channel dimension
        eye = np.expand_dims(
            eye,
            axis=-1
        )


        processed.append(
            eye
        )


    if len(processed) == 0:
        return None


    # Batch both eyes together
    batch = np.stack(
        processed,
        axis=0
    )


    # IMPORTANT:
    # Do NOT divide by 255 here.
    # Our CNN already contains Rescaling(1./255).

    predictions = model(
        batch,
        training=False
    ).numpy().reshape(-1)


    open_probability = float(
        np.mean(predictions)
    )


    return open_probability


# ============================================================
# SET DRIVER AWAKE
# ============================================================

def set_awake():

    global eyes_closed_start
    global alarm_active


    eyes_closed_start = None

    alarm_active = False


    eye_label.configure(
        text="Eye Status: OPEN"
    )


    driver_status.configure(
        text="Driver Status: AWAKE"
    )


    score_label.configure(
        text="Drowsiness Score: 0%"
    )


    warning_label.configure(
        text="SYSTEM ACTIVE - DRIVER AWAKE",
        text_color="green"
    )


# ============================================================
# START ALARM
# ============================================================

def start_alarm():

    global alarm_active


    if alarm_active:
        return


    alarm_active = True


    print(
        "DROWSINESS DETECTED - ALARM STARTED"
    )


    alarm_thread = threading.Thread(
        target=play_alarm,
        daemon=True
    )


    alarm_thread.start()


# ============================================================
# CAMERA LOOP
# ============================================================

def update_camera():

    global eyes_closed_start
    global alarm_active
    global last_eye_boxes


    if not running:
        return


    ret, frame = cap.read()


    if not ret:

        camera_label.after(
            30,
            update_camera
        )

        return


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )


    final_eye_state = None
    smoothed_probability = None


    # ========================================================
    # FACE FOUND
    # ========================================================

    if len(faces) > 0:

        # Largest detected face
        x, y, w, h = max(
            faces,
            key=lambda box:
            box[2] * box[3]
        )


        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Upper 60% of face
        upper_height = int(
            h * 0.60
        )


        upper_gray = gray[
            y:y + upper_height,
            x:x + w
        ]


        # ====================================================
        # HAAR EYE LOCALIZATION
        # ====================================================

        detected_eyes = eye_cascade.detectMultiScale(
            upper_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )


        valid_boxes = []


        for ex, ey, ew, eh in detected_eyes:

            # Reject detections too low on face
            if ey > int(
                upper_height * 0.75
            ):
                continue


            relative_box = (
                ex / w,
                ey / h,
                ew / w,
                eh / h
            )


            valid_boxes.append(
                (
                    ex,
                    ey,
                    ew,
                    eh,
                    relative_box
                )
            )


        # Keep largest two eyes
        valid_boxes = sorted(
            valid_boxes,
            key=lambda item:
            item[2] * item[3],
            reverse=True
        )[:2]


        # ====================================================
        # SAVE EYE LOCATIONS
        # ====================================================

        if len(valid_boxes) > 0:

            last_eye_boxes = [
                item[4]
                for item in valid_boxes
            ]


        # ====================================================
        # CREATE EYE CROPS
        # ====================================================

        eye_crops = []
        draw_boxes = []


        # Use remembered eye positions
        if len(last_eye_boxes) > 0:

            for (
                rx,
                ry,
                rw,
                rh
            ) in last_eye_boxes[:2]:


                ex = int(
                    rx * w
                )

                ey = int(
                    ry * h
                )

                ew = int(
                    rw * w
                )

                eh = int(
                    rh * h
                )


                # Small padding
                pad_x = int(
                    ew * 0.10
                )

                pad_y = int(
                    eh * 0.10
                )


                ex1 = max(
                    0,
                    ex - pad_x
                )

                ey1 = max(
                    0,
                    ey - pad_y
                )

                ex2 = min(
                    w,
                    ex + ew + pad_x
                )

                ey2 = min(
                    upper_height,
                    ey + eh + pad_y
                )


                if (
                    ex2 > ex1
                    and ey2 > ey1
                ):

                    crop = upper_gray[
                        ey1:ey2,
                        ex1:ex2
                    ]


                    if crop.size > 0:

                        eye_crops.append(
                            crop
                        )


                        draw_boxes.append(
                            (
                                ex1,
                                ey1,
                                ex2,
                                ey2
                            )
                        )


        # ====================================================
        # FALLBACK EYE REGIONS
        # ====================================================

        # If Haar has not yet located eyes,
        # use approximate eye regions.

        if len(eye_crops) == 0:

            eye_y1 = int(
                h * 0.22
            )

            eye_y2 = int(
                h * 0.50
            )


            left_x1 = int(
                w * 0.08
            )

            left_x2 = int(
                w * 0.48
            )


            right_x1 = int(
                w * 0.52
            )

            right_x2 = int(
                w * 0.92
            )


            if eye_y2 <= upper_height:

                left_eye = upper_gray[
                    eye_y1:eye_y2,
                    left_x1:left_x2
                ]


                right_eye = upper_gray[
                    eye_y1:eye_y2,
                    right_x1:right_x2
                ]


                if left_eye.size > 0:

                    eye_crops.append(
                        left_eye
                    )


                    draw_boxes.append(
                        (
                            left_x1,
                            eye_y1,
                            left_x2,
                            eye_y2
                        )
                    )


                if right_eye.size > 0:

                    eye_crops.append(
                        right_eye
                    )


                    draw_boxes.append(
                        (
                            right_x1,
                            eye_y1,
                            right_x2,
                            eye_y2
                        )
                    )


        # ====================================================
        # CNN PREDICTION
        # ====================================================

        open_probability = predict_eyes(
            eye_crops
        )


        if open_probability is not None:

            probability_history.append(
                open_probability
            )


            smoothed_probability = float(
                np.mean(
                    probability_history
                )
            )


            # Decide OPEN / CLOSED
            if (
                smoothed_probability
                >= OPEN_THRESHOLD
            ):

                final_eye_state = "OPEN"

            else:

                final_eye_state = "CLOSED"


            # Eye rectangle colour
            if final_eye_state == "OPEN":

                box_color = (
                    0,
                    255,
                    0
                )

            else:

                box_color = (
                    0,
                    0,
                    255
                )


            # Draw eye boxes
            for (
                ex1,
                ey1,
                ex2,
                ey2
            ) in draw_boxes:

                cv2.rectangle(
                    frame,
                    (
                        x + ex1,
                        y + ey1
                    ),
                    (
                        x + ex2,
                        y + ey2
                    ),
                    box_color,
                    2
                )


            # Display CNN probability
            cv2.putText(
                frame,
                (
                    "CNN Open Probability: "
                    f"{smoothed_probability:.2f}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        # ====================================================
        # OPEN EYES
        # ====================================================

        if final_eye_state == "OPEN":

            set_awake()


        # ====================================================
        # CLOSED EYES
        # ====================================================

        elif final_eye_state == "CLOSED":

            current_time = time.time()


            # Start closed-eye timer
            if eyes_closed_start is None:

                eyes_closed_start = current_time


            closed_duration = (
                current_time
                - eyes_closed_start
            )


            # =================================================
            # NORMAL BLINK
            # =================================================

            if (
                closed_duration
                < BLINK_GRACE_TIME
            ):

                eye_label.configure(
                    text="Eye Status: BLINK / CHECKING"
                )


                driver_status.configure(
                    text="Driver Status: AWAKE"
                )


                score_label.configure(
                    text="Drowsiness Score: 0%"
                )


                warning_label.configure(
                    text="SYSTEM ACTIVE",
                    text_color="green"
                )


            # =================================================
            # EYES ACTUALLY CLOSED
            # =================================================

            else:

                eye_label.configure(
                    text="Eye Status: CLOSED"
                )


                effective_duration = (
                    closed_duration
                    - BLINK_GRACE_TIME
                )


                effective_threshold = (
                    DROWSY_TIME
                    - BLINK_GRACE_TIME
                )


                score = int(
                    (
                        effective_duration
                        / effective_threshold
                    )
                    * 100
                )


                score = max(
                    0,
                    min(
                        score,
                        100
                    )
                )


                score_label.configure(
                    text=(
                        "Drowsiness Score: "
                        f"{score}%"
                    )
                )


                # =============================================
                # MONITORING
                # =============================================

                if (
                    closed_duration
                    < DROWSY_TIME
                ):

                    driver_status.configure(
                        text="Driver Status: MONITORING"
                    )


                    warning_label.configure(
                        text=(
                            "EYES CLOSED - "
                            "MONITORING..."
                        ),
                        text_color="orange"
                    )


                # =============================================
                # DROWSINESS DETECTED
                # =============================================

                else:

                    driver_status.configure(
                        text="Driver Status: DROWSY"
                    )


                    warning_label.configure(
                        text=(
                            "WARNING! "
                            "DROWSINESS DETECTED"
                        ),
                        text_color="red"
                    )


                    # Red warning border
                    cv2.rectangle(
                        frame,
                        (5, 5),
                        (
                            frame.shape[1] - 5,
                            frame.shape[0] - 5
                        ),
                        (0, 0, 255),
                        8
                    )


                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (
                            120,
                            frame.shape[0] - 40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 0, 255),
                        3
                    )


                    # START ALARM
                    start_alarm()


        # ====================================================
        # NO CNN RESULT
        # ====================================================

        else:

            eyes_closed_start = None
            alarm_active = False


            eye_label.configure(
                text="Eye Status: DETECTING..."
            )


            driver_status.configure(
                text="Driver Status: AWAKE"
            )


            score_label.configure(
                text="Drowsiness Score: 0%"
            )


            warning_label.configure(
                text="CHECKING EYES...",
                text_color="orange"
            )


    # ========================================================
    # NO FACE
    # ========================================================

    else:

        eyes_closed_start = None
        alarm_active = False

        probability_history.clear()
        last_eye_boxes = []


        eye_label.configure(
            text="Eye Status: ---"
        )


        driver_status.configure(
            text="Driver Status: NO FACE"
        )


        score_label.configure(
            text="Drowsiness Score: 0%"
        )


        warning_label.configure(
            text="FACE NOT DETECTED",
            text_color="orange"
        )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    display_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    display_frame = cv2.resize(
        display_frame,
        (760, 400)
    )


    pil_image = Image.fromarray(
        display_frame
    )


    camera_image = ctk.CTkImage(
        light_image=pil_image,
        dark_image=pil_image,
        size=(760, 400)
    )


    camera_label.configure(
        image=camera_image,
        text=""
    )


    camera_label.image = camera_image


    if running:

        camera_label.after(
            25,
            update_camera
        )


# ============================================================
# CLOSE APP
# ============================================================

def close_app():

    global running
    global alarm_active


    running = False
    alarm_active = False


    if cap.isOpened():

        cap.release()


    app.destroy()


# ============================================================
# STOP BUTTON
# ============================================================

stop_button = ctk.CTkButton(
    app,
    text="STOP CAMERA",
    command=close_app,
    width=220,
    height=42,
    font=("Arial", 16, "bold")
)


stop_button.pack(
    pady=12
)


app.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# ============================================================
# START APPLICATION
# ============================================================

update_camera()

app.mainloop()