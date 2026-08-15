import cv2
import os
import time
import csv
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mediapipe as mp

# MediaPipe Setup using the same technique as SignLangMLKIT (mp.solutions)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

CSV_FILE = 'dataset.csv'
CLASSES = ['active', 'sleeping', 'yawning']

def distance(p1, p2, w, h):
    """ Euclidean distance between two landmarks (ignoring Z for aspect ratio) """
    return np.linalg.norm(np.array([p1.x * w, p1.y * h]) - np.array([p2.x * w, p2.y * h]))

def get_aspect_ratios(landmarks, w, h):
    """
    Calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
    This forces the AI to ONLY look at eye/mouth openness, completely ignoring head tilt/position.
    """
    # Left eye landmarks
    left_ear = (distance(landmarks[386], landmarks[374], w, h) + distance(landmarks[385], landmarks[380], w, h)) / (2.0 * distance(landmarks[362], landmarks[263], w, h))
    
    # Right eye landmarks
    right_ear = (distance(landmarks[159], landmarks[145], w, h) + distance(landmarks[158], landmarks[153], w, h)) / (2.0 * distance(landmarks[33], landmarks[133], w, h))
    
    # Mouth landmarks
    mar = distance(landmarks[13], landmarks[14], w, h) / distance(landmarks[78], landmarks[308], w, h)
    
    return [left_ear, right_ear, mar]

def extract_landmarks(image, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    
    annotated_image = image.copy()
    h, w, _ = image.shape
    
    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        
        # Fast native drawing
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Extract robust features (EAR, MAR) instead of raw coordinates
        features = get_aspect_ratios(face_landmarks.landmark, w, h)
        return features, annotated_image
    
    return None, annotated_image

def collect_samples():
    print("\n🚀 Launching Data Collection...\n")
    
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        for class_idx, class_name in enumerate(CLASSES):
            print(f"\n--- Get ready for '{class_name.upper()}' samples ---")
            
            # Non-blocking countdown
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if ret:
                    remaining = int(3 - (time.time() - start_time)) + 1
                    cv2.putText(frame, f"Ready for {class_name} in {remaining}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('Face Collector', frame)
                    cv2.waitKey(1)
            
            count = 0
            num_samples = 40 # Increased slightly for better EAR/MAR variance
            print(f"Capturing {num_samples} samples for '{class_name}'...")
            last_capture_time = time.time()
            
            while count < num_samples:
                ret, frame = cap.read()
                if not ret: break
                    
                current_time = time.time()
                if current_time - last_capture_time >= 0.15:
                    features, annotated_frame = extract_landmarks(frame, face_mesh)
                    
                    if features:
                        with open(CSV_FILE, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([class_idx] + features)
                        count += 1
                    else:
                        cv2.putText(annotated_frame, "No face detected!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                    last_capture_time = current_time
                else:
                    annotated_frame = frame.copy()
                    
                cv2.putText(annotated_frame, f"Capturing {class_name}: {count}/{num_samples}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Collector', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Sample collection finished! Eye and Mouth Ratios saved to '{CSV_FILE}'.")

def build_model(input_dim, num_classes=3):
    # Model is extremely simple now because the features (EAR/MAR) are mathematically robust
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_export():
    print("\n🚀 Launching Model Training...\n")
    if not os.path.exists(CSV_FILE):
        print("❌ CSV file not found. Please collect data first.")
        return
        
    print("Loading CSV data...")
    try:
        data = np.loadtxt(CSV_FILE, delimiter=',')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    y = data[:, 0]
    X = data[:, 1:]
    
    input_dim = X.shape[1]
    model = build_model(input_dim, num_classes=len(CLASSES))
    
    print("\nTraining lightweight Dense model on Eye/Mouth Ratios...")
    model.fit(X, y, epochs=150, batch_size=32, validation_split=0.2)
    
    print("\n--- Converting to TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = 'drowsiness_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"\n✅ Success! Model exported as '{tflite_path}'.")

def test_model():
    print("\n🚀 Launching Live Testing...\n")
    tflite_path = 'drowsiness_model.tflite'
    if not os.path.exists(tflite_path):
        print("❌ TFLite model not found. Please train first.")
        return
        
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        # --- CALIBRATION PHASE ---
        print("\n=== CALIBRATION ===")
        print("Please look at the camera with a neutral face and open eyes.")
        base_features = []
        calib_start = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            elapsed = time.time() - calib_start
            if elapsed > 4.0:
                break
                
            features, annotated_frame = extract_landmarks(frame, face_mesh)
            if features:
                base_features.append(features)
                
            cv2.putText(annotated_frame, f"CALIBRATING: Keep Eyes Open ({4.0 - elapsed:.1f}s)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Drowsiness AI Testing', annotated_frame)
            cv2.waitKey(1)
            
        if len(base_features) < 10:
            print("Failed to calibrate. Make sure your face is visible.")
            cap.release()
            cv2.destroyAllWindows()
            return
            
        # Calculate baseline averages
        base_features = np.array(base_features)
        base_ear = (np.mean(base_features[:, 0]) + np.mean(base_features[:, 1])) / 2.0
        base_mar = np.mean(base_features[:, 2])
        if base_mar < 0.01: base_mar = 0.01 # prevent div by zero
        
        print(f"\nCalibration Complete! Base EAR: {base_ear:.3f}, Base MAR: {base_mar:.3f}")
        print("Live testing started. Press 'q' to stop.")
        
        last_process_time = time.time()
        
        # Timers to enforce 1.5s sleep and 2s yawn limits
        sleeping_start = None
        last_sleeping_time = None
        yawning_start = None
        current_status = "ACTIVE"
        last_color = (0, 255, 0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            current_time = time.time()
            annotated_frame = frame.copy()
            
            if current_time - last_process_time >= 0.1:
                features, processed_frame = extract_landmarks(frame, face_mesh)
                annotated_frame = processed_frame
                
                if features:
                    l_ear, r_ear, mar = features
                    avg_ear = (l_ear + r_ear) / 2.0
                    
                    # Normalize features for the ML model
                    norm_l_ear = l_ear / base_ear
                    norm_r_ear = r_ear / base_ear
                    norm_mar = mar / base_mar
                    normalized_features = [norm_l_ear, norm_r_ear, norm_mar]
                    
                    input_data = np.array([normalized_features], dtype=np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    class_idx = np.argmax(output_data[0])
                    frame_class = CLASSES[class_idx]
                    
                    # Stateful Timer Logic
                    if frame_class == 'sleeping':
                        yawning_start = None
                        last_sleeping_time = current_time
                        if sleeping_start is None:
                            sleeping_start = current_time
                        
                        if current_time - sleeping_start >= 1.5:
                            current_status = "SLEEPING"
                            last_color = (0, 0, 255)
                        else:
                            current_status = f"Eyes Closed... ({current_time - sleeping_start:.1f}s)"
                            last_color = (0, 255, 255) # Yellow warning
                            
                    elif frame_class == 'yawning':
                        sleeping_start = None
                        if yawning_start is None:
                            yawning_start = current_time
                        
                        if current_time - yawning_start >= 2.0:
                            current_status = "YAWNING"
                            last_color = (0, 165, 255)
                        else:
                            current_status = f"Mouth Open... ({current_time - yawning_start:.1f}s)"
                            last_color = (0, 255, 255)
                            
                    else:
                        yawning_start = None
                        
                        # Debounce buffer: ignore 'active' flickers if we saw 'sleeping' less than 0.5s ago
                        if last_sleeping_time is not None and (current_time - last_sleeping_time < 0.5):
                            pass # Keep the current status as "Eyes Closed"
                        else:
                            sleeping_start = None
                            current_status = "ACTIVE"
                            last_color = (0, 255, 0)
                        
                    # Add debug metrics to screen
                    cv2.putText(annotated_frame, f"EAR: {avg_ear:.3f} | MAR: {mar:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"NORM EAR: {(avg_ear/base_ear):.2f}x | NORM MAR: {(mar/base_mar):.2f}x", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                last_process_time = current_time
            
            cv2.putText(annotated_frame, f"STATUS: {current_status}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, last_color, 3)
            cv2.imshow('Drowsiness AI Testing', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("========================================")
    print("      DROWSINESS AI LAUNCHER            ")
    print("========================================")
    print("What would you like to run?")
    print("  1. Collect Data (Face Markings)")
    print("  2. Train Model")
    print("  3. Test Model Live")
    print("========================================")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        collect_samples()
    elif choice == "2":
        train_and_export()
    elif choice == "3":
        test_model()
    else:
        print("\n❌ Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == '__main__':
    main()
