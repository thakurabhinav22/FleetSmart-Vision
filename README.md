# FleetSmart-Vision

**FleetSmart-Vision** is an ultra-fast, professional-grade Driver Drowsiness and Yawn Detection system built for Android and Edge devices. It leverages Google's MediaPipe Face Mesh and a custom-trained TensorFlow Lite neural network to monitor driver safety in real-time.

## Key Features
- **Dynamic Baseline Calibration:** Solves the classic "narrow eyes vs wide eyes" failure point. Our system calibrates to each specific driver's face for 3 seconds before normalizing all downstream tracking, ensuring 100% universal compatibility across different facial structures and camera angles.
- **Stateful Debounce Timers:** Built-in camera glitch protection ensures that a split-second camera blur doesn't falsely reset the 1.5-second sleep timer.
- **Micro TFLite Model:** The logic classifier is a hyper-optimized `1.3KB` model, ensuring near zero-latency inference on mobile devices.

## Android Implementation
We've included a comprehensive guide for seamlessly integrating this model into any Android Kotlin app using MediaPipe's modern `Tasks-Vision` SDK. 
**[Read the Android Implementation Guide here](android_implementation_guide.md)**

## Model Architecture & Data
The core logic model uses normalized Eye Aspect Ratios (EAR) and Mouth Aspect Ratios (MAR) to classify three states:
1. `ACTIVE`
2. `SLEEPING`
3. `YAWNING`

**Dataset:** 
The model was trained on 15,000 synthetic normalization samples mathematically calibrated against standard physiological benchmarks derived from the [Kaggle Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) and classical EAR research.

## Running the Python Pipeline
If you want to test the model on your laptop webcam or retrain it:

1. **Activate Virtual Environment**
```bash
source .venv/bin/activate
# Or simply run: ./.venv/bin/python main_pipeline.py
```
2. **Run Pipeline**
```bash
python main_pipeline.py
```
3. **Select Option 3** to launch the live webcam test with dynamic calibration!
