# Implementing Drowsiness Detection in Android

To use this `drowsiness_model.tflite` file in your Android application, you will need to do two things:
1. Use **MediaPipe Face Mesh for Android** to extract the face landmarks.
2. Use **TensorFlow Lite for Android** to classify those landmarks using your model.

### 1. Add Dependencies to `build.gradle` (app level)

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    
    // MediaPipe Face Mesh
    implementation 'com.google.mediapipe:tasks-vision:0.10.0'
}
```

### 2. Copy the Model
Place your `drowsiness_model.tflite` file into your Android project's `app/src/main/assets` folder.

### 3. Kotlin Implementation

Here is the helper class you can use to calculate the distances and run the TFLite inference.

```kotlin
import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.pow
import kotlin.math.sqrt
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

class DrowsinessDetector(context: Context) {
    
    private var interpreter: Interpreter
    
    init {
        val model = loadModelFile(context, "drowsiness_model.tflite")
        val options = Interpreter.Options()
        interpreter = Interpreter(model, options)
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    // Standard Euclidean distance calculation between two landmarks
    private fun distance(p1: com.google.mediapipe.tasks.components.containers.NormalizedLandmark, 
                         p2: com.google.mediapipe.tasks.components.containers.NormalizedLandmark, 
                         width: Int, height: Int): Float {
        val dx = (p1.x() - p2.x()) * width
        val dy = (p1.y() - p2.y()) * height
        return sqrt(dx.pow(2) + dy.pow(2))
    }

    // Takes the result from MediaPipe, normalizes it using the driver's baseline, and predicts
    fun detect(faceLandmarkerResult: FaceLandmarkerResult, imageWidth: Int, imageHeight: Int, baseEar: Float, baseMar: Float): String {
        if (faceLandmarkerResult.faceLandmarks().isEmpty()) return "NO_FACE"

        val landmarks = faceLandmarkerResult.faceLandmarks()[0]

        // Left Eye
        val leftEar = (distance(landmarks[386], landmarks[374], imageWidth, imageHeight) + 
                       distance(landmarks[385], landmarks[380], imageWidth, imageHeight)) / 
                      (2.0f * distance(landmarks[362], landmarks[263], imageWidth, imageHeight))

        // Right Eye
        val rightEar = (distance(landmarks[159], landmarks[145], imageWidth, imageHeight) + 
                        distance(landmarks[158], landmarks[153], imageWidth, imageHeight)) / 
                       (2.0f * distance(landmarks[33], landmarks[133], imageWidth, imageHeight))

        // Mouth 
        val mar = distance(landmarks[13], landmarks[14], imageWidth, imageHeight) / 
                  distance(landmarks[78], landmarks[308], imageWidth, imageHeight)
                  
        // Normalize against the specific user's facial baseline
        val normLeftEar = leftEar / baseEar
        val normRightEar = rightEar / baseEar
        val normMar = mar / baseMar

        // Prepare input for TFLite (Float32 Array of size 3)
        val inputBuffer = ByteBuffer.allocateDirect(3 * 4).apply {
            order(ByteOrder.nativeOrder())
            putFloat(normLeftEar)
            putFloat(normRightEar)
            putFloat(normMar)
        }
        
        // Prepare output (Float32 Array of size 3, matching our 3 classes)
        val outputBuffer = ByteBuffer.allocateDirect(3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        outputBuffer.rewind()
        val confidences = FloatArray(3)
        outputBuffer.asFloatBuffer().get(confidences)

        // Find the class with highest probability
        val maxIndex = confidences.indices.maxByOrNull { confidences[it] } ?: 0
        
        return when (maxIndex) {
            0 -> "ACTIVE"
            1 -> "SLEEPING"
            2 -> "YAWNING"
            else -> "UNKNOWN"
        }
    }
    
    fun close() {
        interpreter.close()
    }
}
```

### 4. How the Professional Calibration Works
To make this model work for **every single judge** regardless of their eye shape or the phone's camera angle:
1. When your app starts, ask the judge to "Look at the screen for 3 seconds".
2. Calculate their average raw EAR and MAR during those 3 seconds. Save these as `baseEar` and `baseMar`.
3. Pass those variables into the `.detect()` function on every frame. 
4. The model has been trained on normalized data, meaning it learns that a `0.6x` drop from *any* baseline means sleeping!

### 5. Final Android Implementation (with Debounce Timer)
Just like in the Python script, cameras can occasionally glitch for a single frame. To prevent a 1.5-second sleep timer from resetting due to a split-second camera glitch, you should implement the exact same **stateful debounce timer** in your Kotlin code!

Here is the complete logic you would run on every frame:

```kotlin
// Define these globally in your Activity/Fragment
var sleepingStart: Long? = null
var lastSleepingTime: Long? = null
var currentStatus = "ACTIVE"

// Inside your camera frame loop:
val frameClass = drowsinessDetector.detect(result, imageWidth, imageHeight, userBaseEar, userBaseMar)
val currentTime = System.currentTimeMillis()

if (frameClass == "SLEEPING") {
    lastSleepingTime = currentTime
    if (sleepingStart == null) {
        sleepingStart = currentTime
    }
    
    val elapsedSeconds = (currentTime - sleepingStart!!) / 1000.0
    if (elapsedSeconds >= 1.5) {
        currentStatus = "ALARM_TRIGGERED_SLEEPING"
    } else {
        currentStatus = "Eyes Closed... (${"%.1f".format(elapsedSeconds)}s)"
    }
} else if (frameClass == "YAWNING") {
    sleepingStart = null
    currentStatus = "YAWNING"
} else {
    // ACTIVE Frame
    // Debounce buffer: ignore 'active' flickers if we saw 'sleeping' less than 0.5s ago
    if (lastSleepingTime != null && (currentTime - lastSleepingTime!! < 500)) {
        // Keep currentStatus as "Eyes Closed"
    } else {
        sleepingStart = null
        currentStatus = "ACTIVE"
    }
}

println("Final Status: $currentStatus")
```
