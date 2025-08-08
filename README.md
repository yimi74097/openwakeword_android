
# openwakeword\_android

An **ONNX Runtime–based** Android example project for **on-device wake word detection**.
This project demonstrates a complete pipeline using **Java (or Kotlin) + JNI + C/C++**, from **16 kHz microphone PCM capture**, **feature extraction (MFCC/Mel)**, to **OpenWakeWord ONNX model** inference with threshold, patience, and debounce control.

> License: Apache-2.0

## Features

* 🎙️ Real-time **16 kHz** audio capture via `AudioRecord`
* 🧩 **ONNX Runtime** integration for on-device inference (extendable to NNAPI / XNNPACK)
* 🔎 Streaming detection with **80 ms (1280 samples)** step size
* 📈 Configurable **confidence threshold**, **patience frames**, and **debounce time**
* 🔧 Modular feature extraction and model input adaptation layer
* 🧪 Simple UI/log feedback for demonstration

## Project Structure

```
.
├── app/
│   ├── src/main/java/...        # Java layer: AudioRecord, Detector, UI
│   ├── src/main/cpp/            # C/C++ layer: JNI, optional feature extraction/acceleration
│   ├── src/main/assets/         # Place *.onnx model files here
│   ├── src/main/AndroidManifest.xml
│   └── CMakeLists.txt           # Native build configuration
├── gradle/ ...                  # Gradle wrapper
├── build.gradle                 # Root build script
├── settings.gradle
├── LICENSE
└── README.md
```

> Adjust package names and paths based on your actual code (e.g., `com.yimi.speech.*`).

## Requirements

* Android Studio (latest stable recommended)
* Android Gradle Plugin 8.x / included Gradle wrapper
* **minSdk 26+ / targetSdk 34**
* NDK (if using JNI/C++)
* **ONNX Runtime for Android** (Java API, Native API, or both)
* Mono 16 kHz PCM audio input

## Dependencies & Setup

### 1) Gradle Dependency (Java AAR Example)

In `app/build.gradle`:

```gradle
dependencies {
    implementation "com.microsoft.onnxruntime:onnxruntime-android:1.18.1"
}
```

For Native API usage, link `onnxruntime` in `CMakeLists.txt` or include `.so` files under `jniLibs/`.

### 2) Permissions

`AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Runtime permission is required for Android 6.0+.

## Model & Features

* **Sample rate**: 16 kHz, mono, float or int16 → normalize to `[-1, 1]`
* **Step size**: 80 ms (1280 samples) per inference frame
* **Features**: Must match training pipeline (MFCC / Mel-spectrogram, etc.)
* **Input shape**: Commonly `float32[1, 16, 96]` or stacked 2D features — verify from your trained model
* **Buffering**: Maintain sliding windows (e.g., 76×32 Mel buffer or embedding queue)

> **Important**: Feature extraction parameters must exactly match training (window size, stride, filter banks, cepstral coefficients, log scaling, normalization).

## Quick Start

1. Place your ONNX model (e.g., `hey_jarvis_v0.1.onnx`) in `app/src/main/assets/`.
2. Sync and build the project in Android Studio.
3. Connect a device (with microphone) or use an emulator with audio input.
4. Run the app and grant microphone permission.
5. Observe confidence scores or wake word trigger logs in the UI.

## Example Usage (Java)

```java
// 1) Load ONNX model
ONNXModelRunner runner = new ONNXModelRunner(context, "hey_jarvis_v0.1.onnx");

// 2) Create Detector
WakeWordDetector detector = new WakeWordDetector(runner);
detector.setThreshold(0.6f);
detector.setPatienceFrames(3);
detector.setDebounceMillis(3000);

// 3) Start audio recording
AudioRecorderThread recorder = new AudioRecorderThread(context, detector);
recorder.start();

// 4) Handle wake event
detector.setOnWakeListener(() -> {
    // UI notification or trigger next action
});
```

> If using JNI/C++, load the native library with `System.loadLibrary(...)` and call `native` methods for model init/inference.

## Performance Tips

* **Thread priority**: Separate audio capture and inference threads, prioritize audio to avoid dropouts.
* **Memory reuse**: Reuse feature and tensor buffers to reduce GC pressure.
* **Inference backend**: Use **ONNX Runtime Mobile**; enable **NNAPI** or **XNNPACK** if supported.
* **Windowing strategy**: Tune step and window length for trade-off between latency, false positives, and CPU usage.
* **Debounce**: Prevent rapid re-triggering.

## Troubleshooting

1. **Wake word not triggering**

   * Confirm 16 kHz mono input
   * Verify feature extraction matches training
   * Lower `threshold` or `patience` temporarily
2. **Too many false triggers**

   * Increase `threshold` and `patience`
   * Adjust debounce time
   * Check noise environment/gain
3. **Shape mismatch error**

   * Print ONNX input names/shapes
   * Ensure float32 type and dimension order matches model
4. **Build / missing .so**

   * Match `abiFilters` with device architecture (`arm64-v8a`, `armeabi-v7a`)
   * Verify `.so` is packaged in APK

## Suggested Parameter Ranges

| Parameter       | Suggested Value | Notes                      |
| --------------- | --------------- | -------------------------- |
| Sample rate     | 16000 Hz        | Standard for speech models |
| Frame length    | 80 ms           | 1280 samples/frame         |
| Threshold       | 0.5 – 0.7       | Higher → more conservative |
| Patience frames | 2 – 5           | Reduces false positives    |
| Debounce        | 2 – 5 s         | Prevents rapid retriggers  |

## Roadmap

* [ ] Complete in-app **feature extraction** in Java/C++
* [ ] Support multiple wake words/models
* [ ] Add NNAPI/XNNPACK config examples
* [ ] Real-time confidence visualization
* [ ] Integration sample: wake → ASR pipeline

## Contributing

PRs and issues welcome for:

* Build/compatibility fixes
* Documentation improvements
* Performance and battery optimizations

## Acknowledgments

* [ONNX Runtime](https://onnxruntime.ai/)
* [OpenWakeWord](https://github.com/dscripka/openwakeword)

## License

Licensed under the **Apache-2.0** License. See `LICENSE` for details.

