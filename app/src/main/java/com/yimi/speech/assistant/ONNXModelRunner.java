package com.yimi.speech.assistant;

import android.content.res.AssetManager;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ONNXModelRunner {

    private static final String TAG = "ONNXModelRunner";

    private static final int BATCH_SIZE = 1;

    private static final OrtEnvironment globalEnv = OrtEnvironment.getEnvironment();

    private final OrtSession wakeWordSession;
    private final OrtSession melSession;
    private final OrtSession embedSession;
    private final OrtSession vadSession;

    private final AssetManager assetManager;

    public ONNXModelRunner(AssetManager assetManager) throws IOException, OrtException {
        this.assetManager = assetManager;

        // Use NNApi for better performance
        // OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // options.addNnapi();

        vadSession      = globalEnv.createSession(readModelFile("silero_vad.onnx"));
        wakeWordSession = globalEnv.createSession(readModelFile("hey_jarvis_v0.1.onnx"));

        melSession      = globalEnv.createSession(readModelFile("melspectrogram.onnx"));
        embedSession    = globalEnv.createSession(readModelFile("embedding_model.onnx"));
    }

    public String process(float[] pcmInput) throws OrtException, IOException {

        float[][] mel = getMelSpectrogram(pcmInput);  // [T, F]
        float[][][] embedInput = new float[1][mel.length][mel[0].length];

        for (int i = 0; i < mel.length; i++) {
            System.arraycopy(mel[i], 0, embedInput[0][i], 0, mel[0].length);
        }

        return predictWakeWord(embedInput);
    }

    public float[][] getMelSpectrogram(float[] inputArray) throws OrtException {

        int samples = inputArray.length;
        FloatBuffer floatBuffer = FloatBuffer.wrap(inputArray);

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(globalEnv, floatBuffer, new long[]{BATCH_SIZE, samples});
             OrtSession.Result results = melSession.run(Collections.singletonMap(melSession.getInputNames().iterator().next(), inputTensor))) {

            float[][][][] outputTensor = (float[][][][]) results.get(0).getValue();
            float[][] squeezed = squeeze(outputTensor);

            return applyMelSpecTransform(squeezed);
        }
    }

    public static float[][] squeeze(float[][][][] originalArray) {

        int T = originalArray[0][0].length;
        int F = originalArray[0][0][0].length;
        float[][] squeezed = new float[T][F];
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < F; j++) {
                squeezed[i][j] = originalArray[0][0][i][j];
            }
        }

        return squeezed;
    }

    public static float[][] applyMelSpecTransform(float[][] array) {

        int T = array.length;
        int F = array[0].length;
        float[][] transformed = new float[T][F];

        for (int i = 0; i < T; i++) {
            for (int j = 0; j < F; j++) {
                transformed[i][j] = array[i][j] / 10.0f + 2.0f;
            }
        }

        return transformed;
    }

    public float[][] generateEmbeddings(float[][][][] input) throws OrtException {

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(globalEnv, input);
             OrtSession.Result results = embedSession.run(Collections.singletonMap("input_1", inputTensor))) {

            float[][][][] rawOutput = (float[][][][]) results.get(0).getValue();
            int T = rawOutput.length;
            int D = rawOutput[0][0][0].length;
            float[][] reshaped = new float[T][D];
            for (int i = 0; i < T; i++) {
                System.arraycopy(rawOutput[i][0][0], 0, reshaped[i], 0, D);
            }

            return reshaped;
        }
    }

    public String predictWakeWord(float[][][] inputArray) throws OrtException {

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(globalEnv, inputArray);
             OrtSession.Result outputs = wakeWordSession.run(Collections.singletonMap(wakeWordSession.getInputNames().iterator().next(), inputTensor))) {

            float[][] result = (float[][]) outputs.get(0).getValue();

            return String.format("%.5f", result[0][0]);
        }
    }

    private byte[] readModelFile(String filename) throws IOException {

        try (InputStream is = assetManager.open(filename)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);

            return buffer;
        }
    }

    public void close() {
        try {
            wakeWordSession.close();
            melSession.close();
            embedSession.close();
        } catch (OrtException e) {
            Log.e(TAG, "Failed to close sessions", e);
        }
    }
}
