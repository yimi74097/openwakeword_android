package com.yimi.speech.assistant;

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.*;

public class WakeWordDetector {

    private static final int SAMPLE_RATE                = 16000;
    private static final int FRAME_SIZE                 = 1280;      // 80ms @ 16kHz
    private static final int MEL_SPEC_MAX_LEN           = 10 * 97;   // Max mel frames to buffer (~10 seconds)
    private static final int FEATURE_BUFFER_MAX_LEN     = 120;       // Max feature frames to buffer
    private static final int MEL_WINDOW_SIZE            = 76;
    private static final int MEL_WINDOW_STRIDE          = 8;

    private final ONNXModelRunner modelRunner;

    private float[][] featureBuffer;
    private float[][] melspectrogramBuffer;
    private int accumulatedSamples = 0;

    private final Deque<Float> rawDataBuffer = new ArrayDeque<>(SAMPLE_RATE * 10);
    private float[] rawDataRemainder = new float[0];

    public WakeWordDetector(ONNXModelRunner modelRunner) {
        this.modelRunner = modelRunner;
        this.melspectrogramBuffer = new float[MEL_WINDOW_SIZE][32];
        for (float[] row : melspectrogramBuffer) {
            Arrays.fill(row, 1.0f); // Initialize with ones
        }

        try {
            this.featureBuffer = extractEmbeddingFeatures(generateDummyAudio(), MEL_WINDOW_SIZE, MEL_WINDOW_STRIDE);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private float[] generateDummyAudio() {
        float[] arr = new float[SAMPLE_RATE * 4];
        Random random = new Random();
        for (int i = 0; i < arr.length; i++) {
            arr[i] = random.nextInt(2000) - 1000f;
        }
        return arr;
    }

    public float[][][] getLatestFeatures(int nFrames, int startIndex) {
        if (featureBuffer == null || featureBuffer.length == 0) return null;

        int endIndex;
        if (startIndex != -1) {
            endIndex = startIndex + nFrames;
        } else {
            endIndex = featureBuffer.length;
            startIndex = Math.max(0, endIndex - nFrames);
        }

        int length = endIndex - startIndex;
        float[][][] result = new float[1][length][featureBuffer[0].length];
        for (int i = 0; i < length; i++) {
            System.arraycopy(featureBuffer[startIndex + i], 0, result[0][i], 0, featureBuffer[0].length);
        }
        return result;
    }

    public void bufferRawData(float[] input) {
        while (rawDataBuffer.size() + input.length > SAMPLE_RATE * 10) {
            rawDataBuffer.poll();
        }
        for (float v : input) rawDataBuffer.offer(v);
    }

    public void updateMelSpectrogram(int nSamples) {
        if (rawDataBuffer.size() < 400) {
            throw new IllegalArgumentException("At least 400 samples required for mel spectrogram generation.");
        }

        float[] temp = new float[nSamples + 480];
        Object[] bufferArray = rawDataBuffer.toArray();
        int offset = Math.max(0, bufferArray.length - nSamples - 480);
        for (int i = offset; i < bufferArray.length; i++) {
            temp[i - offset] = (Float) bufferArray[i];
        }

        float[][] newMel;
        try {
            newMel = modelRunner.getMelSpectrogram(temp);
        } catch (Exception e) {
            throw new RuntimeException("Mel spectrogram extraction failed", e);
        }

        float[][] combined = new float[melspectrogramBuffer.length + newMel.length][];
        System.arraycopy(melspectrogramBuffer, 0, combined, 0, melspectrogramBuffer.length);
        System.arraycopy(newMel, 0, combined, melspectrogramBuffer.length, newMel.length);
        melspectrogramBuffer = combined;

        if (melspectrogramBuffer.length > MEL_SPEC_MAX_LEN) {
            float[][] trimmed = new float[MEL_SPEC_MAX_LEN][];
            System.arraycopy(melspectrogramBuffer, melspectrogramBuffer.length - MEL_SPEC_MAX_LEN, trimmed, 0, MEL_SPEC_MAX_LEN);
            melspectrogramBuffer = trimmed;
        }
    }

    public int processStreamingAudio(float[] audioBuffer) {
        if (rawDataRemainder.length > 0) {
            float[] combined = new float[rawDataRemainder.length + audioBuffer.length];
            System.arraycopy(rawDataRemainder, 0, combined, 0, rawDataRemainder.length);
            System.arraycopy(audioBuffer, 0, combined, rawDataRemainder.length, audioBuffer.length);
            audioBuffer = combined;
            rawDataRemainder = new float[0];
        }

        int remainder = (accumulatedSamples + audioBuffer.length) % FRAME_SIZE;
        if (accumulatedSamples + audioBuffer.length >= FRAME_SIZE) {
            float[] evenChunks = Arrays.copyOf(audioBuffer, audioBuffer.length - remainder);
            bufferRawData(evenChunks);
            accumulatedSamples += evenChunks.length;
            rawDataRemainder = Arrays.copyOfRange(audioBuffer, audioBuffer.length - remainder, audioBuffer.length);
        } else {
            bufferRawData(audioBuffer);
            accumulatedSamples += audioBuffer.length;
        }

        if (accumulatedSamples >= FRAME_SIZE && accumulatedSamples % FRAME_SIZE == 0) {
            updateMelSpectrogram(accumulatedSamples);
            for (int i = accumulatedSamples / FRAME_SIZE - 1; i >= 0; i--) {
                int ndx = melspectrogramBuffer.length - 8 * (accumulatedSamples / FRAME_SIZE - i);
                int start = Math.max(0, ndx - MEL_WINDOW_SIZE);
                int end = ndx;

                float[][][][] x = new float[1][MEL_WINDOW_SIZE][32][1];
                for (int j = start, k = 0; j < end && k < MEL_WINDOW_SIZE; j++, k++) {
                    for (int w = 0; w < 32; w++) {
                        x[0][k][w][0] = melspectrogramBuffer[j][w];
                    }
                }

                try {
                    float[][] newFeatures = modelRunner.generateEmbeddings(x);
                    featureBuffer = mergeFeatureBuffers(featureBuffer, newFeatures);
                } catch (Exception e) {
                    throw new RuntimeException("Failed to generate embeddings", e);
                }
            }
            accumulatedSamples = 0;
        }

        if (featureBuffer.length > FEATURE_BUFFER_MAX_LEN) {
            float[][] trimmed = new float[FEATURE_BUFFER_MAX_LEN][featureBuffer[0].length];
            for (int i = 0; i < FEATURE_BUFFER_MAX_LEN; i++) {
                trimmed[i] = featureBuffer[featureBuffer.length - FEATURE_BUFFER_MAX_LEN + i];
            }
            featureBuffer = trimmed;
        }

        return accumulatedSamples;
    }

    private float[][] mergeFeatureBuffers(float[][] base, float[][] newData) {
        if (base == null) return newData;
        float[][] result = new float[base.length + newData.length][base[0].length];
        System.arraycopy(base, 0, result, 0, base.length);
        System.arraycopy(newData, 0, result, base.length, newData.length);
        return result;
    }

    private float[][] extractEmbeddingFeatures(float[] audio, int windowSize, int stepSize) throws OrtException, IOException {
        float[][] spec = modelRunner.getMelSpectrogram(audio);
        List<float[][]> windows = new ArrayList<>();

        for (int i = 0; i <= spec.length - windowSize; i += stepSize) {
            float[][] window = new float[windowSize][spec[0].length];
            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(spec[i + j], 0, window[j], 0, spec[0].length);
            }
            windows.add(window);
        }

        float[][][][] batch = new float[windows.size()][windowSize][spec[0].length][1];
        for (int i = 0; i < windows.size(); i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int k = 0; k < spec[0].length; k++) {
                    batch[i][j][k][0] = windows.get(i)[j][k];
                }
            }
        }

        return modelRunner.generateEmbeddings(batch);
    }

    public String predictWakeWord(float[] audioBuffer) {
        processStreamingAudio(audioBuffer);
        float[][][] x = getLatestFeatures(16, -1);
        try {
            return modelRunner.predictWakeWord(x);
        } catch (OrtException e) {
            throw new RuntimeException("ONNX inference failed", e);
        }
    }
}
