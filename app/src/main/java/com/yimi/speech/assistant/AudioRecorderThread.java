package com.yimi.speech.assistant;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.widget.TextView;
import android.widget.Toast;

public class AudioRecorderThread extends Thread {

    private final String TAG = "AudioRecorderThread";

    private Context context;
    private TextView textView;
    private WakeWordDetector wakeWordDetector;

    private static final int SAMPLE_RATE    = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT   = AudioFormat.ENCODING_PCM_16BIT;

    private AudioRecord audioRecord;

    private boolean isRecording = false;

    private long lastToastTime = 0;

    private static final long TOAST_INTERVAL_MS = 3000;

    AudioRecorderThread(Context context, TextView textView, WakeWordDetector wakeWordDetector) {
        this.context            = context;
        this.textView           = textView;
        this.wakeWordDetector   = wakeWordDetector;
    }

    @SuppressLint("MissingPermission")
    @Override
    public void run() {

        int bufferSizeInShorts = 1280;

        int minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (minBufferSize / 2 < bufferSizeInShorts) {
            minBufferSize = bufferSizeInShorts * 2;
        }

        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, minBufferSize);
        audioRecord.startRecording();
        isRecording = true;

        short[] audioBuffer = new short[bufferSizeInShorts];

        while (isRecording) {
            audioRecord.read(audioBuffer, 0, audioBuffer.length);

            float[] floatBuffer = new float[audioBuffer.length];

            for (int i = 0; i < audioBuffer.length; i++) {
                floatBuffer[i] = audioBuffer[i] / 32768.0f;
            }

            String res = wakeWordDetector.predictWakeWord(floatBuffer);
            if (Double.parseDouble(res) > 0.05) {
                ((Activity) context).runOnUiThread(() -> textView.setText(res));
                if (Double.parseDouble(res) > 0.5) {
                    long currentTime = System.currentTimeMillis();
                    if (currentTime - lastToastTime > TOAST_INTERVAL_MS) {
                        lastToastTime = currentTime;
                        ((Activity) context).runOnUiThread(() ->
                                Toast.makeText(context, "Wake word detected!", Toast.LENGTH_SHORT).show()
                        );
                    }
                }
            } else {
                ((Activity) context).runOnUiThread(() -> textView.setText("0.00000"));
            }
        }
    }
}
