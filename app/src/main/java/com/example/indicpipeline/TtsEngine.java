package com.example.indicpipeline;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import ai.onnxruntime.OrtEnvironment;
import com.k2fsa.sherpa.onnx.GeneratedAudio;
import com.k2fsa.sherpa.onnx.OfflineTts;
import com.k2fsa.sherpa.onnx.OfflineTtsConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig;

/**
 * TTS engine using sherpa-onnx VITS models (same approach as the standalone
 * ttsapp). Supported languages: Bengali ("ben"), Gujarati ("guj"), Hindi ("hin").
 * Models live in assets/tts/, espeak-ng-data is copied to filesDir once.
 */
public class TtsEngine {
    private static final String TTS_DIR = "tts";
    private static final String ESPEAK_DATA_DIR = "espeak-ng-data";

    private OfflineTts sherpaTts;

    public TtsEngine(Context context, OrtEnvironment unusedSharedEnv, String folderCode) throws Exception {
        String langFolder = ModelAvailability.ttsAssetFolder(folderCode);
        String modelName = ModelAvailability.ttsModelFileName(folderCode);
        if (langFolder == null || modelName == null) {
            throw new IllegalArgumentException("Unsupported TTS language: " + folderCode);
        }

        String dataDir = copyEspeakDataIfNeeded(context);

        OfflineTtsVitsModelConfig vits = new OfflineTtsVitsModelConfig();
        vits.setModel(TTS_DIR + "/" + langFolder + "/" + modelName);
        vits.setTokens(TTS_DIR + "/tokens.txt");
        vits.setDataDir(dataDir);

        OfflineTtsModelConfig modelConfig = new OfflineTtsModelConfig();
        modelConfig.setVits(vits);
        modelConfig.setNumThreads(2);
        modelConfig.setDebug(true);
        modelConfig.setProvider("cpu");

        OfflineTtsConfig config = new OfflineTtsConfig();
        config.setModel(modelConfig);

        this.sherpaTts = new OfflineTts(context.getAssets(), config);
    }

    private String copyEspeakDataIfNeeded(Context context) throws Exception {
        File targetRoot = new File(context.getFilesDir(), ESPEAK_DATA_DIR);
        File marker = new File(context.getFilesDir(), ".espeak_copied_v1");
        if (targetRoot.exists() && marker.exists()) {
            return targetRoot.getAbsolutePath();
        }

        copyAssetDir(context, ESPEAK_DATA_DIR, targetRoot);
        if (!marker.exists()) {
            new FileOutputStream(marker).close();
        }
        return targetRoot.getAbsolutePath();
    }

    private void copyAssetDir(Context context, String assetPath, File destDir) throws Exception {
        String[] children = context.getAssets().list(assetPath);
        if (children == null || children.length == 0) {
            copyAssetFile(context, assetPath, destDir);
            return;
        }
        if (!destDir.exists() && !destDir.mkdirs()) {
            throw new Exception("Cannot create directory: " + destDir);
        }
        for (String child : children) {
            copyAssetDir(context, assetPath + "/" + child, new File(destDir, child));
        }
    }

    private void copyAssetFile(Context context, String assetPath, File destFile) throws Exception {
        File parent = destFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new Exception("Cannot create directory: " + parent);
        }
        try (InputStream in = context.getAssets().open(assetPath);
             OutputStream out = new FileOutputStream(destFile)) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = in.read(buffer)) != -1) {
                out.write(buffer, 0, n);
            }
        }
    }

    /** Synthesizes and plays the text; returns the model processing time in ms. */
    public long speak(String text) throws Exception {
        if (sherpaTts == null) return 0;

        long startTime = System.currentTimeMillis();
        GeneratedAudio audio = sherpaTts.generate(text, 0, 1.0f);
        long generationTime = System.currentTimeMillis() - startTime;

        if (audio != null && audio.getSamples() != null) {
            playAudio(audio.getSamples(), audio.getSampleRate());
        }
        return generationTime;
    }

    private void playAudio(float[] audioData, int sampleRate) {
        AudioTrack audioTrack = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA) // Ensures it plays loud out of main speaker
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build())
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build())
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(AudioTrack.getMinBufferSize(
                        sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT))
                .build();

        audioTrack.play();
        audioTrack.write(audioData, 0, audioData.length, AudioTrack.WRITE_BLOCKING);
        audioTrack.release();
    }

    public void close() {
        if (sherpaTts != null) {
            sherpaTts.release();
            sherpaTts = null;
        }
    }

    // --- BATCH PROCESSING METHODS --- //

    public static class TtsResult {
        public long timeMs;
        public float[] audioData;
        public TtsResult(long t, float[] a) { timeMs = t; audioData = a; }
    }

    // Synthesizes audio silently and returns the raw float array to be saved as a file
    public TtsResult synthesizeToFile(String rawText) throws Exception {
        if (sherpaTts == null) return null;
        long startTime = System.currentTimeMillis();
        GeneratedAudio audio = sherpaTts.generate(rawText, 0, 1.0f);
        long generationTime = System.currentTimeMillis() - startTime;
        if (audio == null) return null;
        return new TtsResult(generationTime, audio.getSamples());
    }
}
