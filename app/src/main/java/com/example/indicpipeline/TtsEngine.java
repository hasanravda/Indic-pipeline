package com.example.indicpipeline;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.k2fsa.sherpa.onnx.GeneratedAudio;
import com.k2fsa.sherpa.onnx.OfflineTts;
import com.k2fsa.sherpa.onnx.OfflineTtsConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig;

public class TtsEngine {
    private static final String TTS_DIR = "tts";
    private static final String TTS_CUSTOM_DIR = "tts_custom";
    private static final String ESPEAK_DATA_DIR = "espeak-ng-data";

    private final OrtEnvironment env;
    private OrtSession session;
    private OfflineTts sherpaTts;
    private final Map<Character, Long> vocab = new HashMap<>();
    private final boolean isCustom;

    public TtsEngine(Context context, OrtEnvironment sharedEnv, String folderCode) throws Exception {
        // Use an isolated environment if one isn't provided
        if (sharedEnv == null) {
            this.env = OrtEnvironment.getEnvironment();
        } else {
            this.env = sharedEnv;
        }

        // Check if we should use the new custom Sherpa-ONNX TTS
        if (folderCode.equals("ben") || folderCode.equals("guj") || folderCode.equals("hin")) {
            this.isCustom = true;
            setupSherpaTts(context, folderCode);
        } else {
            this.isCustom = false;
            setupGenericTts(context, folderCode);
        }
    }

    private void setupSherpaTts(Context context, String folderCode) throws Exception {
        String modelName;
        switch (folderCode) {
            case "ben": modelName = "bengali_custom.onnx"; break;
            case "guj": modelName = "gujarati_custom.onnx"; break;
            case "hin": modelName = "hindi_custom.onnx"; break;
            default: throw new IllegalArgumentException("Unsupported custom language: " + folderCode);
        }

        String dataDir = copyEspeakDataIfNeeded(context);
        
        OfflineTtsVitsModelConfig vits = new OfflineTtsVitsModelConfig();
        // Sherpa expects the model path relative to assets if using AssetManager, 
        // but here it might be better to provide the full path if it's in a custom subfolder.
        // Actually, looking at the user's snippet: vits.setModel(MODEL_FILES[index]); 
        // and OfflineTts engine = new OfflineTts(getAssets(), config);
        // This means it looks in assets.
        vits.setModel(TTS_CUSTOM_DIR + "/" + modelName);
        vits.setTokens(TTS_CUSTOM_DIR + "/tokens.txt");
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

    private void setupGenericTts(Context context, String folderCode) throws Exception {
        String langDir = TTS_DIR + "/" + folderCode;

        // Exactly matching your standalone app's vocab loading
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(langDir + "/tokens.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(" ");
                if (parts.length == 2) {
                    String token = parts[0];
                    long id = Long.parseLong(parts[1]);
                    // The crucial space mapping rule
                    if (token.equals("|")) {
                        vocab.put(' ', id);
                    } else if (token.length() == 1) {
                        vocab.put(token.charAt(0), id);
                    }
                }
            }
        }

        File tempModelFile = new File(context.getCacheDir(), "tts_model.onnx");
        try (InputStream is = context.getAssets().open(langDir + "/model.onnx");
             FileOutputStream fos = new FileOutputStream(tempModelFile)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = is.read(buffer)) != -1) {
                fos.write(buffer, 0, read);
            }
        }

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
        session = env.createSession(tempModelFile.getAbsolutePath(), options);
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

    // CHANGED TO 'long' to return the processing time to MainActivity
    public long speak(String uromanText) throws Exception {
        if (isCustom) {
            return speakSherpa(uromanText);
        }
        // 1. Clean the text from any invisible formatting from Translator
        String cleanText = uromanText.replaceAll("[\\n\\t\\r]", " ")
                .replaceAll("\\s+", " ")
                .trim()
                .toLowerCase();

        Log.e("TTS_DEBUG", "FINAL TEXT FED TO TTS: [" + cleanText + "]");

        List<Long> ids = new ArrayList<>();
        for (char c : cleanText.toCharArray()) {
            if (vocab.containsKey(c)) {
                ids.add(vocab.get(c));
            } else {
                // Critical fallback to space token if character is missing
                ids.add(vocab.containsKey(' ') ? vocab.get(' ') : 0L);
            }
        }

        List<Long> interspersed = new ArrayList<>();
        interspersed.add(0L);
        for (long id : ids) {
            interspersed.add(id);
            interspersed.add(0L);
        }

        long[] inputArray = new long[interspersed.size()];
        for (int i = 0; i < interspersed.size(); i++) {
            inputArray[i] = interspersed.get(i);
        }

        if (env == null || session == null) return 0;

        long[] shape = new long[]{1, inputArray.length};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputArray), shape);
        OnnxTensor lengthTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{inputArray.length}), new long[]{1});
        OnnxTensor noiseScale = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{0.667f}), new long[]{1});
        OnnxTensor lengthScale = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{1.0f}), new long[]{1});
        OnnxTensor noiseScaleW = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{0.8f}), new long[]{1});

        // 2. EXPLICIT HARDCODED NAMES (Avoid random array shuffling)
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("x", inputTensor);
        inputs.put("x_length", lengthTensor);
        inputs.put("noise_scale", noiseScale);
        inputs.put("length_scale", lengthScale);
        inputs.put("noise_scale_w", noiseScaleW);

        // --- START AI TIMER ---
        long startTime = System.currentTimeMillis();

        OrtSession.Result result = session.run(inputs);

        OnnxTensor rawOutput = (OnnxTensor) result.get(0);
        FloatBuffer floatBuffer = rawOutput.getFloatBuffer();
        float[] audioArray = new float[floatBuffer.remaining()];
        floatBuffer.get(audioArray);

        // --- END AI TIMER ---
        long generationTime = System.currentTimeMillis() - startTime;

        // Play audio AFTER the timer stops
        playAudio(audioArray);

        result.close();
        inputTensor.close();
        lengthTensor.close();
        noiseScale.close();
        lengthScale.close();
        noiseScaleW.close();

        // Return the elapsed time back to MainActivity
        return generationTime;
    }

    private long speakSherpa(String text) throws Exception {
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

    private void playAudio(float[] audioData) {
        playAudio(audioData, 16000);
    }

    public void close() throws Exception {
        if (session != null) {
            session.close();
        }
        if (sherpaTts != null) {
            sherpaTts.release();
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
        if (isCustom) {
            if (sherpaTts == null) return null;
            long startTime = System.currentTimeMillis();
            GeneratedAudio audio = sherpaTts.generate(rawText, 0, 1.0f);
            long generationTime = System.currentTimeMillis() - startTime;
            if (audio == null) return null;
            return new TtsResult(generationTime, audio.getSamples());
        }
        String cleanText = rawText.replaceAll("[\\n\\t\\r]", " ").replaceAll("\\s+", " ").trim().toLowerCase();

        List<Long> ids = new ArrayList<>();
        for (char c : cleanText.toCharArray()) {
            if (vocab.containsKey(c)) ids.add(vocab.get(c));
            else ids.add(vocab.containsKey(' ') ? vocab.get(' ') : 0L);
        }

        List<Long> interspersed = new ArrayList<>();
        interspersed.add(0L);
        for (long id : ids) {
            interspersed.add(id);
            interspersed.add(0L);
        }

        long[] inputArray = new long[interspersed.size()];
        for (int i = 0; i < interspersed.size(); i++) inputArray[i] = interspersed.get(i);

        if (env == null || session == null) return null;

        long[] shape = new long[]{1, inputArray.length};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputArray), shape);
        OnnxTensor lengthTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{inputArray.length}), new long[]{1});
        OnnxTensor noiseScale = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{0.667f}), new long[]{1});
        OnnxTensor lengthScale = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{1.0f}), new long[]{1});
        OnnxTensor noiseScaleW = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{0.8f}), new long[]{1});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("x", inputTensor);
        inputs.put("x_length", lengthTensor);
        inputs.put("noise_scale", noiseScale);
        inputs.put("length_scale", lengthScale);
        inputs.put("noise_scale_w", noiseScaleW);

        // TIMER
        long startTime = System.currentTimeMillis();
        OrtSession.Result result = session.run(inputs);
        long generationTime = System.currentTimeMillis() - startTime;

        OnnxTensor rawOutput = (OnnxTensor) result.get(0);
        FloatBuffer floatBuffer = rawOutput.getFloatBuffer();
        float[] audioArray = new float[floatBuffer.remaining()];
        floatBuffer.get(audioArray);

        result.close();
        inputTensor.close(); lengthTensor.close(); noiseScale.close(); lengthScale.close(); noiseScaleW.close();

        // RETURN THE DATA (Do NOT call playAudio here)
        return new TtsResult(generationTime, audioArray);
    }
}