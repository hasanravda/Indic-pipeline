package com.example.indicpipeline;

import android.content.Context;
import android.util.Log;
import org.json.JSONObject;
import org.vosk.Model;
import org.vosk.Recognizer;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import ai.onnxruntime.OrtEnvironment;
import com.k2fsa.sherpa.onnx.FeatureConfig;
import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;

/**
 * ASR engine with two backends (same approach as the standalone asrapp):
 *  - Hindi ("hi") and Gujarati ("gu"): Vosk (Kaldi) small models
 *  - Bengali ("bn"): sherpa-onnx streaming Zipformer2 transducer
 *
 * Only one language model is kept in memory at a time; call
 * {@link #setLanguage(String)} to (pre)load a language.
 */
public class AsrEngine {
    private static final String TAG = "ASR";
    private static final int SAMPLE_RATE = 16000;

    private static final String ASR_DIR = "asr";
    private static final String VOSK_HINDI = ASR_DIR + "/hindi";
    private static final String VOSK_GUJARATI = ASR_DIR + "/gujarati";
    private static final String SHERPA_BENGALI = ASR_DIR + "/bengali";

    private final Context appCtx;

    private String currentLang = null;
    private Model voskModel;
    private OnlineRecognizer sherpaRecognizer;

    public static class Result {
        public final String text;
        public final double ms;
        public Result(String text, double ms) {
            this.text = text;
            this.ms = ms;
        }
    }

    public AsrEngine(Context ctx, OrtEnvironment unusedSharedEnv) {
        appCtx = ctx.getApplicationContext();
        Log.i(TAG, "AsrEngine created (models load lazily per language)");
    }

    /** Loads the model for the given language ("hi", "gu", "bn"), releasing the previous one. */
    public synchronized void setLanguage(String lang) throws Exception {
        if (lang.equals(currentLang)) return;

        releaseModels();
        long t0 = System.currentTimeMillis();

        switch (lang) {
            case "hi":
                voskModel = loadVoskModel(VOSK_HINDI, "model-hi");
                break;
            case "gu":
                voskModel = loadVoskModel(VOSK_GUJARATI, "model-gu");
                break;
            case "bn":
                sherpaRecognizer = loadSherpaRecognizer();
                break;
            default:
                throw new IllegalArgumentException("Unsupported ASR language: " + lang);
        }

        currentLang = lang;
        Log.i(TAG, "Loaded ASR model for '" + lang + "' in " + (System.currentTimeMillis() - t0) + " ms");
    }

    public Result transcribe(short[] pcm16, String lang) throws Exception {
        long t0 = System.nanoTime();
        setLanguage(lang);

        String text;
        if ("bn".equals(lang)) {
            text = transcribeSherpa(pcm16);
        } else {
            text = transcribeVosk(pcm16);
        }
        return new Result(text, (System.nanoTime() - t0) / 1e6);
    }

    public synchronized void releaseModels() {
        if (voskModel != null) {
            voskModel.close();
            voskModel = null;
        }
        if (sherpaRecognizer != null) {
            sherpaRecognizer.release();
            sherpaRecognizer = null;
        }
        currentLang = null;
    }

    // ---------- Vosk (Hindi / Gujarati) ----------

    private Model loadVoskModel(String assetDir, String targetDirName) throws Exception {
        File targetDir = syncVoskAssets(assetDir, targetDirName);
        return new Model(targetDir.getAbsolutePath());
    }

    /**
     * Vosk needs real filesystem paths, so the model is copied from assets to
     * filesDir once. The model's "uuid" file is used to detect stale copies.
     */
    private File syncVoskAssets(String assetDir, String targetDirName) throws Exception {
        File targetDir = new File(appCtx.getFilesDir(), targetDirName);
        File uuidFile = new File(targetDir, "uuid");

        String assetUuid = readAssetText(assetDir + "/uuid");
        if (uuidFile.exists() && assetUuid.equals(readFileText(uuidFile))) {
            return targetDir; // already unpacked and up to date
        }

        deleteRecursively(targetDir);
        copyAssetDir(assetDir, targetDir);
        return targetDir;
    }

    private String transcribeVosk(short[] pcm16) throws Exception {
        Model model = voskModel;
        if (model == null) return "";

        byte[] pcm = shortsToBytes(pcm16);
        Recognizer rec = new Recognizer(model, (float) SAMPLE_RATE);
        try {
            StringBuilder sb = new StringBuilder();
            int chunk = 8000; // ~0.25s of 16-bit/16kHz audio per accept call
            int offset = 0;
            while (offset < pcm.length) {
                int len = Math.min(chunk, pcm.length - offset);
                byte[] slice = new byte[len];
                System.arraycopy(pcm, offset, slice, 0, len);
                if (rec.acceptWaveForm(slice, len)) {
                    String piece = new JSONObject(rec.getResult()).optString("text", "");
                    if (!piece.isEmpty()) sb.append(piece).append(' ');
                }
                offset += len;
            }
            String finalPiece = new JSONObject(rec.getFinalResult()).optString("text", "");
            if (!finalPiece.isEmpty()) sb.append(finalPiece);
            return sb.toString().trim();
        } finally {
            rec.close();
        }
    }

    // ---------- sherpa-onnx (Bengali) ----------

    private OnlineRecognizer loadSherpaRecognizer() {
        FeatureConfig feat = new FeatureConfig();
        feat.setSampleRate(SAMPLE_RATE);
        feat.setFeatureDim(80);

        OnlineTransducerModelConfig transducer = new OnlineTransducerModelConfig();
        transducer.setEncoder(SHERPA_BENGALI + "/encoder.onnx");
        transducer.setDecoder(SHERPA_BENGALI + "/decoder.onnx");
        transducer.setJoiner(SHERPA_BENGALI + "/joiner.onnx");

        OnlineModelConfig model = new OnlineModelConfig();
        model.setTransducer(transducer);
        model.setTokens(SHERPA_BENGALI + "/tokens.txt");
        model.setNumThreads(2);
        model.setModelType("zipformer2");

        OnlineRecognizerConfig config = new OnlineRecognizerConfig();
        config.setFeatConfig(feat);
        config.setModelConfig(model);
        config.setDecodingMethod("greedy_search");
        config.setEnableEndpoint(false);

        return new OnlineRecognizer(appCtx.getAssets(), config);
    }

    private String transcribeSherpa(short[] pcm16) {
        OnlineRecognizer recognizer = sherpaRecognizer;
        if (recognizer == null) return "";

        float[] samples = shortsToFloats(pcm16);
        OnlineStream stream = recognizer.createStream("");
        try {
            stream.acceptWaveform(samples, SAMPLE_RATE);
            // Tail padding so the decoder flushes the last words
            stream.acceptWaveform(new float[SAMPLE_RATE], SAMPLE_RATE);
            stream.inputFinished();
            while (recognizer.isReady(stream)) {
                recognizer.decode(stream);
            }
            return recognizer.getResult(stream).getText().trim();
        } finally {
            stream.release();
        }
    }

    // ---------- helpers ----------

    private static byte[] shortsToBytes(short[] pcm16) {
        byte[] out = new byte[pcm16.length * 2];
        for (int i = 0; i < pcm16.length; i++) {
            out[i * 2] = (byte) (pcm16[i] & 0xff);
            out[i * 2 + 1] = (byte) ((pcm16[i] >> 8) & 0xff);
        }
        return out;
    }

    private static float[] shortsToFloats(short[] pcm16) {
        float[] out = new float[pcm16.length];
        for (int i = 0; i < pcm16.length; i++) {
            out[i] = pcm16[i] / 32768.0f;
        }
        return out;
    }

    private String readAssetText(String assetPath) throws Exception {
        try (InputStream in = appCtx.getAssets().open(assetPath)) {
            return streamToString(in);
        }
    }

    private static String readFileText(File file) throws Exception {
        try (InputStream in = new java.io.FileInputStream(file)) {
            return streamToString(in);
        }
    }

    private static String streamToString(InputStream in) throws Exception {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = in.read(buf)) != -1) bos.write(buf, 0, n);
        return bos.toString("UTF-8").trim();
    }

    private void copyAssetDir(String assetPath, File destDir) throws Exception {
        String[] children = appCtx.getAssets().list(assetPath);
        if (children == null || children.length == 0) {
            copyAssetFile(assetPath, destDir);
            return;
        }
        if (!destDir.exists() && !destDir.mkdirs()) {
            throw new Exception("Cannot create directory: " + destDir);
        }
        for (String child : children) {
            copyAssetDir(assetPath + "/" + child, new File(destDir, child));
        }
    }

    private void copyAssetFile(String assetPath, File destFile) throws Exception {
        File parent = destFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new Exception("Cannot create directory: " + parent);
        }
        try (InputStream in = appCtx.getAssets().open(assetPath);
             OutputStream out = new FileOutputStream(destFile)) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
        }
    }

    private static void deleteRecursively(File f) {
        if (f.isDirectory()) {
            File[] children = f.listFiles();
            if (children != null) {
                for (File c : children) deleteRecursively(c);
            }
        }
        f.delete();
    }
}
