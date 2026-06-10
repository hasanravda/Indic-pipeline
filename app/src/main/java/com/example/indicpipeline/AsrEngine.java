package com.example.indicpipeline;

import android.content.Context;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class AsrEngine {
    private static final String TAG = "ASR";
    private static final int SAMPLE_RATE = 16000;
    private static final int N_MELS = 80;
    private static final int FULL_VOCAB = 5633;

    private final Context appCtx;
    private final OrtEnvironment env; // Shared
    private final OrtSession encoder;
    private final OrtSession ctcDecoder;
    private final int blankId;

    public static class Result {
        public final String text;
        public final double ms;
        public Result(String text, double ms) {
            this.text = text;
            this.ms = ms;
        }
    }

    public AsrEngine(Context ctx, OrtEnvironment sharedEnv) throws Exception {
        appCtx = ctx.getApplicationContext();
        env = sharedEnv; // USING SHARED ENV HERE

        File modelDir = AssetUtils.copyAsrAssetsToFiles(appCtx);
        String encoderPath = new File(modelDir, "encoder.onnx").getAbsolutePath();
        String ctcPath = new File(modelDir, "ctc_decoder.onnx").getAbsolutePath();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

        encoder = env.createSession(encoderPath, opts);
        ctcDecoder = env.createSession(ctcPath, opts);

        int tmpBlank = 0;
        try {
            JSONObject cfg = DecodeUtils.readAssetJson(appCtx, "asr/config.json");
            tmpBlank = cfg.optInt("BLANK_ID", 0);
        } catch (Exception e) { tmpBlank = 0; }
        blankId = tmpBlank;
        Log.i(TAG, "AsrEngine init OK. BLANK_ID=" + blankId);
    }

    public Result transcribe(short[] pcm16, String lang) throws Exception {
        long t0 = System.nanoTime();
        float[] wav = pcm16ToFloat(pcm16);
        float[][] featsTx80 = FbankJNI.computeFbank80(wav, SAMPLE_RATE);

        if (featsTx80 == null || featsTx80.length == 0) {
            return new Result("", (System.nanoTime() - t0) / 1e6);
        }

        int T = featsTx80.length;
        float[][] melByT = new float[N_MELS][T];
        for (int t = 0; t < T; t++) {
            for (int m = 0; m < N_MELS; m++) melByT[m][t] = featsTx80[t][m];
        }

        cmvnInPlace(melByT);

        float[] featsFlat = new float[1 * N_MELS * T];
        int idx = 0;
        for (int m = 0; m < N_MELS; m++) {
            for (int t = 0; t < T; t++) featsFlat[idx++] = melByT[m][t];
        }

        ByteBuffer audioBB = ByteBuffer.allocateDirect(featsFlat.length * 4).order(ByteOrder.nativeOrder());
        FloatBuffer audioFB = audioBB.asFloatBuffer();
        audioFB.put(featsFlat);
        audioFB.rewind();

        ByteBuffer lenBB = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
        LongBuffer lenLB = lenBB.asLongBuffer();
        lenLB.put((long) T);
        lenLB.rewind();

        long[] audioShape = new long[]{1, N_MELS, T};
        long[] lenShape = new long[]{1};

        try (OnnxTensor audioTensor = OnnxTensor.createTensor(env, audioFB, audioShape);
             OnnxTensor lenTensor = OnnxTensor.createTensor(env, lenLB, lenShape)) {

            Map<String, OnnxTensor> encIn = new HashMap<>();
            encIn.put("audio_signal", audioTensor);
            encIn.put("length", lenTensor);

            OrtSession.Result encRes = encoder.run(encIn);
            float[][][] encOut = (float[][][]) encRes.get(0).getValue();

            try (OnnxTensor encTensor = OnnxTensor.createTensor(env, encOut)) {
                Map<String, OnnxTensor> ctcIn = new HashMap<>();
                ctcIn.put("encoder_output", encTensor);

                OrtSession.Result ctcRes = ctcDecoder.run(ctcIn);
                float[][][] logits = normalizeLogitsToBTv(ctcRes.get(0).getValue());
                float[][] logitsTFull = logits[0];

                LangPack lp = loadLangPack(lang);
                float[][] logitsLang = applyMask(logitsTFull, lp.maskBool);

                int[] path = argmaxPerFrame(logitsLang);
                int[] collapsed = uniqueConsecutive(path);
                String rawText = tokensToText(collapsed, lp.vocab, blankId);
                String finalText = PipeJoinDecode.decode(rawText);

                ctcRes.close();
                encRes.close();
                return new Result(finalText, (System.nanoTime() - t0) / 1e6);
            }
        }
    }

    private static float[] pcm16ToFloat(short[] pcm) {
        float[] out = new float[pcm.length];
        for (int i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768f;
        return out;
    }

    private static void cmvnInPlace(float[][] melByT) {
        int M = melByT.length;
        int T = melByT[0].length;
        for (int m = 0; m < M; m++) {
            double sum = 0;
            for (int t = 0; t < T; t++) sum += melByT[m][t];
            double mean = sum / T;
            double var = 0;
            for (int t = 0; t < T; t++) {
                double d = melByT[m][t] - mean;
                var += d * d;
            }
            double std = Math.sqrt(var / T);
            if (std < 1e-5) std = 1e-5;
            for (int t = 0; t < T; t++) melByT[m][t] = (float) ((melByT[m][t] - mean) / std);
        }
    }

    private static float[][][] normalizeLogitsToBTv(Object raw) {
        float[][][] x = (float[][][]) raw;
        if (x[0].length == FULL_VOCAB) {
            float[][][] y = new float[x.length][x[0][0].length][FULL_VOCAB];
            for (int b = 0; b < x.length; b++) {
                for (int v = 0; v < FULL_VOCAB; v++) {
                    for (int t = 0; t < x[0][0].length; t++) y[b][t][v] = x[b][v][t];
                }
            }
            return y;
        }
        return x;
    }

    private static float[][] applyMask(float[][] logitsTV, boolean[] maskBool5633) {
        int T = logitsTV.length;
        int outV = 0;
        for (boolean b : maskBool5633) if (b) outV++;
        float[][] out = new float[T][outV];
        for (int t = 0; t < T; t++) {
            int j = 0;
            for (int v = 0; v < maskBool5633.length; v++) {
                if (maskBool5633[v]) out[t][j++] = logitsTV[t][v];
            }
        }
        return out;
    }

    private static int[] argmaxPerFrame(float[][] logitsTV) {
        int[] path = new int[logitsTV.length];
        for (int t = 0; t < logitsTV.length; t++) {
            int best = 0;
            float bestVal = logitsTV[t][0];
            for (int v = 1; v < logitsTV[0].length; v++) {
                if (logitsTV[t][v] > bestVal) {
                    bestVal = logitsTV[t][v];
                    best = v;
                }
            }
            path[t] = best;
        }
        return path;
    }

    private static int[] uniqueConsecutive(int[] arr) {
        if (arr.length == 0) return new int[0];
        int[] tmp = new int[arr.length];
        int n = 0, prev = arr[0];
        tmp[n++] = prev;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] != prev) {
                prev = arr[i];
                tmp[n++] = prev;
            }
        }
        int[] out = new int[n];
        System.arraycopy(tmp, 0, out, 0, n);
        return out;
    }

    private static String tokensToText(int[] ids, String[] vocab, int blankId) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id == blankId) continue;
            if (id >= 0 && id < vocab.length) sb.append(vocab[id]);
        }
        return sb.toString().replace("▁", " ").trim().replaceAll("\\s+", " ");
    }

    private static class LangPack {
        final String[] vocab;
        final boolean[] maskBool;
        LangPack(String[] v, boolean[] m) { vocab = v; maskBool = m; }
    }

    private LangPack loadLangPack(String lang) throws Exception {
        JSONObject vocabJson = DecodeUtils.readAssetJson(appCtx, "asr/vocab.json");
        JSONObject masksJson = DecodeUtils.readAssetJson(appCtx, "asr/language_masks.json");

        JSONArray vArr = vocabJson.getJSONArray(lang);
        String[] v = new String[vArr.length()];
        for (int i = 0; i < vArr.length(); i++) v[i] = vArr.getString(i);

        JSONArray mArr = masksJson.getJSONArray(lang);
        boolean[] mask = new boolean[mArr.length()];
        for (int i = 0; i < mArr.length(); i++) {
            Object o = mArr.get(i);
            if (o instanceof Boolean) mask[i] = (Boolean) o;
            else mask[i] = (mArr.getInt(i) != 0);
        }
        return new LangPack(v, mask);
    }
}