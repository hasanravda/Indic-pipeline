package com.example.indicpipeline;

import android.content.Context;
import ai.onnxruntime.*;
import org.json.JSONObject;
import java.io.*;
import java.nio.LongBuffer;
import java.util.*;

public class OfflineTranslator {
    private OrtEnvironment env; // Shared
    private OrtSession encoderSession, decoderSession, lmHeadSession;
    private SentencePieceTokenizer tokenizer;
    private Map<String, Long> srcVocab;
    private Map<Long, String> tgtVocab;

    private static final String TRANS_DIR = "trans";

    public OfflineTranslator(Context context, OrtEnvironment sharedEnv) throws Exception {
        env = sharedEnv; // USING SHARED ENV HERE
        tokenizer = new SentencePieceTokenizer();

        String tokenizerPath = assetToCache(context, TRANS_DIR + "/tokenizer.model");
        String encoderPath = assetToCache(context, TRANS_DIR + "/encoder_quant.onnx");
        String decoderPath = assetToCache(context, TRANS_DIR + "/decoder_quant.onnx");
        String lmPath = assetToCache(context, TRANS_DIR + "/lm_head_quant.onnx");

        srcVocab = loadVocab(context, TRANS_DIR + "/dict.SRC.json");
        tgtVocab = loadReverseVocab(context, TRANS_DIR + "/dict.TGT.json");

        tokenizer.loadModel(tokenizerPath);
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);
        lmHeadSession = env.createSession(lmPath, options);
    }

    private Map<String, Long> loadVocab(Context context, String fileName) throws Exception {
        Map<String, Long> vocab = new HashMap<>();
        try (InputStream is = context.getAssets().open(fileName);
             Scanner scanner = new Scanner(is, "UTF-8").useDelimiter("\\A")) {
            JSONObject json = new JSONObject(scanner.hasNext() ? scanner.next() : "");
            Iterator<String> keys = json.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                vocab.put(key, json.getLong(key));
            }
        }
        return vocab;
    }

    private Map<Long, String> loadReverseVocab(Context context, String fileName) throws Exception {
        Map<Long, String> vocab = new HashMap<>();
        try (InputStream is = context.getAssets().open(fileName);
             Scanner scanner = new Scanner(is, "UTF-8").useDelimiter("\\A")) {
            JSONObject json = new JSONObject(scanner.hasNext() ? scanner.next() : "");
            Iterator<String> keys = json.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                vocab.put(json.getLong(key), key);
            }
        }
        return vocab;
    }

    private String assetToCache(Context context, String assetPath) throws IOException {
        // Use only the basename for the cache file so subdirectories in asset
        // paths (e.g. "trans/tokenizer.model") don't break FileOutputStream.
        int slash = assetPath.lastIndexOf('/');
        String cacheName = (slash >= 0) ? assetPath.substring(slash + 1) : assetPath;

        File file = new File(context.getCacheDir(), cacheName);
        if (!file.exists()) {
            try (InputStream is = context.getAssets().open(assetPath);
                 FileOutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = is.read(buffer)) != -1) os.write(buffer, 0, read);
            }
        }
        return file.getAbsolutePath();
    }

    public String translate(String text, String srcLang, String tgtLang) throws Exception {
        String unifiedText = ScriptConverter.convertToDevanagari(text, srcLang);
        String[] pieces = tokenizer.tokenizeAsPieces(unifiedText);

        long[] inputIds = new long[pieces.length + 3];
        long unkId = srcVocab.getOrDefault("<unk>", 3L);

        inputIds[0] = srcVocab.getOrDefault(srcLang, unkId);
        inputIds[1] = srcVocab.getOrDefault(tgtLang, unkId);
        for (int i = 0; i < pieces.length; i++) inputIds[i + 2] = srcVocab.getOrDefault(pieces[i], unkId);
        inputIds[inputIds.length - 1] = 2L;

        long[] attentionMask = new long[inputIds.length];
        Arrays.fill(attentionMask, 1L);

        long[] shape = {1, inputIds.length};
        OnnxTensor tInput = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape);
        OnnxTensor tMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape);

        Map<String, OnnxTensor> encInputs = new HashMap<>();
        encInputs.put("input_ids", tInput);
        encInputs.put("attention_mask", tMask);

        float[][][] encoderHidden;
        try (OrtSession.Result encOut = encoderSession.run(encInputs)) {
            encoderHidden = (float[][][]) encOut.get(0).getValue();
        } finally { tInput.close(); tMask.close(); }

        List<Long> generated = new ArrayList<>();
        generated.add(2L);

        for (int i = 0; i < 256; i++) {
            long[] curInput = new long[generated.size()];
            for (int j = 0; j < generated.size(); j++) curInput[j] = generated.get(j);

            OnnxTensor decInput = OnnxTensor.createTensor(env, new long[][]{curInput});
            OnnxTensor encHiddenTensor = OnnxTensor.createTensor(env, encoderHidden);
            OnnxTensor encMaskTensor = OnnxTensor.createTensor(env, new long[][]{attentionMask});

            Map<String, OnnxTensor> decInputs = new HashMap<>();
            decInputs.put("decoder_input_ids", decInput);
            decInputs.put("encoder_hidden_states", encHiddenTensor);
            decInputs.put("encoder_attention_mask", encMaskTensor);

            try (OrtSession.Result decOut = decoderSession.run(decInputs)) {
                float[][][] dHidden = (float[][][]) decOut.get(0).getValue();
                float[][] lastToken = { dHidden[0][dHidden[0].length - 1] };

                OnnxTensor lmInput = OnnxTensor.createTensor(env, new float[][][]{lastToken});
                try (OrtSession.Result lmOut = lmHeadSession.run(Collections.singletonMap("decoder_hidden_states", lmInput))) {
                    float[][][] logits = (float[][][]) lmOut.get(0).getValue();
                    int nextToken = argmax(logits[0][0]);
                    if (nextToken == 2 && generated.size() > 1) break;
                    generated.add((long) nextToken);
                } finally { lmInput.close(); }
            } finally {
                decInput.close(); encHiddenTensor.close(); encMaskTensor.close();
            }
        }

        String[] outPieces = new String[generated.size() - 1];
        for (int i = 1; i < generated.size(); i++) {
            outPieces[i - 1] = tgtVocab.getOrDefault(generated.get(i), "<unk>");
        }

        String decodedDevanagari = tokenizer.decodePieces(outPieces);
        return ScriptConverter.convertFromDevanagari(decodedDevanagari, tgtLang);
    }

    private int argmax(float[] array) {
        int bestIdx = 0;
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxVal) { maxVal = array[i]; bestIdx = i; }
        }
        return bestIdx;
    }
}