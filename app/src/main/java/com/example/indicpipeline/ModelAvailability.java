package com.example.indicpipeline;

import android.content.Context;
import android.content.res.AssetManager;
import java.io.IOException;

/** Checks which language models are actually present under assets/asr and assets/tts. */
public final class ModelAvailability {
    private static final String ASR_ROOT = "asr";
    private static final String TTS_ROOT = "tts";

    private ModelAvailability() {}

    public static String asrAssetFolder(String asrCode) {
        switch (asrCode) {
            case "hi": return "hindi";
            case "gu": return "gujarati";
            case "bn": return "bengali";
            default: return null;
        }
    }

    public static String ttsAssetFolder(String ttsFolderCode) {
        switch (ttsFolderCode) {
            case "hin": return "hindi";
            case "guj": return "gujarati";
            case "ben": return "bengali";
            default: return null;
        }
    }

    public static String ttsModelFileName(String ttsFolderCode) {
        switch (ttsFolderCode) {
            case "hin": return "hindi_custom.onnx";
            case "guj": return "gujarati_custom.onnx";
            case "ben": return "bengali_custom.onnx";
            default: return null;
        }
    }

    public static boolean hasAsrModel(Context context, String asrCode) {
        String folder = asrAssetFolder(asrCode);
        if (folder == null) return false;

        String base = ASR_ROOT + "/" + folder;
        if ("bn".equals(asrCode)) {
            return assetExists(context, base + "/encoder.onnx")
                    && assetExists(context, base + "/decoder.onnx")
                    && assetExists(context, base + "/joiner.onnx")
                    && assetExists(context, base + "/tokens.txt");
        }
        // Vosk models
        return assetExists(context, base + "/uuid")
                && assetExists(context, base + "/am/final.mdl");
    }

    public static boolean hasTtsModel(Context context, String ttsFolderCode) {
        String folder = ttsAssetFolder(ttsFolderCode);
        String modelFile = ttsModelFileName(ttsFolderCode);
        if (folder == null || modelFile == null) return false;

        return assetExists(context, TTS_ROOT + "/" + folder + "/" + modelFile)
                && assetExists(context, TTS_ROOT + "/tokens.txt");
    }

    public static boolean isLanguageFullySupported(Context context, LangConfig lang) {
        return hasAsrModel(context, lang.asrCode) && hasTtsModel(context, lang.ttsFolder);
    }

    private static boolean assetExists(Context context, String assetPath) {
        AssetManager assets = context.getAssets();
        try {
            String[] parts = assetPath.split("/");
            if (parts.length == 1) {
                String[] top = assets.list("");
                if (top == null) return false;
                for (String name : top) if (name.equals(parts[0])) return true;
                return false;
            }

            String dir = assetPath.substring(0, assetPath.lastIndexOf('/'));
            String file = assetPath.substring(assetPath.lastIndexOf('/') + 1);
            String[] children = assets.list(dir);
            if (children == null) return false;
            for (String child : children) {
                if (child.equals(file)) return true;
            }
            return false;
        } catch (IOException e) {
            return false;
        }
    }
}
