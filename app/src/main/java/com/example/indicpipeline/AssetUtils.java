package com.example.indicpipeline;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public final class AssetUtils {
    private static final String TAG = "ASR";

    private AssetUtils() {}

    /**
     * Copies everything inside assets/asr/* into filesDir/asr/*
     * This is REQUIRED because encoder.onnx uses external-data files like onnx__MatMul_8914.
     */
    public static File copyAsrAssetsToFiles(Context ctx) throws Exception {
        File outDir = new File(ctx.getFilesDir(), "asr");
        if (!outDir.exists()) outDir.mkdirs();

        AssetManager am = ctx.getAssets();

        String[] names = am.list("asr");
        if (names == null || names.length == 0) {
            throw new IllegalStateException("assets/asr folder is empty or missing!");
        }

        for (String name : names) {
            String assetPath = "asr/" + name;
            File outFile = new File(outDir, name);

            // Skip if already copied
            if (outFile.exists() && outFile.length() > 0) continue;

            try (InputStream is = am.open(assetPath);
                 FileOutputStream fos = new FileOutputStream(outFile)) {

                byte[] buf = new byte[8192];
                int r;
                while ((r = is.read(buf)) != -1) {
                    fos.write(buf, 0, r);
                }
            }
        }

        // Debug log: show a few important files
        Log.i(TAG, "Copied assets/asr -> " + outDir.getAbsolutePath());
        File check = new File(outDir, "onnx__MatMul_8914");
        Log.i(TAG, "Check external file: " + check.getAbsolutePath()
                + " exists=" + check.exists() + " size=" + (check.exists() ? check.length() : 0));

        return outDir;
    }
}
