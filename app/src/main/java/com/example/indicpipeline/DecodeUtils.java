package com.example.indicpipeline;

import android.content.Context;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public final class DecodeUtils {
    private DecodeUtils() {}

    public static String readAssetText(Context ctx, String assetPath) throws Exception {
        try (InputStream is = ctx.getAssets().open(assetPath)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int r;
            while ((r = is.read(buf)) != -1) bos.write(buf, 0, r);
            return bos.toString(StandardCharsets.UTF_8.name());
        }
    }

    public static JSONObject readAssetJson(Context ctx, String assetPath) throws Exception {
        return new JSONObject(readAssetText(ctx, assetPath));
    }
}
