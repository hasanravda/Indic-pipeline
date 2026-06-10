package com.example.indicpipeline;

public final class FbankJNI {
    static {
        System.loadLibrary("fbank");
    }
    private FbankJNI() {}

    public static native float[][] computeFbank80(float[] wav, int sampleRate);
}
