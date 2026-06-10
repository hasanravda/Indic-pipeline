package com.example.indicpipeline;

public class SentencePieceTokenizer {
    static {
        System.loadLibrary("spjni");
    }
    public native boolean loadModel(String modelPath);

    // Now explicitly handling String arrays like Python
    public native String[] tokenizeAsPieces(String inputText);
    public native String decodePieces(String[] pieces);
}