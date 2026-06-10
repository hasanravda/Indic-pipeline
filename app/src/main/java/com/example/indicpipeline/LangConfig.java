package com.example.indicpipeline;

public class LangConfig {
    public final String name;
    public final String asrCode;
    public final String transCode;
    public final String ttsFolder;

    public LangConfig(String name, String asrCode, String transCode, String ttsFolder) {
        this.name = name;
        this.asrCode = asrCode;
        this.transCode = transCode;
        this.ttsFolder = ttsFolder;
    }

    @Override
    public String toString() {
        return name;
    }
}