package com.example.indicpipeline;

public class ScriptConverter {
    private static final int DEVA_START = 0x0900;
    private static final int DEVA_END = 0x097F;

    private static int getOffset(String langCode) {
        if (langCode == null) return 0;

        // These languages use Devanagari natively or don't use Indic scripts
        if (langCode.endsWith("_Deva") || langCode.endsWith("_Latn") ||
                langCode.endsWith("_Arab") || langCode.endsWith("_Olck") ||
                langCode.endsWith("_Mtei")) {
            return 0;
        }

        // Shift the Unicode blocks to match Devanagari
        switch (langCode.substring(4)) {
            case "Beng": return 0x0980 - DEVA_START; // Bengali, Assamese
            case "Guru": return 0x0A00 - DEVA_START; // Punjabi
            case "Gujr": return 0x0A80 - DEVA_START; // Gujarati
            case "Orya": return 0x0B00 - DEVA_START; // Odia
            case "Taml": return 0x0B80 - DEVA_START; // Tamil
            case "Telu": return 0x0C00 - DEVA_START; // Telugu
            case "Knda": return 0x0C80 - DEVA_START; // Kannada
            case "Mlym": return 0x0D00 - DEVA_START; // Malayalam
            default: return 0;
        }
    }

    // Convert source language characters INTO Devanagari for the AI
    public static String convertToDevanagari(String text, String langCode) {
        int offset = getOffset(langCode);
        if (offset == 0) return text;

        int blockStart = DEVA_START + offset;
        int blockEnd = DEVA_END + offset;

        StringBuilder sb = new StringBuilder(text.length());
        for (char c : text.toCharArray()) {
            if (c >= blockStart && c <= blockEnd) {
                sb.append((char) (c - offset));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    // Convert AI output back FROM Devanagari to the target language script
    public static String convertFromDevanagari(String text, String langCode) {
        int offset = getOffset(langCode);
        if (offset == 0) return text;

        StringBuilder sb = new StringBuilder(text.length());
        for (char c : text.toCharArray()) {
            if (c >= DEVA_START && c <= DEVA_END) {
                sb.append((char) (c + offset));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}