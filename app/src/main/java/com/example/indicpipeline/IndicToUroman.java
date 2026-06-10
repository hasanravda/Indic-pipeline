package com.example.indicpipeline;

import java.util.HashMap;
import java.util.Map;

public class IndicToUroman {
    private static final Map<Character, String> devaMap = new HashMap<>();

    static {
        // Vowels
        devaMap.put('अ', "a"); devaMap.put('आ', "a"); devaMap.put('इ', "i"); devaMap.put('ई', "i");
        devaMap.put('उ', "u"); devaMap.put('ऊ', "u"); devaMap.put('ऋ', "ri"); devaMap.put('ॠ', "ri");
        devaMap.put('ए', "e"); devaMap.put('ऐ', "ai"); devaMap.put('ओ', "o"); devaMap.put('औ', "au");

        // Consonants (Strictly single characters to avoid Java compilation errors)
        devaMap.put('क', "k"); devaMap.put('ख', "kh"); devaMap.put('ग', "g"); devaMap.put('घ', "gh"); devaMap.put('ङ', "ng");
        devaMap.put('च', "c"); devaMap.put('छ', "ch"); devaMap.put('ज', "j"); devaMap.put('झ', "jh"); devaMap.put('ञ', "ny");
        devaMap.put('ट', "t"); devaMap.put('ठ', "th"); devaMap.put('ड', "d"); devaMap.put('ढ', "dh"); devaMap.put('ण', "n");
        devaMap.put('त', "t"); devaMap.put('थ', "th"); devaMap.put('द', "d"); devaMap.put('ध', "dh"); devaMap.put('न', "n");
        devaMap.put('प', "p"); devaMap.put('फ', "f"); devaMap.put('ब', "b"); devaMap.put('भ', "bh"); devaMap.put('म', "m");
        devaMap.put('य', "y"); devaMap.put('र', "r"); devaMap.put('ल', "l"); devaMap.put('व', "v");
        devaMap.put('श', "sh"); devaMap.put('ष', "sh"); devaMap.put('स', "s"); devaMap.put('ह', "h");
        devaMap.put('ळ', "l");

        // Matras (Dependent Vowels)
        devaMap.put('ा', "a"); devaMap.put('ि', "i"); devaMap.put('ी', "i");
        devaMap.put('ु', "u"); devaMap.put('ू', "u"); devaMap.put('ृ', "ri");
        devaMap.put('े', "e"); devaMap.put('ै', "ai"); devaMap.put('ो', "o"); devaMap.put('ौ', "au");

        // Modifiers
        devaMap.put('ं', "n");
        devaMap.put('ँ', "n");
        devaMap.put('ः', "h");
        devaMap.put('्', "");   // Halant
        devaMap.put('़', "");   // Nukta
    }

    public static String transliterate(String nativeText, String langCode) {
        String devaText = convertToDevanagari(nativeText, langCode);

        // Handle multi-character ligatures safely as Strings, NOT chars!
        devaText = devaText.replace("क्ष", "ksh").replace("ज्ञ", "gy").replace("श्र", "shr");

        StringBuilder uroman = new StringBuilder();

        for (int i = 0; i < devaText.length(); i++) {
            char c = devaText.charAt(i);

            if (devaMap.containsKey(c)) {
                uroman.append(devaMap.get(c));

                if (isConsonant(c)) {
                    boolean addInherentA = true;
                    if (i + 1 < devaText.length()) {
                        char nextChar = devaText.charAt(i + 1);
                        if (isMatraOrModifier(nextChar) || nextChar == ' ' || nextChar == '.' || nextChar == ',') {
                            addInherentA = false;
                        }
                    } else {
                        addInherentA = false;
                    }
                    if (addInherentA) {
                        uroman.append("a");
                    }
                }
            } else if (c == ' ' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                uroman.append(c);
            }
        }
        return uroman.toString().trim().replaceAll("\\s+", " ").toLowerCase();
    }

    private static boolean isConsonant(char c) {
        return (c >= 'क' && c <= 'ह') || c == 'ळ';
    }

    private static boolean isMatraOrModifier(char c) {
        return (c >= 'ा' && c <= '्') || c == 'ं' || c == 'ँ' || c == 'ः' || c == '़';
    }

    private static final int DEVA_START = 0x0900;
    private static final int DEVA_END = 0x097F;

    private static int getOffset(String langCode) {
        if (langCode == null) return 0;
        if (langCode.endsWith("_Deva") || langCode.endsWith("_Latn") || langCode.endsWith("_Arab") || langCode.endsWith("_Olck") || langCode.endsWith("_Mtei")) return 0;
        switch (langCode.substring(4)) {
            case "Beng": return 0x0980 - DEVA_START;
            case "Guru": return 0x0A00 - DEVA_START;
            case "Gujr": return 0x0A80 - DEVA_START;
            case "Orya": return 0x0B00 - DEVA_START;
            case "Taml": return 0x0B80 - DEVA_START;
            case "Telu": return 0x0C00 - DEVA_START;
            case "Knda": return 0x0C80 - DEVA_START;
            case "Mlym": return 0x0D00 - DEVA_START;
            default: return 0;
        }
    }

    private static String convertToDevanagari(String text, String langCode) {
        int offset = getOffset(langCode);
        if (offset == 0) return text;
        int blockStart = DEVA_START + offset;
        int blockEnd = DEVA_END + offset;
        StringBuilder sb = new StringBuilder(text.length());
        for (char c : text.toCharArray()) {
            if (c >= blockStart && c <= blockEnd) sb.append((char) (c - offset));
            else sb.append(c);
        }
        return sb.toString();
    }
}