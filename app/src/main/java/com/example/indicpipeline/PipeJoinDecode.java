package com.example.indicpipeline;

public final class PipeJoinDecode {
    private PipeJoinDecode() {}

    public static String decode(String s) {
        if (s == null) return "";
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != '|') {
                out.append(c);
                continue;
            }
            char left = (i > 0) ? s.charAt(i - 1) : ' ';
            char right = (i + 1 < s.length()) ? s.charAt(i + 1) : ' ';
            boolean leftNonSpace = !Character.isWhitespace(left);
            boolean rightNonSpace = !Character.isWhitespace(right);
            if (leftNonSpace && rightNonSpace) {
                // join => skip pipe
            } else {
                // separator => space
                out.append(' ');
            }
        }
        return out.toString().trim().replaceAll("\\s+", " ");
    }
}
