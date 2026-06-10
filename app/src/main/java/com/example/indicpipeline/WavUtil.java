package com.example.indicpipeline;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class WavUtil {

    // Reads a WAV file directly from the Android Assets folder
    public static short[] readWavFromStream(InputStream is) throws IOException {
        byte[] header = new byte[44];
        int read = is.read(header);
        if (read < 44) throw new IOException("Invalid WAV file");

        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[4096];
        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        buffer.flush();
        byte[] audioBytes = buffer.toByteArray();
        is.close();

        short[] pcm = new short[audioBytes.length / 2];
        ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pcm);
        return pcm;
    }

    // Converts the AI float array into a playable 16-bit WAV file and saves it
    public static void writeWav(float[] audioData, File outputFile) throws IOException {
        short[] pcm = new short[audioData.length];
        for (int i = 0; i < audioData.length; i++) {
            float val = audioData[i] * 32767.0f;
            if (val > 32767.0f) val = 32767.0f;
            if (val < -32768.0f) val = -32768.0f;
            pcm[i] = (short) val;
        }

        FileOutputStream fos = new FileOutputStream(outputFile);
        writeWavHeader(fos, pcm.length * 2);

        ByteBuffer bb = ByteBuffer.allocate(pcm.length * 2);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        bb.asShortBuffer().put(pcm);
        fos.write(bb.array());
        fos.close();
    }

    private static void writeWavHeader(FileOutputStream out, int audioDataSize) throws IOException {
        int totalDataLen = audioDataSize + 36;
        int sampleRate = 16000;
        int channels = 1;
        int byteRate = 16000 * 2;

        byte[] header = new byte[44];
        header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
        header[4] = (byte) (totalDataLen & 0xff); header[5] = (byte) ((totalDataLen >> 8) & 0xff);
        header[6] = (byte) ((totalDataLen >> 16) & 0xff); header[7] = (byte) ((totalDataLen >> 24) & 0xff);
        header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
        header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
        header[20] = 1; header[21] = 0;
        header[22] = (byte) channels; header[23] = 0;
        header[24] = (byte) (sampleRate & 0xff); header[25] = (byte) ((sampleRate >> 8) & 0xff);
        header[26] = (byte) ((sampleRate >> 16) & 0xff); header[27] = (byte) ((sampleRate >> 24) & 0xff);
        header[28] = (byte) (byteRate & 0xff); header[29] = (byte) ((byteRate >> 8) & 0xff);
        header[30] = (byte) ((byteRate >> 16) & 0xff); header[31] = (byte) ((byteRate >> 24) & 0xff);
        header[32] = (byte) (2); header[33] = 0;
        header[34] = 16; header[35] = 0;
        header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
        header[40] = (byte) (audioDataSize & 0xff); header[41] = (byte) ((audioDataSize >> 8) & 0xff);
        header[42] = (byte) ((audioDataSize >> 16) & 0xff); header[43] = (byte) ((audioDataSize >> 24) & 0xff);
        out.write(header, 0, 44);
    }
}