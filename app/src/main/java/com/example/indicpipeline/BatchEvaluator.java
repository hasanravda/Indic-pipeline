package com.example.indicpipeline;

import android.content.Context;
import android.os.Environment;
import android.util.Log;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.util.List;

public class BatchEvaluator {

    private static final double POWER_KW = 0.005; // 5 Watts
    private static final double CARBON_INTENSITY_G_KWH = 714.0;

    // INTERFACE TO TALK TO THE UI
    public interface BenchmarkCallback {
        void onProgress(int currentFile, int totalFiles, String statusText);
        void onComplete(String finalMessage);
        void onError(String errorMsg);
    }

    private static double calculateCarbon(long timeMs) {
        double timeHours = timeMs / 3600000.0;
        return timeHours * POWER_KW * CARBON_INTENSITY_G_KWH;
    }

    public static void runBatchBenchmark(Context context, AsrEngine asrEngine, OfflineTranslator translator, List<LangConfig> allLangs, BenchmarkCallback callback) {
        new Thread(() -> {
            File downloadsDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "IndicBenchmark");
            if (!downloadsDir.exists()) downloadsDir.mkdirs();

            try {
                String[] sourceFolders = context.getAssets().list("test_data");
                if (sourceFolders == null || sourceFolders.length == 0) {
                    callback.onError("No 'test_data' folder found in assets!");
                    return;
                }

                // 1. PRE-CALCULATE TOTAL FILES FOR THE PROGRESS BAR
                int totalFilesToProcess = 0;
                for (String srcFolder : sourceFolders) {
                    LangConfig srcLang = getLangConfigByName(srcFolder, allLangs);
                    if (srcLang == null) continue;

                    String[] files = context.getAssets().list("test_data/" + srcFolder);
                    if (files != null) {
                        int validWavs = 0;
                        for (String f : files) if (f.toLowerCase().endsWith(".wav")) validWavs++;
                        // Total = (Valid WAVs) * (Number of TARGET languages minus the source language)
                        totalFilesToProcess += validWavs * (allLangs.size() - 1);
                    }
                }

                if (totalFilesToProcess == 0) {
                    callback.onError("No .wav files found inside the test_data language folders.");
                    return;
                }

                int filesProcessed = 0;

                // 2. RUN THE ACTUAL BENCHMARK
                for (String srcFolderName : sourceFolders) {
                    LangConfig srcLang = getLangConfigByName(srcFolderName, allLangs);
                    if (srcLang == null) continue;

                    String[] audioFiles = context.getAssets().list("test_data/" + srcFolderName);
                    if (audioFiles == null || audioFiles.length == 0) continue;

                    for (LangConfig tgtLang : allLangs) {
                        if (srcLang.name.equalsIgnoreCase(tgtLang.name)) continue;

                        String pairName = srcFolderName.toLowerCase() + "_to_" + tgtLang.name.toLowerCase();

                        File outputFolder = new File(downloadsDir, pairName);
                        if (!outputFolder.exists()) outputFolder.mkdirs();

                        File reportFile = new File(downloadsDir, pairName + "_report.txt");

                        try (FileWriter writer = new FileWriter(reportFile, false)) {
                            writer.write("=== BENCHMARK REPORT: " + pairName.toUpperCase() + " ===\n\n");

                            TtsEngine ttsEngine = new TtsEngine(context, null, tgtLang.ttsFolder);

                            for (String audioFile : audioFiles) {
                                if (!audioFile.toLowerCase().endsWith(".wav")) continue;

                                // UPDATE PROGRESS BAR
                                filesProcessed++;
                                callback.onProgress(filesProcessed, totalFilesToProcess, "Processing: " + pairName + " (" + audioFile + ")");

                                writer.write("File: " + audioFile + "\n");

                                InputStream is = context.getAssets().open("test_data/" + srcFolderName + "/" + audioFile);
                                short[] pcm = WavUtil.readWavFromStream(is);

                                long asrStart = System.currentTimeMillis();
                                AsrEngine.Result asrRes = asrEngine.transcribe(pcm, srcLang.asrCode);
                                long asrTime = System.currentTimeMillis() - asrStart;

                                long transStart = System.currentTimeMillis();
                                String transStr = translator.translate(asrRes.text, srcLang.transCode, tgtLang.transCode);
                                long transTime = System.currentTimeMillis() - transStart;

                                long ttsTime = 0;
                                try {
                                    TtsEngine.TtsResult ttsRes = ttsEngine.synthesizeToFile(transStr);
                                    if (ttsRes != null) {
                                        ttsTime = ttsRes.timeMs;
                                        File outWav = new File(outputFolder, audioFile);
                                        WavUtil.writeWav(ttsRes.audioData, outWav);
                                    }
                                } catch (Exception e) {
                                    Log.e("BATCH", "TTS failed for " + audioFile);
                                }

                                long totalTime = asrTime + transTime + ttsTime;

                                double asrCo2 = calculateCarbon(asrTime);
                                double transCo2 = calculateCarbon(transTime);
                                double ttsCo2 = calculateCarbon(ttsTime);
                                double totalCo2 = asrCo2 + transCo2 + ttsCo2;

                                writer.write(String.format("Transcription: %s\n", asrRes.text));
                                writer.write(String.format("Translation:   %s\n", transStr));
                                writer.write(String.format("Times (ms):    ASR=%d, Trans=%d, TTS=%d, Total=%d\n", asrTime, transTime, ttsTime, totalTime));
                                writer.write(String.format("Carbon (gCO2): ASR=%.6f, Trans=%.6f, TTS=%.6f, Total=%.6f\n", asrCo2, transCo2, ttsCo2, totalCo2));
                                writer.write("--------------------------------------------------\n");
                            }
                            ttsEngine.close();
                            writer.write("End of Pair Matrix Report.");
                        } catch (Exception e) {
                            Log.e("BATCH", "Failed writing report for " + pairName, e);
                        }
                    }
                }
                // FINAL SUCCESS CALLBACK
                callback.onComplete("Benchmark Complete! Data saved to Downloads/IndicBenchmark");
            } catch (Exception e) {
                callback.onError("Critical benchmark failure: " + e.getMessage());
            }
        }).start();
    }

    private static LangConfig getLangConfigByName(String folderName, List<LangConfig> langs) {
        for (LangConfig lc : langs) {
            if (lc.name.equalsIgnoreCase(folderName)) return lc;
        }
        return null;
    }
}