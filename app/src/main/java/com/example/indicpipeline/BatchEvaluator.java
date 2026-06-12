package com.example.indicpipeline;

import android.content.Context;
import android.os.Environment;
import android.util.Log;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class BatchEvaluator {

    private static final String TAG = "BATCH";
    private static final double POWER_KW = 0.005; // 5 Watts
    private static final double CARBON_INTENSITY_G_KWH = 714.0;

    public interface BenchmarkCallback {
        void onProgress(int currentFile, int totalFiles, String statusText);
        void onComplete(String finalMessage);
        void onError(String errorMsg);
    }

    private static double calculateCarbon(long timeMs) {
        double timeHours = timeMs / 3600000.0;
        return timeHours * POWER_KW * CARBON_INTENSITY_G_KWH;
    }

    /** App-private output dir (no storage permission required on Android 10+). */
    private static File getBenchmarkOutputRoot(Context context) {
        File root = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        if (root == null) {
            root = context.getFilesDir();
        }
        File out = new File(root, "IndicBenchmark");
        if (!out.exists() && !out.mkdirs()) {
            Log.w(TAG, "Could not create benchmark dir: " + out.getAbsolutePath());
        }
        return out;
    }

    public static void runBatchBenchmark(Context context, AsrEngine asrEngine, OfflineTranslator translator,
                                         List<LangConfig> allLangs, BenchmarkCallback callback) {
        new Thread(() -> {
            File downloadsDir = getBenchmarkOutputRoot(context);

            try {
                String[] sourceFolders = context.getAssets().list("test_data");
                if (sourceFolders == null || sourceFolders.length == 0) {
                    callback.onError("No 'test_data' folder found in assets!");
                    return;
                }

                // Only languages with ASR + TTS models on disk
                List<LangConfig> supportedLangs = new ArrayList<>();
                for (LangConfig lang : allLangs) {
                    if (ModelAvailability.isLanguageFullySupported(context, lang)) {
                        supportedLangs.add(lang);
                    } else {
                        Log.i(TAG, "Skipping unsupported language: " + lang.name
                                + " (ASR=" + ModelAvailability.hasAsrModel(context, lang.asrCode)
                                + ", TTS=" + ModelAvailability.hasTtsModel(context, lang.ttsFolder) + ")");
                    }
                }

                if (supportedLangs.isEmpty()) {
                    callback.onError("No languages with both ASR and TTS models found in assets.");
                    return;
                }

                if (translator == null) {
                    callback.onError("Translation engine is not loaded.");
                    return;
                }

                int totalFilesToProcess = 0;
                List<String> runnableSources = new ArrayList<>();

                for (String srcFolder : sourceFolders) {
                    LangConfig srcLang = getLangConfigByName(srcFolder, supportedLangs);
                    if (srcLang == null) continue;
                    if (!ModelAvailability.hasAsrModel(context, srcLang.asrCode)) continue;

                    String[] files = context.getAssets().list("test_data/" + srcFolder);
                    if (files == null) continue;

                    int validWavs = 0;
                    for (String f : files) {
                        if (f.toLowerCase().endsWith(".wav")) validWavs++;
                    }
                    if (validWavs == 0) continue;

                    runnableSources.add(srcFolder);
                    int targetCount = 0;
                    for (LangConfig tgt : supportedLangs) {
                        if (!srcLang.name.equalsIgnoreCase(tgt.name)
                                && ModelAvailability.hasTtsModel(context, tgt.ttsFolder)) {
                            targetCount++;
                        }
                    }
                    totalFilesToProcess += validWavs * targetCount;
                }

                if (totalFilesToProcess == 0) {
                    callback.onError("No benchmark pairs found. Add .wav files under test_data/hindi, "
                            + "test_data/gujarati, or test_data/bengali.");
                    return;
                }

                int filesProcessed = 0;
                int pairsRun = 0;
                int pairsSkipped = 0;

                for (String srcFolderName : runnableSources) {
                    LangConfig srcLang = getLangConfigByName(srcFolderName, supportedLangs);
                    if (srcLang == null) continue;

                    String[] audioFiles = context.getAssets().list("test_data/" + srcFolderName);
                    if (audioFiles == null || audioFiles.length == 0) continue;

                    for (LangConfig tgtLang : supportedLangs) {
                        if (srcLang.name.equalsIgnoreCase(tgtLang.name)) continue;
                        if (!ModelAvailability.hasTtsModel(context, tgtLang.ttsFolder)) {
                            pairsSkipped++;
                            continue;
                        }

                        String pairName = srcFolderName.toLowerCase() + "_to_" + tgtLang.name.toLowerCase();
                        File outputFolder = new File(downloadsDir, pairName);
                        if (!outputFolder.exists() && !outputFolder.mkdirs()) {
                            Log.e(TAG, "Cannot create output folder: " + outputFolder.getAbsolutePath());
                            pairsSkipped++;
                            continue;
                        }

                        File reportFile = new File(downloadsDir, pairName + "_report.txt");

                        TtsEngine ttsEngine = null;
                        try (FileWriter writer = new FileWriter(reportFile, false)) {
                            writer.write("=== BENCHMARK REPORT: " + pairName.toUpperCase() + " ===\n\n");
                            ttsEngine = new TtsEngine(context, null, tgtLang.ttsFolder);
                            pairsRun++;

                            for (String audioFile : audioFiles) {
                                if (!audioFile.toLowerCase().endsWith(".wav")) continue;

                                filesProcessed++;
                                callback.onProgress(filesProcessed, totalFilesToProcess,
                                        "Processing: " + pairName + " (" + audioFile + ")");

                                writer.write("File: " + audioFile + "\n");

                                try (InputStream is = context.getAssets().open(
                                        "test_data/" + srcFolderName + "/" + audioFile)) {
                                    short[] pcm = WavUtil.readWavFromStream(is);

                                    long asrStart = System.currentTimeMillis();
                                    AsrEngine.Result asrRes = asrEngine.transcribe(pcm, srcLang.asrCode);
                                    long asrTime = System.currentTimeMillis() - asrStart;

                                    long transStart = System.currentTimeMillis();
                                    String transStr = translator.translate(
                                            asrRes.text, srcLang.transCode, tgtLang.transCode);
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
                                        Log.e(TAG, "TTS failed for " + audioFile, e);
                                    }

                                    long totalTime = asrTime + transTime + ttsTime;
                                    double asrCo2 = calculateCarbon(asrTime);
                                    double transCo2 = calculateCarbon(transTime);
                                    double ttsCo2 = calculateCarbon(ttsTime);
                                    double totalCo2 = asrCo2 + transCo2 + ttsCo2;

                                    writer.write(String.format("Transcription: %s\n", asrRes.text));
                                    writer.write(String.format("Translation:   %s\n", transStr));
                                    writer.write(String.format("Times (ms):    ASR=%d, Trans=%d, TTS=%d, Total=%d\n",
                                            asrTime, transTime, ttsTime, totalTime));
                                    writer.write(String.format(
                                            "Carbon (gCO2): ASR=%.6f, Trans=%.6f, TTS=%.6f, Total=%.6f\n",
                                            asrCo2, transCo2, ttsCo2, totalCo2));
                                    writer.write("--------------------------------------------------\n");
                                }
                            }
                            writer.write("End of Pair Matrix Report.");
                        } catch (Exception e) {
                            Log.e(TAG, "Failed benchmark pair " + pairName, e);
                            pairsSkipped++;
                        } finally {
                            if (ttsEngine != null) ttsEngine.close();
                        }
                    }
                }

                String msg = "Benchmark complete! " + pairsRun + " language pair(s) processed";
                if (pairsSkipped > 0) {
                    msg += ", " + pairsSkipped + " skipped (missing models or write error)";
                }
                msg += ".\nSaved to: " + downloadsDir.getAbsolutePath();
                callback.onComplete(msg);
            } catch (Exception e) {
                Log.e(TAG, "Critical benchmark failure", e);
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
