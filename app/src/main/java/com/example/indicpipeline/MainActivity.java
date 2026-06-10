package com.example.indicpipeline;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import ai.onnxruntime.OrtEnvironment;

public class MainActivity extends AppCompatActivity {

    private AudioRecorder recorder;
    private AsrEngine asrEngine;
    private OfflineTranslator translator;
    private TtsEngine ttsEngine;
    private OrtEnvironment sharedEnv;

    private Spinner sourceSpinner, targetSpinner;
    private Button btnCall, btnEnd, btnBenchmark;
    private TextView tvSystemStatus, tvAsrOutput, tvTransOutput;

    // Benchmark Progress UI
    private LinearLayout layoutBenchmarkProgress;
    private TextView tvBenchmarkStatus;
    private ProgressBar pbBenchmark;

    // Timing TextViews
    private TextView tvAsrTime, tvTransTime, tvTtsTime, tvTotalTime;

    private LinearLayout layoutLoading;

    private BlockingQueue<short[]> audioQueue;
    private Thread pipelineThread;
    private volatile boolean inCall = false;

    private volatile boolean isSpeaking = false;
    private boolean isInitialBoot = true;

    private short[] accumulatedAudio = new short[0];
    private static final double SILENCE_RMS = 1000.0;

    private List<LangConfig> languages;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        sourceSpinner = findViewById(R.id.sourceSpinner);
        targetSpinner = findViewById(R.id.targetSpinner);
        btnCall = findViewById(R.id.btnCall);
        btnEnd = findViewById(R.id.btnEnd);
        btnBenchmark = findViewById(R.id.btnBenchmark);
        tvSystemStatus = findViewById(R.id.tvSystemStatus);
        tvAsrOutput = findViewById(R.id.tvAsrOutput);
        tvTransOutput = findViewById(R.id.tvTransOutput);
        layoutLoading = findViewById(R.id.layoutLoading);

        // Setup Progress UI
        layoutBenchmarkProgress = findViewById(R.id.layoutBenchmarkProgress);
        tvBenchmarkStatus = findViewById(R.id.tvBenchmarkStatus);
        pbBenchmark = findViewById(R.id.pbBenchmark);

        tvAsrTime = findViewById(R.id.tvAsrTime);
        tvTransTime = findViewById(R.id.tvTransTime);
        tvTtsTime = findViewById(R.id.tvTtsTime);
        tvTotalTime = findViewById(R.id.tvTotalTime);

        btnCall.setEnabled(false);
        btnEnd.setEnabled(false);
        btnBenchmark.setEnabled(false);

        setupLanguages();
        recorder = new AudioRecorder(this);
        audioQueue = new LinkedBlockingQueue<>();

        try {
            System.loadLibrary("onnxruntime");
            sharedEnv = OrtEnvironment.getEnvironment();
        } catch (Exception e) {
            Log.e("MainActivity", "Error loading ONNX Runtime: " + e.getMessage());
            e.printStackTrace();
        }

        bootAllModels();

        targetSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (isInitialBoot) return;
                swapTtsModel(languages.get(position).ttsFolder);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        btnCall.setOnClickListener(v -> startCall());
        btnEnd.setOnClickListener(v -> endCall());

        // NEW: Benchmark Button Click
        btnBenchmark.setOnClickListener(v -> startBenchmark());
    }

    private void setupLanguages() {
        languages = new ArrayList<>();
        languages.add(new LangConfig("Hindi", "hi", "hin_Deva", "hin"));
        languages.add(new LangConfig("Gujarati", "gu", "guj_Gujr", "guj"));
        languages.add(new LangConfig("Bengali", "bn", "ben_Beng", "ben"));

        ArrayAdapter<LangConfig> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, languages);
        sourceSpinner.setAdapter(adapter);
        targetSpinner.setAdapter(adapter);
        targetSpinner.setSelection(1);
    }

    private void bootAllModels() {
        btnCall.setEnabled(false);
        layoutLoading.setVisibility(View.VISIBLE);
        tvSystemStatus.setText("Booting Pipeline...");

        new Thread(() -> {
            try {
                long t0 = System.currentTimeMillis();

                // Start ASR and Translation loading in parallel
                Thread tAsr = new Thread(() -> {
                    try {
                        asrEngine = new AsrEngine(this, sharedEnv);
                    } catch (Exception e) {
                        Log.e("BOOT", "Failed to load ASR Engine", e);
                    }
                });

                Thread tTrans = new Thread(() -> {
                    try {
                        translator = new OfflineTranslator(this, sharedEnv);
                    } catch (Exception e) {
                        Log.e("BOOT", "Failed to load Translation Engine", e);
                    }
                });

                tAsr.start();
                tTrans.start();

                // TTS can start immediately or wait. Let's run it in parallel too.
                // We need the selection from targetSpinner, which is safe to read here as it's not being modified.
                final LangConfig defaultTgt = (LangConfig) targetSpinner.getSelectedItem();
                Thread tTts = new Thread(() -> {
                    try {
                        ttsEngine = new TtsEngine(this, sharedEnv, defaultTgt.ttsFolder);
                    } catch (Exception e) {
                        Log.e("BOOT", "Failed to load TTS Engine", e);
                    }
                });
                tTts.start();

                // Wait for all to finish
                tAsr.join();
                tTrans.join();
                tTts.join();

                long t1 = System.currentTimeMillis();
                runOnUiThread(() -> {
                    layoutLoading.setVisibility(View.GONE);
                    tvSystemStatus.setText("Ready! (Loaded in " + (t1 - t0) / 1000 + "s)");
                    btnCall.setEnabled(true);
                    btnBenchmark.setEnabled(true); // Enable Benchmark Button
                    isInitialBoot = false;
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> tvSystemStatus.setText("Error loading models!"));
            }
        }).start();
    }

    private void swapTtsModel(String folderName) {
        layoutLoading.setVisibility(View.VISIBLE);
        btnCall.setEnabled(false);
        tvSystemStatus.setText("Swapping TTS Voice...");

        new Thread(() -> {
            try {
                if (ttsEngine != null) ttsEngine.close();
                ttsEngine = new TtsEngine(this, null, folderName);
                runOnUiThread(() -> {
                    layoutLoading.setVisibility(View.GONE);
                    tvSystemStatus.setText("System Ready.");
                    btnCall.setEnabled(true);
                });
            } catch (Exception e) { e.printStackTrace(); }
        }).start();
    }

    // --- NORMAL MICROPHONE CALL LOGIC ---
    private void startCall() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1001);
            return;
        }

        inCall = true;
        isSpeaking = false;
        btnCall.setEnabled(false);
        btnEnd.setEnabled(true);
        btnBenchmark.setEnabled(false); // Disable benchmark during a live call
        audioQueue.clear();
        accumulatedAudio = new short[0];

        tvAsrTime.setText("ASR Time: -- ms");
        tvTransTime.setText("Translation Time: -- ms");
        tvTtsTime.setText("TTS Processing Time: -- ms");
        tvTotalTime.setText("Total AI Processing: -- ms");
        tvSystemStatus.setText("Listening...");

        LangConfig src = (LangConfig) sourceSpinner.getSelectedItem();
        LangConfig tgt = (LangConfig) targetSpinner.getSelectedItem();

        recorder.setChunkListener(chunk -> {
            if (!isSpeaking) {
                audioQueue.offer(chunk);
            }
        });

        recorder.start();

        pipelineThread = new Thread(() -> runPipelineLoop(src, tgt));
        pipelineThread.start();
    }

    private void runPipelineLoop(LangConfig src, LangConfig tgt) {
        while (inCall) {
            try {
                short[] chunk = audioQueue.take();
                List<short[]> pending = new ArrayList<>();
                pending.add(chunk);
                audioQueue.drainTo(pending);
                short[] combinedNew = combineChunks(pending);

                double rms = calculateRMS(combinedNew);

                if (rms < SILENCE_RMS) {
                    if (accumulatedAudio.length > 0) {
                        short[] audioToProcess = accumulatedAudio;
                        accumulatedAudio = new short[0];

                        runOnUiThread(() -> tvSystemStatus.setText("Thinking..."));

                        long asrStart = System.currentTimeMillis();
                        AsrEngine.Result asrRes = asrEngine.transcribe(audioToProcess, src.asrCode);
                        long asrTime = System.currentTimeMillis() - asrStart;

                        if (asrRes.text == null || asrRes.text.trim().isEmpty()) continue;
                        runOnUiThread(() -> tvAsrOutput.setText("Heard: " + asrRes.text));

                        long transStart = System.currentTimeMillis();
                        String translatedStr = translator.translate(asrRes.text, src.transCode, tgt.transCode);
                        long transTime = System.currentTimeMillis() - transStart;

                        runOnUiThread(() -> tvTransOutput.setText("Translated: " + translatedStr));

                        runOnUiThread(() -> tvSystemStatus.setText("Speaking..."));
                        isSpeaking = true;
                        audioQueue.clear();

                        long ttsTime = 0;
                        try {
                            ttsTime = ttsEngine.speak(translatedStr);
                        } catch (Exception ex) {
                            Log.e("PIPELINE_ERROR", "TTS failed: " + ex.getMessage());
                        } finally {
                            isSpeaking = false;
                        }

                        long finalTtsTime = ttsTime;
                        long totalTime = asrTime + transTime + ttsTime;

                        runOnUiThread(() -> {
                            tvAsrTime.setText("ASR Time: " + asrTime + " ms");
                            tvTransTime.setText("Translation Time: " + transTime + " ms");
                            tvTtsTime.setText("TTS Processing Time: " + finalTtsTime + " ms");
                            tvTotalTime.setText("Total AI Processing: " + totalTime + " ms");
                            tvSystemStatus.setText("Listening...");
                        });
                    }
                } else {
                    short[] newBuffer = new short[accumulatedAudio.length + combinedNew.length];
                    System.arraycopy(accumulatedAudio, 0, newBuffer, 0, accumulatedAudio.length);
                    System.arraycopy(combinedNew, 0, newBuffer, accumulatedAudio.length, combinedNew.length);
                    accumulatedAudio = newBuffer;

                    if (accumulatedAudio.length > 16000 * 8) accumulatedAudio = new short[0];
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void endCall() {
        inCall = false;
        isSpeaking = false;
        recorder.stop();
        if (pipelineThread != null) pipelineThread.interrupt();
        btnCall.setEnabled(true);
        btnEnd.setEnabled(false);
        btnBenchmark.setEnabled(true);
        tvSystemStatus.setText("Call Ended.");
    }

    // --- NEW: BENCHMARK AUTOMATION LOGIC ---
    private void startBenchmark() {
        btnCall.setEnabled(false);
        btnEnd.setEnabled(false);
        btnBenchmark.setEnabled(false);

        layoutBenchmarkProgress.setVisibility(View.VISIBLE);
        pbBenchmark.setProgress(0);
        tvBenchmarkStatus.setText("Scanning dataset folders...");

        BatchEvaluator.runBatchBenchmark(this, asrEngine, translator, languages, new BatchEvaluator.BenchmarkCallback() {
            @Override
            public void onProgress(int currentFile, int totalFiles, String statusText) {
                runOnUiThread(() -> {
                    pbBenchmark.setMax(totalFiles);
                    pbBenchmark.setProgress(currentFile);
                    tvBenchmarkStatus.setText(statusText + "\n(" + currentFile + " of " + totalFiles + ")");
                });
            }

            @Override
            public void onComplete(String finalMessage) {
                runOnUiThread(() -> {
                    tvBenchmarkStatus.setText(finalMessage);
                    Toast.makeText(MainActivity.this, finalMessage, Toast.LENGTH_LONG).show();

                    // Re-enable buttons when finished
                    btnCall.setEnabled(true);
                    btnBenchmark.setEnabled(true);
                });
            }

            @Override
            public void onError(String errorMsg) {
                runOnUiThread(() -> {
                    tvBenchmarkStatus.setText("Error: " + errorMsg);
                    Toast.makeText(MainActivity.this, "Benchmark Error!", Toast.LENGTH_SHORT).show();

                    btnCall.setEnabled(true);
                    btnBenchmark.setEnabled(true);
                });
            }
        });
    }

    // --- UTILS ---
    private double calculateRMS(short[] chunk) {
        if (chunk.length == 0) return 0;
        double sum = 0;
        for (short s : chunk) sum += s * s;
        return Math.sqrt(sum / chunk.length);
    }

    private short[] combineChunks(List<short[]> chunks) {
        int total = 0;
        for (short[] c : chunks) total += c.length;
        short[] res = new short[total];
        int pos = 0;
        for (short[] c : chunks) {
            System.arraycopy(c, 0, res, pos, c.length);
            pos += c.length;
        }
        return res;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1001) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCall();
            } else {
                tvSystemStatus.setText("Status: Mic Permission Denied!");
            }
        }
    }
}