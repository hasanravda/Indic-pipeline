#include "fbank.h"
#include <cmath>
#include <algorithm>

// Very small, self-contained mel fbank (approx).
// Good enough to run and test the pipeline.
// Later we can replace with exact Kaldi.

static inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> ComputeFbank80Flat(const std::vector<float>& wav, int sampleRate, int* outT) {
    const int nMels = 80;
    const float frameLenMs = 25.0f;
    const float frameShiftMs = 10.0f;

    int frameLen = (int)std::round(sampleRate * frameLenMs / 1000.0f);
    int frameShift = (int)std::round(sampleRate * frameShiftMs / 1000.0f);

    if ((int)wav.size() < frameLen) {
        *outT = 0;
        return {};
    }

    int T = 1 + ((int)wav.size() - frameLen) / frameShift;
    *outT = T;

    // FFT size (next power of 2)
    int nfft = 1;
    while (nfft < frameLen) nfft <<= 1;

    int nFreq = nfft / 2 + 1;

    // Hanning window
    std::vector<float> win(frameLen);
    for (int i = 0; i < frameLen; i++) {
        win[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (frameLen - 1));
    }

    // Mel filterbank
    float fMin = 20.0f;
    float fMax = sampleRate / 2.0f;

    float melMin = hz_to_mel(fMin);
    float melMax = hz_to_mel(fMax);

    std::vector<float> melPoints(nMels + 2);
    for (int i = 0; i < nMels + 2; i++) {
        melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
    }

    std::vector<int> bin(nMels + 2);
    for (int i = 0; i < nMels + 2; i++) {
        float hz = mel_to_hz(melPoints[i]);
        int b = (int)std::floor((nfft + 1) * hz / sampleRate);
        b = std::max(0, std::min(b, nFreq - 1));
        bin[i] = b;
    }

    // For simplicity, build triangular filters on the fly
    std::vector<float> out(T * nMels);

    // scratch
    std::vector<float> frame(nfft);
    std::vector<float> power(nFreq);

    for (int t = 0; t < T; t++) {
        int start = t * frameShift;

        // windowed frame, zero pad
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int i = 0; i < frameLen; i++) {
            frame[i] = wav[start + i] * win[i];
        }

        // naive DFT magnitude (slow but OK for now)
        // If you want faster, we can add kissfft later.
        for (int k = 0; k < nFreq; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < nfft; n++) {
                double ang = -2.0 * M_PI * k * n / nfft;
                re += frame[n] * std::cos(ang);
                im += frame[n] * std::sin(ang);
            }
            double p = re * re + im * im;
            power[k] = (float)p;
        }

        // apply mel filters
        for (int m = 0; m < nMels; m++) {
            int left = bin[m];
            int center = bin[m + 1];
            int right = bin[m + 2];

            double e = 0.0;

            for (int k = left; k < center; k++) {
                float w = (center == left) ? 0.0f : (float)(k - left) / (center - left);
                e += power[k] * w;
            }
            for (int k = center; k < right; k++) {
                float w = (right == center) ? 0.0f : (float)(right - k) / (right - center);
                e += power[k] * w;
            }

            // log energy
            float loge = (float)std::log(std::max(1e-10, e));
            out[t * nMels + m] = loge;
        }
    }

    return out;
}
