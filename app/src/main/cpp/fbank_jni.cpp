#include <jni.h>
#include <vector>
#include "fbank.h"

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_indicpipeline_FbankJNI_computeFbank80(JNIEnv* env, jclass,
                                                       jfloatArray wav_, jint sampleRate) {
    jsize n = env->GetArrayLength(wav_);
    std::vector<float> wav(n);
    env->GetFloatArrayRegion(wav_, 0, n, wav.data());

    int T = 0;
    std::vector<float> flat = ComputeFbank80Flat(wav, (int)sampleRate, &T); // T*80

    jclass floatArrayClass = env->FindClass("[F");
    jobjectArray out = env->NewObjectArray(T, floatArrayClass, nullptr);

    for (int t = 0; t < T; t++) {
        jfloatArray row = env->NewFloatArray(80);
        env->SetFloatArrayRegion(row, 0, 80, flat.data() + t * 80);
        env->SetObjectArrayElement(out, t, row);
        env->DeleteLocalRef(row);
    }

    return out;
}