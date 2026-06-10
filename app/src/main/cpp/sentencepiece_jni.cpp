#include <jni.h>
#include <string>
#include <vector>
#include "sentencepiece_processor.h"

static sentencepiece::SentencePieceProcessor* g_processor = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_indicpipeline_SentencePieceTokenizer_loadModel(JNIEnv* env, jobject thiz, jstring modelPath) {
    if (modelPath == nullptr) return JNI_FALSE;
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!g_processor) g_processor = new sentencepiece::SentencePieceProcessor();
    bool ok = g_processor->Load(path).ok();
    env->ReleaseStringUTFChars(modelPath, path);
    return (jboolean)ok;
}

JNIEXPORT jobjectArray JNICALL
Java_com_example_indicpipeline_SentencePieceTokenizer_tokenizeAsPieces(JNIEnv* env, jobject thiz, jstring inputText) {
    if (!g_processor || inputText == nullptr) return nullptr;
    const char* text = env->GetStringUTFChars(inputText, nullptr);

    std::vector<std::string> pieces;
    g_processor->Encode(text, &pieces);
    env->ReleaseStringUTFChars(inputText, text);

    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(pieces.size(), stringClass, nullptr);

    for (size_t i = 0; i < pieces.size(); i++) {
        jstring str = env->NewStringUTF(pieces[i].c_str());
        env->SetObjectArrayElement(result, i, str);
        env->DeleteLocalRef(str);
    }
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_example_indicpipeline_SentencePieceTokenizer_decodePieces(JNIEnv* env, jobject thiz, jobjectArray piecesArray) {
    if (!g_processor || piecesArray == nullptr) return nullptr;

    jsize len = env->GetArrayLength(piecesArray);
    std::vector<std::string> pieces;

    for (int i = 0; i < len; i++) {
        jstring jstr = (jstring)env->GetObjectArrayElement(piecesArray, i);
        const char* str = env->GetStringUTFChars(jstr, nullptr);
        pieces.push_back(std::string(str));
        env->ReleaseStringUTFChars(jstr, str);
        env->DeleteLocalRef(jstr);
    }

    std::string detokenized;
    g_processor->Decode(pieces, &detokenized);
    return env->NewStringUTF(detokenized.c_str());
}

}