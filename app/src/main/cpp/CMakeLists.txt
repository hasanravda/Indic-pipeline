cmake_minimum_required(VERSION 3.22.1)
project(unified_ai CXX)

# Support for 16KB Page Sizes (Android 15+)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_16K_ALIGN")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

find_library(log-lib log)

# ==========================================
# 1. FBANK (ASR Model Feature Extractor)
# ==========================================
add_library(fbank SHARED
        fbank_jni.cpp
        fbank.cpp
)
target_link_libraries(fbank ${log-lib})

# ==========================================
# 2. SENTENCEPIECE (Translation Model Tokenizer)
# ==========================================
include(FetchContent)
FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.1.99
)
FetchContent_MakeAvailable(sentencepiece)
target_link_libraries(sentencepiece ${log-lib})

add_library(spjni SHARED
        sentencepiece_jni.cpp
)

target_include_directories(spjni PRIVATE
        ${sentencepiece_SOURCE_DIR}/src
)

# Explicitly link sentencepiece to our JNI wrapper
target_link_libraries(spjni sentencepiece ${log-lib})

# Strip debug symbols to keep APK size small
add_custom_command(TARGET spjni POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:spjni>
        COMMENT "Stripping debug symbols from libspjni.so"
)