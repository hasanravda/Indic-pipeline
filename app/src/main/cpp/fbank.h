#pragma once
#include <vector>

// Returns flat features (T*80) and writes T into outT.
// Output layout: row-major [t][m] => flat[t*80 + m]
std::vector<float> ComputeFbank80Flat(const std::vector<float>& wav, int sampleRate, int* outT);
