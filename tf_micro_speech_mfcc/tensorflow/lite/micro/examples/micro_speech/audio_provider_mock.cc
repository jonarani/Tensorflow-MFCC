/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/no_1000ms_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/yes_1000ms_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/traffic_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/industrial_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/human_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/air_sample_data.h"

#include <stdio.h>

namespace {
int16_t g_dummy_audio_data[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;
}  // namespace

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  //start_ms = start_ms / 10;
  const int traffic_start = (0 * kAudioSampleFrequency) / 1000;
  const int traffic_end = (1000 * kAudioSampleFrequency) / 1000;
  const int ind_start = (5000 * kAudioSampleFrequency) / 1000;
  const int ind_end = (6000 * kAudioSampleFrequency) / 1000;
  const int human_start = (10000 * kAudioSampleFrequency) / 1000;
  const int human_end = (11000 * kAudioSampleFrequency) / 1000;
  const int air_start = (14000 * kAudioSampleFrequency) / 1000;
  const int air_end = (15000 * kAudioSampleFrequency) / 1000;
  const int wraparound = (1000 * kAudioSampleFrequency) / 1000;
  const int start_sample = (start_ms * kAudioSampleFrequency) / 1000;
  if (start_ms > 1000)
    while(1) ;

  // printf("start_sample: %d\n", start_sample);
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    const int sample_index = (start_sample + i) % wraparound;
    //printf ("sample_index: %d\n", sample_index);
    // TF_LITE_REPORT_ERROR(error_reporter, "%d", sample_index);
    // int16_t sample;
    // if ((sample_index >= traffic_start) && (sample_index < traffic_end)) {
    //   sample = g_traffic_sample_data[sample_index - traffic_start];
    // } 
    // else if ((sample_index >= ind_start) && (sample_index < ind_end)) {
    //   sample = g_industrial_sample_data[sample_index - ind_start];
    // } 
    // else if ((sample_index >= human_start) && (sample_index < human_end)) {
    //   sample = g_human_sample_data[sample_index - human_start];
    // } 
    // else if ((sample_index >= air_start) && (sample_index < air_end)) {
    //   sample = g_air_sample_data[sample_index - air_start];
    // } 

      g_dummy_audio_data[i] = g_human_sample_data[sample_index];
      //g_dummy_audio_data[i] = leftChannel[sample_index];

      // constexpr int kFeatureSliceSize = 40;
      // constexpr int kFeatureSliceCount = 30;
      // constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
      // constexpr int kFeatureSliceStrideMs = 32;
      // constexpr int kFeatureSliceDurationMs = 64;

      // constexpr int kFeatureSliceSize = 40;
      // constexpr int kFeatureSliceCount = 49;
      // constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
      // constexpr int kFeatureSliceStrideMs = 20;
      // constexpr int kFeatureSliceDurationMs = 30;
  }
  // for (int j = 0; j < 40; ++j)
  // {
  //   printf ("%d ", g_dummy_audio_data[j]);
  // }
  // printf("\n");
  //TF_LITE_REPORT_ERROR(error_reporter, "%d", cnt);
 // TF_LITE_REPORT_ERROR(error_reporter, "yes_start: %d, yes_end: %d, no_start: %d, no_end: %d, wraparound: %d, start_sample: %d, start_ms: %d", yes_start, yes_end, no_start, no_end, wraparound, start_sample, start_ms);
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_dummy_audio_data;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  g_latest_audio_timestamp += 64; // (kMaxAudioSampleSize * 1000) / kAudioSampleFrequency
  return g_latest_audio_timestamp;
}
