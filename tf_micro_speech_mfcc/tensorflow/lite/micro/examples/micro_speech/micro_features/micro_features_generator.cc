/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h"

#include <cmath>
#include <cstring>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/kernels/internal/spectrogram.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include "tensorflow/core/kernels/mfcc.h"

// Configure FFT to output 16 bit fixed point.
#define FIXED_POINT 16

// From training
#define MFCC_FEATURE_MIN  (-247.0)
#define MFCC_FEATURE_MAX  (30.0)

static int window_size = 512;
static int stride = 256;
static bool magnitude_squared = true;
static int output_height;
static tflite::internal::Spectrogram spectrogram;

static std::ofstream spectrogram_file;
static std::ofstream mfcc_file;

namespace {

FrontendState g_micro_features_state;
bool g_is_first_time = true;
bool need_resize = true;

}  // namespace

TfLiteStatus InitializeMicroFeatures(tflite::ErrorReporter* error_reporter) {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 3500.0; // 3500 with our changes, 7500 tensorflow default
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    TF_LITE_REPORT_ERROR(error_reporter, "FrontendPopulateState() failed");
    return kTfLiteError;
  }

  if (mfcc.Initialize (g_micro_features_state.fft.fft_size / 2 + 1,
                       kAudioSampleFrequency,
                       config.filterbank.lower_band_limit,
                       config.filterbank.upper_band_limit,
                       config.filterbank.num_channels,
                       kFeatureSliceSize,
                       1e-12
                       ) == false) {
    std::cout << "Mfcc initalize failed\n";
  }
  
  g_is_first_time = true;
  return kTfLiteOk;
}

// This is not exposed in any header, and is only used for testing, to ensure
// that the state is correctly set up before generating results.
void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}

TfLiteStatus GenerateMicroFeatures(tflite::ErrorReporter* error_reporter,
                                   const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input;
  static std::vector<std::vector<double>> spectrogram_output;
  static std::vector<double> spectrogram_input;
  static std::vector<double> mfcc_output;

  if (g_is_first_time) {
    //printf ("Generatemicro first \n");
    spectrogram.Initialize(window_size, stride);        // TODO: + 256 to stide so that it would match what is outputed in Python?
                                                        // Window stride as big as window size in Python??
    frontend_input = input;
    g_is_first_time = false;

    spectrogram_input.resize(input_size);

    spectrogram_file.open("spectrogram_new.txt", std::ios::out);
    mfcc_file.open("mfcc_features.txt", std::ios::out);
  } else {
    int stride_size_samples = ((kAudioSampleFrequency * (kFeatureSliceDurationMs - kFeatureSliceStrideMs)) / 1000);
    //printf ("Generatemicro \n");
    frontend_input = input + stride_size_samples;
    
    // Don't calculate spectrogram again for overlapping window
    input_size -= stride_size_samples;
    input += stride_size_samples;

    // Resize spectrogram_input vector also
    // Since after first time only stride length of samples are used
    if (need_resize == true) {
      spectrogram_input.resize(input_size);
      need_resize = false;
    }
  }
  // printf (" input_size: %d\n", input_size);
  // for (int i = 0; i < input_size; ++i)
  // {
  //   printf ("%d ", input[i]);
  // }
  // printf ("\n");

  //printf ("INPUT \n");
  for (int i = 0; i < input_size; i++)
  {
    // Normalize according to what is done in training.
    // WAV values from [-32768, 32767] to [-1, 1]
    if (input[i] < 0)
    {
      //spectrogram_input.push_back((double)input[i] / 32768.0);
      spectrogram_input[i] = (double)input[i] / 32768.0;
    }
    else 
    {
      //spectrogram_input.push_back((double)input[i] / 32767.0);
      spectrogram_input[i] = (double)input[i] / 32767.0;
    }
    //if (i < 10)
      //std::cout << spectrogram_input[i] << " ";
  }
  //printf ("\n\n");


  spectrogram.ComputeSquaredMagnitudeSpectrogram(spectrogram_input, &spectrogram_output);
  //spectrogram_input.clear();
  //printf ("SPECTROGRAM \n");
  for (size_t i = 0; i < spectrogram_output.size(); ++i)
  {
    for (size_t j = 0; j < spectrogram_output[i].size(); ++j)
    {
     //std::cout << spectrogram_output[i][j] << " ";
     spectrogram_file << spectrogram_output[i][j] << "\n";
     //printf("%f ",spectrogram_output[i][j]);
    }
    spectrogram_file << std::endl;
    //printf("\n");
  }
  printf ("Spectrogram.size() = %zu\n", spectrogram_output.size());
  printf ("Spectrogram[0].size() = %zu\n", spectrogram_output[0].size());
  //spectrogram_file << std::endl;

  // MFCC
  mfcc.Compute(spectrogram_output[0], &mfcc_output);

  // output_size if 40, coming from model_settings.h
  for (int i = 0; i < output_size; i++){
    // Quantization as done in training
    int16_t quantized_value = round((255. * (mfcc_output[i] - MFCC_FEATURE_MIN)) / (MFCC_FEATURE_MAX - MFCC_FEATURE_MIN));
    if (quantized_value < 0) {
      quantized_value = 0;
    }
    else if (quantized_value > 255) {
      quantized_value = 255;
    }
    quantized_value -= 128;

    output[i] = quantized_value;
    printf ("%d ", output[i]);
  }
  printf ("\n");

  printf ("Mfcc size: %zu\n", mfcc_output.size());

  for (size_t i = 0; i < mfcc_output.size(); ++i){
    mfcc_file << mfcc_output[i] << " ";
  }
  mfcc_file << "\n";
  mfcc_file.flush();

  //printf ("size=%zu, input_used=%zu\n", g_micro_features_state.window.size, g_micro_features_state.window.input_used);
  
  // FrontendOutput frontend_output = FrontendProcessSamples(
  //     &g_micro_features_state, frontend_input, input_size, num_samples_read); // input_size -  ((kAudioSampleFrequency * (kFeatureSliceDurationMs - kFeatureSliceStrideMs)) / 1000)

  // //printf ("frontend_output\n");
  // for (size_t i = 0; i < frontend_output.size; ++i) {
  //   // These scaling values are derived from those used in input_data.py in the
  //   // training pipeline.
  //   // The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
  //   // range. In training, these are then arbitrarily divided by 25.6 to get
  //   // float values in the rough range of 0.0 to 26.0. This scaling is performed
  //   // for historical reasons, to match up with the output of other feature
  //   // generators.
  //   // The process is then further complicated when we quantize the model. This
  //   // means we have to scale the 0.0 to 26.0 real values to the -128 to 127
  //   // signed integer numbers.
  //   // All this means that to get matching values from our integer feature
  //   // output into the tensor input, we have to perform:
  //   // input = (((feature / 25.6) / 26.0) * 256) - 128
  //   // To simplify this and perform it in 32-bit integer math, we rearrange to:
  //   // input = (feature * 256) / (25.6 * 26.0) - 128
    
  //   constexpr int32_t value_scale = 256;
  //   constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
  //   int32_t value =
  //       ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
  //       value_div;
  //   value -= 128;
  //   if (value < -128) {
  //     value = -128;
  //   }
  //   if (value > 127) {
  //     value = 127;
  //   }
  //   output[i] = value;
  //   //printf ("%d ", output[i]);
  // }
  printf ("\n");
  return kTfLiteOk;
}
