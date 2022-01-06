# Tensorflow-MFCC
Tensorflow micro speech with MFCC (draft)

tf_micro_speech_mfcc contains tf micro speech application that uses MFCC. The main part is in [tf_micro_speech_mfcc/tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc](https://github.com/jonarani/Tensorflow-MFCC/blob/master/tf_micro_speech_mfcc/tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc). In this file important functions are spectrogram.Initalize(), mfcc.Initalize(), spectrogram.ComputeSquaredMagnitudeSpectrogram() and mfcc.Compute().


audio_processing_py contains a python script audio_processing.py that writes wav file, spectrogram and mfcc values to different files. Functions that are used for calculating spectrogram and MFCC are the ones used in micro speech training.
