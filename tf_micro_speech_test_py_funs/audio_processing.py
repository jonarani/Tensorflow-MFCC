from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

tf.compat.v1.disable_eager_execution()

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None

wav_loader = io_ops.read_file("dataset/human.wav")
wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=8000)

# sess = tf.compat.v1.Session()
# with sess.as_default():
#     np_arr = wav_decoder.audio.eval()
#     #np_arr = np_arr * 32767
#     #np.savetxt("wav_file.txt", np_arr, delimiter="\n")
#     with open ("wav_file.txt", 'w') as file:
#         for num in np_arr:
#             result = int (num * 32768)
#             file.write("{}".format(result))
#             file.write("\n")

sample_rate = 8000
window_size_ms = 64
window_step_ms = 32
window_size_samples = 512 # 64 / 1000 * 8000
window_stride_samples = 256
fingerprint_width = 40

# for quantizing
mfcc_features_min = -247.0
mfcc_features_max = 30.0

# int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
#int16_input = tf.cast(tf.multiply(wav_decoder.audio, 32768), tf.int16)
# micro_frontend = frontend_op.audio_microfrontend(
#     int16_input,
#     sample_rate=sample_rate,
#     window_size=window_size_ms,
#     window_step=window_step_ms,
#     upper_band_limit=3500, # tf default 7500, in our case 3500
#     num_channels=40,
#     out_scale=1,
#     out_type=tf.float32)

#output = tf.multiply(micro_frontend, (10.0 / 256.0))

spectrogram = audio_ops.audio_spectrogram(
          wav_decoder.audio,
          window_size=window_size_samples,
          stride=window_stride_samples,
          magnitude_squared=True)

print(spectrogram)
        
output = audio_ops.mfcc(
    spectrogram,
    wav_decoder.sample_rate,
    upper_frequency_limit=3500,
    lower_frequency_limit=125,
    filterbank_channel_count=40,
    dct_coefficient_count=fingerprint_width)

print (output)

sess = tf.compat.v1.Session()
np_arr = []
with sess.as_default():
  np_audio = wav_decoder.audio.eval()
  np_arr = spectrogram.eval()
  np_arr2 = output.eval()


np_arr = np.squeeze(np_arr, axis=0)
print (np_arr.shape)

np_arr2 = np.squeeze(np_arr2, axis=0)
print (np_arr2.shape)

features_min = 0.0
features_max = 26.0

np_audio = np.array(np_audio) # * 32768

with open("audio.txt", 'w') as file:
  for row in np_audio:
      for num in row:
        file.write("{}\n".format(num))

with open("spectrogram.txt", 'w') as file:
  for row in np_arr:
      for num in row:
         # quantized_value = int(round((255 * (num - features_min)) / (features_max - features_min))) - 128
          #print ("{}".format(num), sep=" ", end=" ")
        file.write("{}\n".format(num))
      file.write("\n")
      #print ()

with open("mfcc.txt", 'w') as file:
  for row in np_arr2:
      for num in row:
        #quantized_value = int(round((255 * (num - mfcc_features_min)) / (mfcc_features_max - mfcc_features_min))) - 128
        #print ("{}".format(num), sep=" ", end=" ")
        file.write("{}\n".format(num))
        #file.write("{}\n".format(quantized_value))
      file.write("\n")
      #print ()
      