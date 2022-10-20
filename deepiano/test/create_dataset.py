"""Create the tfrecord files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re

from deepiano.wav2mid import split_audio_and_label_data

from deepiano.music import audio_io
from deepiano.music import midi_io

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '../../data/theone',
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', '../../data/theone',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

test_dirs = ['tap']
train_dirs = ['tap']


def generate_train_set():
  """Generate the train TFRecord."""
  train_file_pairs = []
  for directory in train_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      train_file_pairs.append((wav_file, mid_file))

  train_output_name = os.path.join(FLAGS.output_dir,
                                   'theone_train.tfrecord')

  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for idx, pair in enumerate(train_file_pairs):
      print('{} of {}: {}'.format(idx, len(train_file_pairs), pair[0]))
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], 'rb').read()
      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])
      for example in split_audio_and_label_data.process_record(
          wav_data, ns, pair[0], FLAGS.min_length, FLAGS.max_length,
          FLAGS.sample_rate, load_audio_with_librosa=True):
        writer.write(example.SerializeToString())


def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []
  for directory in test_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      test_file_pairs.append((wav_file, mid_file))

  test_output_name = os.path.join(FLAGS.output_dir,
                                  'theone_test.tfrecord')

  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for idx, pair in enumerate(test_file_pairs):
      print('{} of {}: {}'.format(idx, len(test_file_pairs), pair[0]))
      # load the wav data and resample it.
      samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      example = split_audio_and_label_data.create_example(pair[0], ns, wav_data)
      writer.write(example.SerializeToString())


def main(unused_argv):
  generate_train_set()
  # generate_test_set()


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
