# Lint as: python3
"""Create the tfrecord files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

import json
import os


from deepiano.wav2mid import audio_label_data_utils

from deepiano.music import midi_io

import tensorflow.compat.v1 as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/data/maestro/maestro-v2.0.0-16k-noised',
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', '/data/maestro/maestro-v2.0.0-16k-noised-tfrecord',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')


def parse_config_json():
  json_file_name = os.path.join(FLAGS.input_dir, 'maestro-v2.0.0.json')

  train_data_set = {}
  test_data_set = {}
  validation_data_set = {}
  with open(json_file_name) as f:
    configs = json.load(f)
    for config in configs:
      year = config['year']
      midi_filename = config['midi_filename']
      audio_filename = config['audio_filename']
      duration=config['duration']
      split = config['split']
      if split == 'train':
        train_data = train_data_set.get(year)
        if not train_data:
          train_data_set[year] = [(year, midi_filename, audio_filename, duration)]
        else:
          train_data.append((year, midi_filename, audio_filename, duration))
      elif split == 'test':
        test_data = test_data_set.get(year)
        if not test_data:
          test_data_set[year] = [(year, midi_filename, audio_filename, duration)]
        else:
          test_data.append((year, midi_filename, audio_filename, duration))
      elif split == 'validation':
        validation_data = validation_data_set.get(year)
        if not validation_data:
          validation_data_set[year] = [(year, midi_filename, audio_filename, duration)]
        else:
          validation_data.append((year, midi_filename, audio_filename, duration))
  return train_data_set, test_data_set, validation_data_set


def generate_set(data_type, year, dataset):
  """Generate the train TFRecord."""
  file_pairs = []
  total_time = 0
  for year, midi_filename, audio_filename, duration in dataset:
    fixed_midi_filename = midi_filename.replace('.midi', '_16k.midi')
    fixed_audio_filename = audio_filename.replace('.wav', '_16k.wav')
    fixed_midi_path = os.path.join(FLAGS.input_dir, fixed_midi_filename)
    fixed_audio_path = os.path.join(FLAGS.input_dir, fixed_audio_filename)
    # find matching mid files
    if os.path.isfile(fixed_midi_path) and os.path.isfile(fixed_audio_path):
        file_pairs.append((fixed_audio_path, fixed_midi_path))
        total_time = total_time + duration
  return total_time

def main(unused_argv):
  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  total_train_time = 0
  train_data_set, test_data_set, validation_data_set = parse_config_json()
  for year, dataset in train_data_set.items():
    duration = generate_set('train', year, dataset)
    total_train_time = total_train_time + duration
  print('total time: ' + str(total_train_time))

def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
