# Lint as: python3
"""Create the tfrecord files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

import json
import os
import glob
from shutil import copyfile


from deepiano.wav2mid import audio_label_data_utils, audio_transform

from deepiano.music import audio_io
from deepiano.music import midi_io

import tensorflow.compat.v1 as tf

from deepiano.wav2mid import configs

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/Users/xyz/piano_dataset/exported_AI_tagging_for_train',
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', './',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')

tf.app.flags.DEFINE_enum(
    'dataset', 'self', ['v1', 'v2', 'self', 'flat'],
    'which dataset will be used')

tf.app.flags.DEFINE_enum(
    'mode', 'train', ['all', 'train', 'test', 'validation'],
    'which dataset will be used')

# for train
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')

tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')


# add for preprocess examples: add noise etc
tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_boolean(
    'preprocess_examples', True,
    'Whether to preprocess examples or assume they have already been '
    'preprocessed.')


test_dirs = ['']
train_dirs = ['']

all_wav_dir = []


def parse_config_json(dataset):
  if dataset == 'v2':
    json_file_name = os.path.join(FLAGS.input_dir, 'maestro-v2.0.0.json')
  else:
    json_file_name = os.path.join(FLAGS.input_dir, 'maestro-v1.0.0.json')


  train_data_set = {}
  test_data_set = {}
  validation_data_set = {}
  with open(json_file_name) as f:
    configs = json.load(f)
    for config in configs:
      year = config['year']
      midi_filename = config['midi_filename']
      audio_filename = config['audio_filename']
      split = config['split']
      if split == 'train':
        train_data = train_data_set.get(year)
        if not train_data:
          train_data_set[year] = [(year, midi_filename, audio_filename)]
        else:
          train_data.append((year, midi_filename, audio_filename))
      elif split == 'test':
        test_data = test_data_set.get(year)
        if not test_data:
          test_data_set[year] = [(year, midi_filename, audio_filename)]
        else:
          test_data.append((year, midi_filename, audio_filename))
      elif split == 'validation':
        validation_data = validation_data_set.get(year)
        if not validation_data:
          validation_data_set[year] = [(year, midi_filename, audio_filename)]
        else:
          validation_data.append((year, midi_filename, audio_filename))
  return train_data_set, test_data_set, validation_data_set


def generate_maestro_dataset(version, data_type, year, dataset, pre_audio_transform=False, hparams=None):
  """Generate the train TFRecord."""
  file_pairs = []
  for year, midi_filename, audio_filename in dataset:
    fixed_midi_filename = midi_filename.replace('.midi', '.midi')
    if version == 'v2':
      fixed_audio_filename = audio_filename.replace('.wav', '_16k.wav') # for v2
    elif version == 'v1':
      fixed_audio_filename = audio_filename.replace('.wav', '.wav') # for v1
    fixed_midi_path = os.path.join(FLAGS.input_dir, fixed_midi_filename)
    fixed_audio_path = os.path.join(FLAGS.input_dir, fixed_audio_filename)
    # find matching mid files
    if os.path.isfile(fixed_midi_path) and os.path.isfile(fixed_audio_path):
        file_pairs.append((fixed_audio_path, fixed_midi_path))

  output_name = os.path.join(FLAGS.output_dir,
                                   '{}.{}-tfrecord-{}-in-2020-07-22'.format(data_type, version, year))


  if data_type == 'train' or data_type == 'validation':
    with tf.python_io.TFRecordWriter(output_name) as writer:
      for idx, pair in enumerate(file_pairs):
        print('{} of {}: {}'.format(idx, len(file_pairs), pair[0]))
        # load the wav data
        wav_data = tf.gfile.Open(pair[0], 'rb').read()
        # load the midi data and convert to a notesequence
        ns = midi_io.midi_file_to_note_sequence(pair[1])
        for example in audio_label_data_utils.process_record(
            wav_data, ns, pair[0], FLAGS.min_length, FLAGS.max_length,
            FLAGS.sample_rate, pre_audio_transform=pre_audio_transform, hparams=hparams):
          writer.write(example.SerializeToString())
  elif data_type == 'test':
    with tf.python_io.TFRecordWriter(output_name) as writer:
      for idx, pair in enumerate(file_pairs):
        print('{} of {}: {}'.format(idx, len(file_pairs), pair[0]))
        if False:
          wav_base_name = os.path.basename(pair[0])
          target_wav_file = os.path.join(FLAGS.output_dir, wav_base_name + '.wav')
          target_mid_file = os.path.join(FLAGS.output_dir, wav_base_name + '.mid')
          copyfile(pair[0], target_wav_file)
          copyfile(pair[1], target_mid_file)
          continue
        # load the wav data and resample it.
        samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
        wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

        # load the midi data and convert to a notesequence
        ns = midi_io.midi_file_to_note_sequence(pair[1])

        if pre_audio_transform and hparams:
          wav_data = audio_transform.transform_wav_audio(wav_data, hparams)
        example = audio_label_data_utils.create_example(pair[0], ns, wav_data)
        writer.write(example.SerializeToString())


def generate_self_train_set(pre_audio_transform=False, hparams=None):
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
      if os.path.isfile(mid_file):
        train_file_pairs.append((wav_file, mid_file))

  train_output_name = os.path.join(FLAGS.output_dir,
                                   'train.AI_tagging_part_fixed_for_train_2020_07_22')

  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for idx, pair in enumerate(train_file_pairs):
      print('{} of {}: {}'.format(idx, len(train_file_pairs), pair[0]))
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], 'rb').read()
      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])
      for example in audio_label_data_utils.process_record(
          wav_data, ns, pair[0], FLAGS.min_length, FLAGS.max_length,
          FLAGS.sample_rate, pre_audio_transform=pre_audio_transform, hparams=hparams):
        writer.write(example.SerializeToString())


def generate_self_test_set():
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
      if os.path.isfile(mid_file):
        test_file_pairs.append((wav_file, mid_file))

  test_output_name = os.path.join(FLAGS.output_dir, 'test.AI_tagging_part_fixed_for_test_2020_06_23')

  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for idx, pair in enumerate(test_file_pairs):
      print('{} of {}: {}'.format(idx, len(test_file_pairs), pair[0]))
      # load the wav data and resample it.
      samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      example = audio_label_data_utils.create_example(pair[0], ns, wav_data)
      writer.write(example.SerializeToString())


def iter_dirs(rootDir):
  for root, dirs, files in os.walk(rootDir):
    if dirs != []:
      for dirname in dirs:
        full_dirname = os.path.join(root, dirname)
        all_wav_dir.append(full_dirname)
        iter_dirs(full_dirname)


def generate_flat_set(data_type, desc, pre_audio_transform=False, hparams=None):
  """Generate the train/test for flat midi TFRecord."""
  iter_dirs(FLAGS.input_dir)

  wav_file_pairs = []
  for directory in all_wav_dir:
    path = directory
    path = os.path.join(path, '*_16k.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root.replace('_16k', '')
      if os.path.isfile(mid_file):
        wav_file_pairs.append((wav_file, mid_file))

  output_name = os.path.join(FLAGS.output_dir,
                                   '{}.{}-tfrecord-in-2020-06-28'.format(data_type, desc))

  if data_type == 'train':
    with tf.python_io.TFRecordWriter(output_name) as writer:
      for idx, pair in enumerate(wav_file_pairs):
        print('{} of {}: {}'.format(idx, len(wav_file_pairs), pair[0]))
        # load the wav data
        wav_data = tf.gfile.Open(pair[0], 'rb').read()
        # load the midi data and convert to a notesequence
        ns = midi_io.midi_file_to_note_sequence(pair[1])
        for example in audio_label_data_utils.process_record(
          wav_data, ns, pair[0], FLAGS.min_length, FLAGS.max_length,
          FLAGS.sample_rate, pre_audio_transform=pre_audio_transform, hparams=hparams):
          writer.write(example.SerializeToString())
  elif data_type == 'test':
    with tf.python_io.TFRecordWriter(output_name) as writer:
      for idx, pair in enumerate(wav_file_pairs):
        print('{} of {}: {}'.format(idx, len(wav_file_pairs), pair[0]))
        # load the wav data and resample it.
        samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
        wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

        # load the midi data and convert to a notesequence
        ns = midi_io.midi_file_to_note_sequence(pair[1])

        if pre_audio_transform and hparams:
          wav_data = audio_transform.transform_wav_audio(wav_data, hparams)
        example = audio_label_data_utils.create_example(pair[0], ns, wav_data)
        writer.write(example.SerializeToString())


def main(unused_argv):
  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  config = configs.CONFIG_MAP[FLAGS.config]

  # maestro dataset
  if FLAGS.dataset == 'v1' or FLAGS.dataset == 'v2':
    train_data_set, test_data_set, validation_data_set = parse_config_json(FLAGS.dataset)

    # train: Need process the wav examples if necessary
    if FLAGS.mode == 'all' or FLAGS.mode == 'train':
      # enable pre audio transform for v1/v2 train dataset
      config.hparams.transform_audio = True

      for year, dataset in train_data_set.items():
        generate_maestro_dataset(FLAGS.dataset, 'train', year, dataset, FLAGS.preprocess_examples, config.hparams)

    if FLAGS.mode == 'all' or FLAGS.mode == 'test':
      config.hparams.transform_audio = True
      for year, dataset in test_data_set.items():
        generate_maestro_dataset(FLAGS.dataset, 'test', year, dataset, FLAGS.preprocess_examples, config.hparams)
        #generate_maestro_dataset(FLAGS.dataset, 'test', year, dataset)

    if FLAGS.mode == 'all' or FLAGS.mode == 'validation':
      config.hparams.transform_audio = True
      for year, dataset in validation_data_set.items():
        generate_maestro_dataset(FLAGS.dataset, 'validation', year, dataset, FLAGS.preprocess_examples, config.hparams)

  elif FLAGS.dataset == 'self':
    config.hparams.transform_audio = False
    if FLAGS.mode == 'all' or FLAGS.mode == 'train':
      generate_self_train_set()

    if FLAGS.mode == 'all' or FLAGS.mode == 'test':
      generate_self_test_set()
  elif FLAGS.dataset == 'flat':
    config.hparams.transform_audio = False
    if FLAGS.mode == 'all' or FLAGS.mode == 'train' or FLAGS.mode == 'test':
      generate_flat_set(FLAGS.mode, 'flat-midi-noised', FLAGS.preprocess_examples, config.hparams)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
