"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import time
import os
import glob
import logging
from shutil import copyfile

import tensorflow as tf

from deepiano.music import constants as music_constants
from deepiano.music import midi_io
from deepiano.protobuf import music_pb2
from deepiano.server import wav2spec
from deepiano.wav2mid import configs_tflite as configs
from deepiano.wav2mid import constants
from deepiano.wav2mid import data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_path', '/Users/xyz/tf_model/converted_model_50000_full_noise_2020_02_19.tflite',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'hparams',
    '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.3,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_string(
    'input', '../../data/test/record.wav',
    'input wav file path')
tf.app.flags.DEFINE_integer(
  'chunk_padding', 3,
  'chunk_padding')


tf.app.flags.DEFINE_string('input_dir', '../../data/AI_tagging_for_test',
                           'Directory where the wav & midi labels are')

tf.app.flags.DEFINE_string('output_dir', '../../data/test/predict_1',
                           'Directory where the predicted midi & midi labels will be placed.')


tf.app.flags.DEFINE_string(
    'input_tfrecord', '../../data/maestro/maestro-v1.0.0-tfrecord/test.*',
    'input wav file path')

# dummy flags
tf.app.flags.DEFINE_string('type', 'predict',
                           'Type for predict')

tf.app.flags.DEFINE_string('name', 'predict_1',
                           'Name for predict')


ChunkPrediction = collections.namedtuple(
    'ChunkPrediction',
    ('onset_predictions', 'velocity_values'))


all_mp3_dir = []

def pianoroll_to_note_sequence(chunk_func_c,
                               frames_per_second,
                               min_duration_ms,
                               velocity=70,
                               instrument=0,
                               program=0,
                               qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                               min_midi_pitch=music_constants.MIN_MIDI_PITCH,
                               wav_file=None):

  frame_length_seconds = 1 / frames_per_second

  sequence = music_pb2.NoteSequence()
  sequence.tempos.add().qpm = qpm
  sequence.ticks_per_quarter = music_constants.STANDARD_PPQ

  note_duration = frame_length_seconds * 3  # to remove redundant same midi
  total_frames = FLAGS.chunk_padding  # left padding

  last_note = {} # {'pitch': time}

  def unscale_velocity(velocity):
    unscaled = max(min(velocity, 1.), 0) * 80. + 10.
    if math.isnan(unscaled):
      return 0
    return int(unscaled)

  def process_chunk(chunk_prediction):
    nonlocal total_frames

    onset_predictions = chunk_prediction.onset_predictions
    velocity_values = chunk_prediction.velocity_values

    for i, onset in enumerate(onset_predictions):
      for pitch, active in enumerate(onset):
        if active:
          time = (total_frames + i) * frame_length_seconds
          pitch = pitch + min_midi_pitch
          if time - last_note.get(pitch, -1) > note_duration:
            note = sequence.notes.add()
            note.start_time = time
            note.end_time = time + note_duration
            note.pitch = pitch
            note.velocity = unscale_velocity(velocity_values[i, pitch] if velocity_values else velocity)
            note.instrument = instrument
            note.program = program

            last_note[note.pitch] = note.start_time
            print('note:', note.pitch)

    total_frames += len(onset_predictions)

  print('begin process chunk')
  for chunk in chunk_func_c(wav_file):
    process_chunk(chunk)

  print('end process chunk')
  sequence.total_time = total_frames * frame_length_seconds
  return sequence


def generate_predict_set(input_dirs):
  predict_file_pairs = []
  logging.info('generate_predict_set %s' % input_dirs)
  if len(input_dirs) == 0:
    input_dirs = [FLAGS.input_dir]
  for directory in input_dirs:
    # path = os.path.join(FLAGS.input_dir, directory)
    path = directory
    logging.info('generate_predict_set! path: %s' % path)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name, _ = os.path.splitext(wav_file)
      mid_file = base_name + '.mid'
      if os.path.isfile(mid_file):
        predict_file_pairs.append((wav_file, mid_file))
  logging.info('generate_predict_set! %d' % len(predict_file_pairs))
  return predict_file_pairs


def iter_files(rootDir):
  for root, dirs, files in os.walk(rootDir):
    if dirs != []:
      for dirname in dirs:
        full_dirname = os.path.join(root, dirname)
        all_mp3_dir.append(full_dirname)
        iter_files(full_dirname)


def transcribe_chunked(argv):
  del argv

  tf.logging.set_verbosity(FLAGS.log)
  tf.logging.info('init...')

  config = configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams

  hparams.use_cudnn = tf.test.is_gpu_available()
  hparams.parse(FLAGS.hparams)
  hparams.truncated_length_secs = 0

  interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  chunk_frames = int(input_details[0]['shape'].tolist()[0] / 229)
  chunk_padding = FLAGS.chunk_padding
  frames_nopadding = chunk_frames - chunk_padding * 2
  assert frames_nopadding > 0

  #print('chunk_frames: %d chunk_padding: %d frames_nopadding: %d' % (chunk_frames, chunk_padding, frames_nopadding))
  print(input_details)
  print(output_details)
  #print(input_details[0]['shape'].tolist()[2])
  #return

  # filename = FLAGS.input

  # input_midi_dir = FLAGS.input_dir
  # output_midi_dir = FLAGS.output_dir

  def gen_input(wav_filename):
    spec = wav2spec.wav2spec(wav_filename)
    for i in range(chunk_padding, spec.shape[0], frames_nopadding):
      start = i - chunk_padding
      end  = i + frames_nopadding + chunk_padding
      chunk_spec = spec[start:end]
      if chunk_spec.shape[0] == chunk_padding * 2 + frames_nopadding:
        input_item = {
          'spec': chunk_spec.flatten()
        }
        yield input_item


  def chunk_func(wav_filename):
    start_time = time.time()
    print(wav_filename)
    for input_item in gen_input(wav_filename):
      interpreter.set_tensor(input_details[0]['index'], input_item['spec'])
      interpreter.invoke()

      onset_probs_flat = interpreter.get_tensor(output_details[0]['index'])
      # velocity_values_flat = interpreter.get_tensor(output_details[1]['index'])

      if chunk_padding > 0:
        onset_probs_flat = onset_probs_flat[chunk_padding:-chunk_padding]
        # velocity_values_flat = velocity_values_flat[chunk_padding:-chunk_padding]

      onset_predictions = onset_probs_flat > FLAGS.onset_threshold
      # velocity_values = velocity_values_flat

      yield ChunkPrediction(
        onset_predictions=onset_predictions,
        velocity_values=None)

    # logging.info('predict time: ', time.time() - start_time)

  iter_files(FLAGS.input_dir)
  predict_file_pairs = generate_predict_set(all_mp3_dir)

  logging.info('predict start! %d' % len(predict_file_pairs))

  for wav_file, label_midi_file in predict_file_pairs:

    if not os.path.isdir(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)

    _, label_midi_file_name = os.path.split(label_midi_file)
    copyed_label_midi_file = os.path.join(FLAGS.output_dir, label_midi_file_name + '.label.midi')
    copyfile(label_midi_file, copyed_label_midi_file)

    predicted_label_midi_file = os.path.join(FLAGS.output_dir, label_midi_file_name + '.predicted.midi')
    if os.path.isfile(predicted_label_midi_file):
      continue
    sequence_prediction = pianoroll_to_note_sequence(
        chunk_func,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        min_midi_pitch=constants.MIN_MIDI_PITCH,
        wav_file=wav_file
    )

    midi_io.sequence_proto_to_midi_file(sequence_prediction, predicted_label_midi_file)
  print('predict end!')


def console_entry_point():
  tf.app.run(transcribe_chunked)


if __name__ == '__main__':
  console_entry_point()

