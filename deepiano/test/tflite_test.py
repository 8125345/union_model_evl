"""Real time transcribe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import math
import time
import numpy as np

from deepiano.server import wav2spec
from deepiano.wav2mid import configs_lite as configs
from deepiano.wav2mid import constants
from deepiano.wav2mid import data
from deepiano.wav2mid import split_audio_and_label_data
from deepiano.wav2mid import train_util
from deepiano.music import midi_io
from deepiano.music import sequences_lib
from deepiano.music import constants as music_constants
from deepiano.protobuf import music_pb2
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_path', './converted_model.tflite',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'hparams',
    '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.5,
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


ChunkPrediction = collections.namedtuple(
    'ChunkPrediction',
    ('onset_predictions', 'velocity_values'))


def pianoroll_to_note_sequence(chunk_func,
                               frames_per_second,
                               min_duration_ms,
                               velocity=70,
                               instrument=0,
                               program=0,
                               qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                               min_midi_pitch=music_constants.MIN_MIDI_PITCH):

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

  for chunk in chunk_func():
    process_chunk(chunk)

  sequence.total_time = total_frames * frame_length_seconds
  return sequence


def transcribe_chunked(filename):
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

  print(input_details)
  print(output_details)

  def gen_input():
    spec = wav2spec.wav2spec(filename)
    for i in range(0, spec.shape[0], frames_nopadding):
      chunk_spec = spec[i - chunk_padding: i + frames_nopadding + chunk_padding]
      if chunk_spec.shape[0] == chunk_padding * 2 + frames_nopadding:
        input_item = {
          'spec': chunk_spec.flatten()
        }
        yield input_item


  def chunk_func():
    start_time = time.time()

    for input_item in gen_input():
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

    print('predict time: ', time.time() - start_time)


  sequence_prediction = pianoroll_to_note_sequence(
      chunk_func,
      frames_per_second=data.hparams_frames_per_second(hparams),
      min_duration_ms=0,
      min_midi_pitch=constants.MIN_MIDI_PITCH)

  midi_io.sequence_proto_to_midi_file(sequence_prediction, filename + '.midi')


if __name__ == '__main__':
  transcribe_chunked(FLAGS.input)

