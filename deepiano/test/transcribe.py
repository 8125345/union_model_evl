"""Transcribe a recording of piano audio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import zlib
import six
import numpy as np

from deepiano.wav2mid import configs, configs_tflite
from deepiano.wav2mid import constants
from deepiano.wav2mid import data
from deepiano.wav2mid import train_util
from deepiano.music import midi_io
from deepiano.music import sequences_lib
import tensorflow as tf
from deepiano.server.wav2spec import wav2spec

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
tf_session = tf.Session(config=sess_config)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', '../../data/models/maestro',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
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
    'log', 'DEBUG',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_boolean(
    'lite', False,
    'Use lite config')

tf.app.flags.DEFINE_string(
    'input', '../../data/test/record.wav',
    'input wav file path')


def pred2mid(prediction, hparams, frame_threshold, onset_threshold):
  """Transcribes an audio file."""
  frame_predictions = prediction['frame_probs_flat'] > frame_threshold
  onset_predictions = prediction['onset_probs_flat'] > onset_threshold
  velocity_values = prediction['velocity_values_flat']

  sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
      frame_predictions,
      frames_per_second=data.hparams_frames_per_second(hparams),
      min_duration_ms=0,
      min_midi_pitch=constants.MIN_MIDI_PITCH,
      onset_predictions=onset_predictions,
      velocity_values=velocity_values)

  return sequence_prediction


def get_output_defs():
  features = data.FeatureTensors(
    spec=tf.float32,
    length=tf.int32,
    sequence_id=tf.string,
    spectrogram_hash=tf.int64)

  labels = data.LabelTensors(
    labels=tf.float32,
    label_weights=tf.float32,
    onsets=tf.float32,
    offsets=tf.float32,
    velocities=tf.float32,
    note_sequence=tf.string)

  output_types = (features, labels)

  features_shape = data.FeatureTensors(
    spec=tf.TensorShape([1, None, 229, 1]),
    length=tf.TensorShape([1]),
    sequence_id=tf.TensorShape([1]),
    spectrogram_hash=tf.TensorShape([1]))

  labels_shape = data.LabelTensors(
    labels=tf.TensorShape([1, None, 88]),
    label_weights=tf.TensorShape([1, None, 88]),
    onsets=tf.TensorShape([1, None, 88]),
    offsets=tf.TensorShape([1, None, 88]),
    velocities=tf.TensorShape([1, None, 88]),
    note_sequence=tf.TensorShape([1]))

  output_shapes = (features_shape, labels_shape)

  return output_types, output_shapes


def get_spectrogram_hash(spectrogram):
  # Compute a hash of the spectrogram, save it as an int64.
  # Uses adler because it's fast and will fit into an int (md5 is too large).
  spectrogram_serialized = six.BytesIO()
  np.save(spectrogram_serialized, spectrogram)
  spectrogram_hash = np.int64(zlib.adler32(spectrogram_serialized.getvalue()))
  return spectrogram_hash


def preprocess(spec, hparams):
  print(time.time(), 'expand dims')
  spec = np.expand_dims(spec, -1)
  print(time.time(), 'get n frame')
  length = spec.shape[0]
  print(time.time(), 'get spec hash')
  spectrogram_hash = get_spectrogram_hash(spec)

  print(time.time(), 'generate data')
  features = data.FeatureTensors(
    spec=np.expand_dims(spec, 0),
    length=np.expand_dims(length, 0),
    sequence_id=np.array(['xxx'], dtype=object),
    spectrogram_hash=np.expand_dims(spectrogram_hash, 0),
  )

  lable_shape = (1, length, 88)
  labels = data.LabelTensors(
    labels=np.zeros(lable_shape, dtype=np.float32),
    label_weights=np.zeros(lable_shape, dtype=np.float32),
    onsets=np.zeros(lable_shape, dtype=np.float32),
    offsets=np.zeros(lable_shape, dtype=np.float32),
    velocities=np.zeros(lable_shape, dtype=np.float32),
    note_sequence=np.array([b''], dtype=object),
  )
  print(time.time(), 'done')

  return features, labels


def transcribe(filename, hparams):

  spec = wav2spec(filename)

  sess = tf_session
  with sess.graph.as_default():
    estimator = train_util.create_estimator(config.model_fn,
                                        os.path.expanduser(FLAGS.model_dir),
                                        hparams)
    sess.run([
      tf.initializers.global_variables(),
      tf.initializers.local_variables()
    ])

    def gen_input():
      item = preprocess(spec, hparams)
      yield item

    def input_fn(params):
      output_types, output_shapes = get_output_defs()
      dataset = tf.data.Dataset.from_generator(gen_input, output_types, output_shapes)
      return dataset

    prediction = list(estimator.predict(
      input_fn,
      checkpoint_path=None,
      yield_single_examples=False))[0]

    sequence_prediction = pred2mid(prediction, hparams,
                                           FLAGS.frame_threshold,
                                           FLAGS.onset_threshold)

    midi_filename = filename + '.midi'
    midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)


if __name__ == '__main__':
  config = configs_tflite.CONFIG_MAP[FLAGS.config] if FLAGS.lite else configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams
  hparams.use_cudnn = tf.test.is_gpu_available()
  hparams.parse(FLAGS.hparams)
  hparams.batch_size = 1
  hparams.truncated_length_secs = 0

  transcribe(FLAGS.input, hparams)
