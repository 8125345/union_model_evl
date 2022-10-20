"""Export estimator as a saved_model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import tensorflow as tf
from deepiano.wav2mid import configs_tflite as configs
from deepiano.wav2mid import train_util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', '../../data/models/test-lite-lstm',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
  'hparams',
  '',
  'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
  'log', 'DEBUG',
  'The threshold for what messages will be logged: '
  'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_string(
  'input', '../../data/test/record.wav',
  'input wav file path')
tf.app.flags.DEFINE_integer(
  'chunk_frames', 8,
  'chunk frame count')

tf.app.flags.DEFINE_string('output_model_path', 'converted_model.tflite',
                           'Path to .tflite')

# dummy flags
tf.app.flags.DEFINE_string('type', 'export',
                           'Type for export tflite')

tf.app.flags.DEFINE_string('name', 'export_1',
                           'Name for export tflite')

def serving_input_receiver_fn():
  """Serving input_fn that builds features from placeholders

  Returns
  -------
  tf.estimator.export.ServingInputReceiver
  """
  config = configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams

  hparams.use_cudnn = tf.test.is_gpu_available()
  hparams.parse(FLAGS.hparams)
  hparams.truncated_length_secs = 0

  chunk_frames = FLAGS.chunk_frames

  input_spec = tf.placeholder(dtype=tf.float32, shape=[chunk_frames * 229], name='input_spec')

  # feature_shape = [1, -1, 229, 1]
  input_2d = tf.reshape(input_spec, [chunk_frames, 229])
  feature_spec = tf.expand_dims(tf.expand_dims(input_2d, 0), -1)

  return tf.estimator.export.ServingInputReceiver({'spec': feature_spec}, {'spec': input_spec})


ExportSession = collections.namedtuple(
  'ExportSession',
  ('estimator'))


def init_export_session():
  tf.logging.set_verbosity(FLAGS.log)

  config = configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams

  tf.logging.info('hparams: %s', hparams)

  hparams.use_cudnn = False
  hparams.parse(FLAGS.hparams)
  hparams.truncated_length_secs = 0

  with tf.Graph().as_default():
    estimator = train_util.create_estimator(config.model_fn,
                                            os.path.expanduser(FLAGS.model_dir),
                                            hparams)
    return ExportSession(
      estimator=estimator)


if __name__ == '__main__':
  session = init_export_session()
  ret = session.estimator.export_saved_model('../../data/models/saved_model', serving_input_receiver_fn)
  print(ret)

  # Converting a SavedModel.
  converter = tf.lite.TFLiteConverter.from_saved_model(ret)
  converter.post_training_quantize = True
  converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  tflite_model = converter.convert()
  open(FLAGS.output_model_path, "wb").write(tflite_model)
