"""Real time transcribe."""

import collections
import os
import io
import sys
import math
import time
import numpy as np
import pickle
import base64
from threading import Thread
import zlib
import six
import redis
import tensorflow as tf
import functools
import signal
from deepiano.server import utils

import logging

formatter = logging.Formatter('[%(asctime)s:%(worker_id)s:%(levelname)s:%(queue_key)s:%(request_id)s:%(task_key)s] %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

for handler in logger.handlers:
    handler.setFormatter(formatter)

extra = {'request_id': '-', 'worker_id': '-', 'task_key': '-'}
logger = logging.LoggerAdapter(logger, extra)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
tf_session = tf.Session(config=sess_config)


from deepiano.wav2mid import configs
from deepiano.wav2mid import constants
from deepiano.wav2mid import data
from deepiano.wav2mid import train_util
from deepiano.music import midi_io
from deepiano.music import constants as music_constants
from deepiano.protobuf import music_pb2
from deepiano.server import config as server_config

from google.protobuf.json_format import MessageToDict

redis_pool = redis.ConnectionPool.from_url(server_config.redis_uri)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', '/data/models/tmp',
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
tf.app.flags.DEFINE_float(
    'offset_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'chunk_frames', 200,
    'frames per chunk.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_string(
    'input', '../../data/test/record.wav',
    'input wav file path')
tf.app.flags.DEFINE_string(
    'task_key', 'transcribe_tasks_v5',
    'is using test model')
tf.app.flags.DEFINE_integer(
    'timeout', 200,
    'task timeout')



ChunkPrediction = collections.namedtuple(
    'ChunkPrediction',
    ('frame_predictions', 'onset_predictions', 'offset_predictions', 'velocity_values', 'onset_probs_flat'))


pid_file = '/tmp/transcribe_worker_pid'

worker_id = '{}:{}'.format(os.uname().nodename, os.getpid())
logger.extra['worker_id'] = worker_id

def get_spectrogram_hash(spectrogram):
  # Compute a hash of the spectrogram, save it as an int64.
  # Uses adler because it's fast and will fit into an int (md5 is too large).
  spectrogram_serialized = six.BytesIO()
  np.save(spectrogram_serialized, spectrogram)
  spectrogram_hash = np.int64(zlib.adler32(spectrogram_serialized.getvalue()))
  return spectrogram_hash


class Transcriber:
  def __init__(self, id=None):
    self.id = id or '{}:{}:{}'.format(os.uname().nodename, os.getpid(), int(time.time() * 1000))
    self.busy = False
    self.queue_key = FLAGS.task_key
    logger.extra['queue_key'] = self.queue_key
    logger.info('Transcriber init')
    self.task_key = None
    self.terminated = False

    config = configs.CONFIG_MAP[FLAGS.config]
    hparams = config.hparams
    self.hparams = hparams

    hparams.use_cudnn = tf.test.is_gpu_available()
    hparams.parse(FLAGS.hparams)
    hparams.batch_size = 1
    hparams.truncated_length_secs = 0

    if sys.platform == 'darwin':
      # Force cqt in MacOS since the default 'mel' will crash
      hparams.spec_type = 'cqt'

    self.fps = hparams.sample_rate / hparams.spec_hop_length

    self.sess = tf_session
    with self.sess.graph.as_default():

      example = tf.placeholder(tf.string, None)
      self.example_placeholder = example

      self.input_data = data.provide_single(
        example=example,
        hparams=hparams,
        is_training=False)

      self.estimator = train_util.create_estimator(config.model_fn,
                                              os.path.expanduser(FLAGS.model_dir),
                                              hparams)


      self.sess.run([
          tf.initializers.global_variables(),
          tf.initializers.local_variables()
      ])

      signal.signal(signal.SIGINT, self.signal_handler)
      signal.signal(signal.SIGUSR1, self.signal_handler)
      signal.signal(signal.SIGTERM, self.signal_handler)

  def __str__(self):
    return '<Transcriber {}, {}>'.format(self.id, '\033[1;31mBUSY\033[1;m' if self.busy else '\033[1;32mIDLE\033[1;m')


  def signal_handler(self, signum, frame):
      logger.warn('Received signal: {}, terminating'.format(signum))
      if os.path.isfile(pid_file):
        os.remove(pid_file)

      self.terminated = True


  def preprocess(self, spec, hparams):
    logger.info('expand dims')
    spec = np.expand_dims(spec, -1)
    logger.info('get n frame')
    self.fps = hparams.sample_rate / hparams.spec_hop_length
    length = spec.shape[0]
    logger.info('get spec hash')
    spectrogram_hash = get_spectrogram_hash(spec)

    logger.info('generate data')
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
    logger.info('done')

    return features, labels



  def gen_input(self):
    r = redis.Redis(connection_pool=redis_pool)
    idle_start = time.time()
    last_log = idle_start

    # save pid in /tmp
    pid = os.getpid()
    ppid = os.getppid()
    with open('/tmp/transcribe_worker_pid', 'w') as f:
      f.write('{},{}'.format(str(pid), str(ppid)))

      logger.info('init...pid: %d, pid saved in %s' % (pid, pid_file))

    while True:
      if self.terminated:
          break
      logger.extra['request_id'] = '-'
      logger.extra['task_key'] = '-'

      task = r.brpop(self.queue_key, 2)

      if time.time() - last_log > 10:
          logger.info('IDLE for {:.1f} seconds'.format(time.time() - idle_start))
          last_log = time.time()

      if task is None:
          pass
      else:
          self.busy = True
          _, task_value = task
          task = pickle.loads(task_value)

          logger.extra['request_id'] = task.get('request_id', '-')
          logger.extra['task_key'] = task.get('key', '-')

          self.task_key = task['key']
          spec = task.pop('spec')

          r.set(self.task_key, pickle.dumps(task), 300)

          logger.info('task get {}'.format(self.task_key))
          if time.time() - task['timestamp'] > FLAGS.timeout - 1: # 1 sec for processing
              logger.info('task timeout, droped')
              continue

          logger.info('start preprocess')
          spec = utils.decode_array(spec)
          logger.info('decoded')
          item = self.preprocess(spec, self.hparams)
          logger.info('end preprocess')
          yield item
          self.busy = False
          idle_start = time.time()
          last_log = idle_start

  def input_fn(self, params):
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

    dataset = tf.data.Dataset.from_generator(self.gen_input, output_types, output_shapes)
    return dataset


  def chunk_func(self):
    for prediction in self.estimator.predict(
      self.input_fn,
      checkpoint_path=FLAGS.checkpoint_path,
      yield_single_examples=False):

      frame_predictions = prediction['frame_probs_flat'] > FLAGS.frame_threshold
      onset_predictions = prediction['onset_probs_flat'] > FLAGS.onset_threshold
      offset_predictions = None #prediction['offset_probs_flat'] > FLAGS.offset_threshold
      velocity_values = prediction['velocity_values_flat']

      logger.info('yield results')
      yield ChunkPrediction(
        frame_predictions=frame_predictions,
        onset_predictions=onset_predictions,
        onset_probs_flat=prediction['onset_probs_flat'],
        offset_predictions=offset_predictions,
        velocity_values=velocity_values)


  def pianoroll_to_note_sequence(self,
                                 # frames_per_second,
                                 min_duration_ms,
                                 velocity=70,
                                 instrument=0,
                                 program=0,
                                 qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                                 min_midi_pitch=music_constants.MIN_MIDI_PITCH):

    """Convert frames to a NoteSequence."""
    # frame_length_seconds = 1 / frames_per_second
    frame_length_seconds = 1 / self.fps

    pitch_start_step = {}
    onset_velocities = velocity * np.ones(
        constants.MAX_MIDI_PITCH, dtype=np.int32)

    total_frames = 0

    def end_pitch(pitch, i):
      """End an active pitch."""
      start_time = pitch_start_step[pitch] * frame_length_seconds
      end_time = (i + total_frames) * frame_length_seconds

      if (end_time - start_time) * 1000 >= min_duration_ms:
        note = sequence.notes.add()
        note.start_time = start_time
        note.end_time = end_time
        note.pitch = pitch + min_midi_pitch
        note.velocity = onset_velocities[pitch]
        note.instrument = instrument
        note.program = program

      del pitch_start_step[pitch]


    def unscale_velocity(velocity):
      """Translates a velocity estimate to a MIDI velocity value."""
      unscaled = max(min(velocity, 1.), 0) * 80. + 10.
      if math.isnan(unscaled):
        return 0
      return int(unscaled)


    def process_active_pitch(pitch, i, onset_predictions, velocity_values):
      """Process a pitch being active in a given frame."""
      if pitch not in pitch_start_step:
        if onset_predictions is not None:
          # If onset predictions were supplied, only allow a new note to start
          # if we've predicted an onset.
          if onset_predictions[i, pitch]:
            pitch_start_step[pitch] = i + total_frames
            onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])
          else:
            # Even though the frame is active, the onset predictor doesn't
            # say there should be an onset, so ignore it.
            pass
        else:
          pitch_start_step[pitch] = i + total_frames

        # if pitch in pitch_start_step:
        #   logger.info('note_on: %d', pitch)
      else:
        if onset_predictions is not None:
          # pitch is already active, but if this is a new onset, we should end
          # the note and start a new one.
          if onset_predictions[i, pitch] and not onset_predictions[i - 1, pitch]:
            end_pitch(pitch, i)
            pitch_start_step[pitch] = i + total_frames
            onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])


    def process_chunk(chunk_prediction):
      nonlocal total_frames
      nonlocal frame_length_seconds

      total_frames = 0
      frame_length_seconds = 1 / self.fps

      frames = chunk_prediction.frame_predictions
      onset_predictions = chunk_prediction.onset_predictions
      offset_predictions = chunk_prediction.offset_predictions
      velocity_values = chunk_prediction.velocity_values

      # Add silent frame at the end so we can do a final loop and terminate any
      # notes that are still active.
      frames = np.append(frames, [np.zeros(frames[0].shape)], 0)

      if velocity_values is None:
        velocity_values = velocity * np.ones_like(frames, dtype=np.int32)

      if onset_predictions is not None:
        onset_predictions = np.append(onset_predictions,
                                              [np.zeros(onset_predictions[0].shape)], 0)
        # Ensure that any frame with an onset prediction is considered active.
        frames = np.logical_or(frames, onset_predictions)

      if offset_predictions is not None:
        # If the frame and offset are both on, then turn it off
        offset_predictions = np.append(offset_predictions,
                                                       [np.zeros(offset_predictions[0].shape)], 0)
        frames[np.where(np.logical_and(frames > 0, offset_predictions > 0))] = 0

      for i, frame in enumerate(frames):
        for pitch, active in enumerate(frame):
          if active:
            process_active_pitch(pitch, i, onset_predictions, velocity_values)
          elif pitch in pitch_start_step:
            end_pitch(pitch, i)

      total_frames += len(frames)

    for chunk in self.chunk_func():
      sequence = music_pb2.NoteSequence()
      sequence.tempos.add().qpm = qpm
      sequence.ticks_per_quarter = music_constants.STANDARD_PPQ

      process_chunk(chunk)

    # Add silent frame at the end so we can do a final loop and terminate any
    # notes that are still active.
    # frames = np.append(frames, [np.zeros(frames[0].shape)], 0)

    # if onset_predictions is not None:
    #   onset_predictions = np.append(onset_predictions,
    #                                 [np.zeros(onset_predictions[0].shape)], 0)

    # if offset_predictions is not None:
    #   offset_predictions = np.append(offset_predictions,
    #                                  [np.zeros(offset_predictions[0].shape)], 0)

    # total_frames += 1

      sequence.total_time = total_frames * frame_length_seconds
      if sequence.notes:
        assert sequence.total_time >= sequence.notes[-1].end_time

      logger.info('enqueue result: ' + self.task_key)
      if self.task_key:
        # for debugging
        # midi_io.sequence_proto_to_midi_file(sequence, '/tmp/{}.midi'.format(self.task_key))
        self.enqueue_result(sequence, chunk)


  def _run(self):
    with self.sess.graph.as_default():
      logger.info('Transcriber Start')
      sequence_prediction = self.pianoroll_to_note_sequence(
          # frames_per_second=data.hparams_frames_per_second(self.hparams),
          min_duration_ms=0,
          min_midi_pitch=constants.MIN_MIDI_PITCH,
      )


  def run(self):
    t = Thread(target=self._run)
    t.start()


  def enqueue_result(self, sequence, chunk):
    pretty_midi_object = midi_io.note_sequence_to_pretty_midi(sequence)
    midi_b64 = ''
    with io.BytesIO() as buf:
      pretty_midi_object.write(buf)
      midi_b64 = base64.b64encode(buf.getvalue()).decode()

    result = {
        'midi': midi_b64,
        'preds' : {'onsets': utils.encode_array(chunk.onset_probs_flat)},
        'sequence': MessageToDict(sequence),
    }
    r = redis.Redis(connection_pool=redis_pool)
    task = pickle.loads(r.get(self.task_key))
    task['result'] = result
    data = pickle.dumps(task)
    r.lpush(task['group_id'], data)
    r.publish('transcribe-channel', data)
    r.expire(self.task_key, 60)


if __name__ == '__main__':
  transcriber = Transcriber()
  transcriber.run()
