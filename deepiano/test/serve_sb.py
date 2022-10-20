"""Reload and serve a saved model"""
import base64
import collections
import io
import json
import math
import sys
import time
import zlib
from pathlib import Path
from threading import Thread

import numpy as np
import redis
import six
import tensorflow as tf
from deepiano.wav2mid import constants
from deepiano.music import midi_io
from deepiano.music import constants as music_constants
from deepiano.protobuf import music_pb2
from deepiano.wav2mid import configs
from deepiano.wav2mid.data import FeatureTensors
from tensorflow.contrib import predictor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', '/tmp',
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

tf.app.flags.DEFINE_string(
  'input', '../../data/test/record.wav',
  'input wav file path')

redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=False)

ChunkPrediction = collections.namedtuple(
  'ChunkPrediction',
  ('frame_predictions', 'onset_predictions', 'offset_predictions', 'velocity_values', 'onset_probs_flat'))


def get_spectrogram_hash(spectrogram):
  # Compute a hash of the spectrogram, save it as an int64.
  # Uses adler because it's fast and will fit into an int (md5 is too large).
  spectrogram_serialized = six.BytesIO()
  np.save(spectrogram_serialized, spectrogram)
  spectrogram_hash = np.int64(zlib.adler32(spectrogram_serialized.getvalue()))
  return spectrogram_hash


class TfServing:
  def __init__(self, id=None):
    self.id = id or int(time.time() * 1000)
    self.busy = False
    self.log('TfServing init')
    self.queue_key = 'transcribe_tasks_v3'
    self.task_key = None
    self.busy = False
    tf.logging.set_verbosity(FLAGS.log)
    tf.logging.info('init...')

    self.tic = time.time()
    self.toc = time.time()

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

    export_dir = '../../data/models/saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    self.predict_fn = predictor.from_saved_model(latest)

  def __str__(self):
    return '<TfServing {}, {}>'.format(self.id, '\033[1;31mBUSY\033[1;m' if self.busy else '\033[1;32mIDLE\033[1;m')

  def log(self, *args, **kwargs):
    print(self, time.time(), *args, **kwargs)

  def preprocess(self, spec, hparams):
    # spec_shape = [-1, 229]
    return {
      'spec': spec,
    }


  def gen_input(self):
    r = redis.Redis(connection_pool=redis_pool)
    while True:
      self.log('wait for next task')
      task = r.brpop(self.queue_key, 1)
      if task is None:
        pass
      else:
        self.tic = time.time()

        _, task_value = task
        task = json.loads(task_value)
        self.task_key, spec, timestamp = task['key'], task['spec'], task['timestamp']
        self.log('task get {}'.format(self.task_key))
        # if time.time() - task['timestamp'] > server_config.timeout - 1: # 1 sec for processing
        #     self.log('task timeout, droped')
        #     continue

        self.log('start preprocess')
        spec = base64.decodebytes(spec.encode())
        with io.BytesIO() as buf:
          buf.write(spec)
          buf.seek(0)
          spec = np.load(buf)
          self.log('decoded')
          item = self.preprocess(spec, self.hparams)
          self.log('end preprocess')
          yield item


  def fake_input(self):
    from deepiano.server.wav2spec import wav2spec
    spec = wav2spec(FLAGS.input)
    item = self.preprocess(spec, self.hparams)
    yield item


  def chunk_func(self):
    for nb in self.gen_input():
      pred = self.predict_fn(nb)
      frame_predictions = pred['frame_probs_flat'] > FLAGS.frame_threshold
      onset_predictions = pred['onset_probs_flat'] > FLAGS.onset_threshold if pred.get('onset_probs_flat') else None
      velocity_values = pred['velocity_values_flat']

      self.log('yield results')
      yield ChunkPrediction(
        frame_predictions=frame_predictions,
        onset_predictions=onset_predictions,
        onset_probs_flat=pred.get('onset_probs_flat'),
        offset_predictions=None,
        velocity_values=velocity_values)


  def pianoroll_to_note_sequence(self,
                                 # frames_per_second,
                                 min_duration_ms,
                                 velocity=70,
                                 instrument=0,
                                 program=0,
                                 qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                                 min_midi_pitch=constants.MIN_MIDI_PITCH):

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
          onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])

        # if pitch in pitch_start_step:
        #   tf.logging.info('note_on: %d', pitch)
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

      if velocity_values is None:
        velocity_values = velocity * np.ones_like(frames, dtype=np.int32)

      if onset_predictions is not None:
        # Ensure that any frame with an onset prediction is considered active.
        frames = np.logical_or(frames, onset_predictions)

      if offset_predictions is not None:
        # If the frame and offset are both on, then turn it off
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

      sequence.total_time = total_frames * frame_length_seconds
      if sequence.notes:
        assert sequence.total_time >= sequence.notes[-1].end_time
      # print(sequence, chunk.onset_probs_flat)

      print(self.task_key)
      self.toc = time.time()

      print('Cost time in serving: {}s'.format((self.toc - self.tic)))

      if self.task_key:
        # for debugging
        # midi_io.sequence_proto_to_midi_file(sequence, '/tmp/{}.midi'.format(self.task_key))
        self.enqueue_result(sequence, chunk)

  def enqueue_result(self, sequence, chunk):
    pretty_midi_object = midi_io.note_sequence_to_pretty_midi(sequence)
    midi_b64 = ''
    with io.BytesIO() as buf:
      pretty_midi_object.write(buf)
      midi_b64 = base64.b64encode(buf.getvalue()).decode()

    result = {
      'midi': midi_b64,
      'preds': {'onsets': chunk.onset_probs_flat.tolist()},
    }
    r = redis.Redis(connection_pool=redis_pool)
    r.lpush(self.task_key, json.dumps(result))
    r.expire(self.task_key, 60)

  def _run(self):
    # with self.sess.graph.as_default():
    self.log('TfServing Start')
    self.pianoroll_to_note_sequence(
      # frames_per_second=data.hparams_frames_per_second(self.hparams),
      min_duration_ms=0,
      min_midi_pitch=constants.MIN_MIDI_PITCH,
    )

  def run(self):
    t = Thread(target=self._run)
    t.start()


if __name__ == '__main__':
  serving = TfServing()
  serving.run()
