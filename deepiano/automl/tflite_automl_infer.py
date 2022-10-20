"""Inference for tfrecord.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

from deepiano.wav2mid import configs_tflite as configs
from deepiano.wav2mid import constants
from deepiano.wav2mid import data
from deepiano.wav2mid import infer_util
from deepiano.wav2mid import train_util
from deepiano.music import midi_io
from deepiano.music import sequences_lib
from deepiano.protobuf import music_pb2

from deepiano.music import constants as music_constants

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('master', '',
                           'Name of the TensorFlow runtime to use.')
tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', '../../data/models/test-lite-lstm', 'Path to look for checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
tf.app.flags.DEFINE_string('examples_path', '../../data/maestro/maestro-v1.0.0-tfrecord/test.*',
                           'Path to test examples TFRecord.')
tf.app.flags.DEFINE_string(
    'output_dir', '~/tmp/onsets_frames/infer',
    'Path to store output midi files and summary events.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.3,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'offset_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_integer(
    'max_seconds_per_sequence', None,
    'If set, will truncate sequences to be at most this many seconds long.')

tf.app.flags.DEFINE_boolean(
    'require_frame', False,
    'If set, require an frame prediction for a new note to start.')

tf.app.flags.DEFINE_boolean(
    'require_onset', True,
    'If set, require an onset prediction for a new note to start.')
tf.app.flags.DEFINE_boolean(
    'use_offset', False,
    'If set, use the offset predictions to force notes to end.')

tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def pianoroll_to_note_sequence(frames,
                               frames_per_second,
                               min_duration_ms,
                               velocity=70,
                               instrument=0,
                               program=0,
                               qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                               min_midi_pitch=constants.MIN_MIDI_PITCH,
                               onset_predictions=None,
                               offset_predictions=None,
                               velocity_values=None):
  """Convert frames to a NoteSequence."""
  frame_length_seconds = 1 / frames_per_second

  sequence = music_pb2.NoteSequence()
  sequence.tempos.add().qpm = qpm
  sequence.ticks_per_quarter = music_constants.STANDARD_PPQ

  pitch_start_step = {}
  onset_velocities = velocity * np.ones(
      constants.MAX_MIDI_PITCH, dtype=np.int32)

  # Add silent frame at the end so we can do a final loop and terminate any
  # notes that are still active.
  if frames:
    frames = np.append(frames, [np.zeros(frames[0].shape)], 0)

  if onset_predictions is not None:
    onset_predictions = np.append(onset_predictions,
                                  [np.zeros(onset_predictions[0].shape)], 0)
    # Ensure that any frame with an onset prediction is considered active.
    frames = np.logical_or(frames, onset_predictions)

  # if velocity_values is None:
  #   velocity_values = velocity * np.ones_like(frames, dtype=np.int32)

  if offset_predictions is not None:
    offset_predictions = np.append(offset_predictions,
                                   [np.zeros(offset_predictions[0].shape)], 0)
    # If the frame and offset are both on, then turn it off
    frames[np.where(np.logical_and(frames > 0, offset_predictions > 0))] = 0

  def end_pitch(pitch, end_frame):
    """End an active pitch."""
    start_time = pitch_start_step[pitch] * frame_length_seconds
    end_time = end_frame * frame_length_seconds

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

  def process_active_pitch(pitch, i):
    """Process a pitch being active in a given frame."""
    if pitch not in pitch_start_step:
      if onset_predictions is not None:
        # If onset predictions were supplied, only allow a new note to start
        # if we've predicted an onset.
        if onset_predictions[i, pitch]:
          pitch_start_step[pitch] = i
          onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch] if velocity_values else velocity)
        else:
          # Even though the frame is active, the onset predictor doesn't
          # say there should be an onset, so ignore it.
          pass
      else:
        onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])
        pitch_start_step[pitch] = i
    else:
      if onset_predictions is not None:
        # pitch is already active, but if this is a new onset, we should end
        # the note and start a new one.
        if (onset_predictions[i, pitch] and
            not onset_predictions[i - 1, pitch]):
          end_pitch(pitch, i)
          pitch_start_step[pitch] = i
          onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])

  for i, frame in enumerate(frames):
    for pitch, active in enumerate(frame):
      if active:
        process_active_pitch(pitch, i)
      elif pitch in pitch_start_step:
        end_pitch(pitch, i)

  sequence.total_time = len(frames) * frame_length_seconds
  if sequence.notes:
    assert sequence.total_time >= sequence.notes[-1].end_time

  return sequence


def model_inference(model_fn,
                    model_dir,
                    checkpoint_path,
                    hparams,
                    examples_path,
                    output_dir,
                    summary_writer,
                    master,
                    write_summary_every_step=True):
  """Runs inference for the given examples."""
  tf.logging.info('model_dir=%s', model_dir)
  tf.logging.info('checkpoint_path=%s', checkpoint_path)
  tf.logging.info('examples_path=%s', examples_path)
  tf.logging.info('output_dir=%s', output_dir)

  estimator = train_util.create_estimator(
      model_fn, model_dir, hparams, master=master)

  with tf.Graph().as_default():
    num_dims = constants.MIDI_PITCHES

    dataset = data.provide_batch(
        examples=examples_path,
        preprocess_examples=True,
        params=hparams,
        is_training=False,
        shuffle_examples=False,
        skip_n_initial_records=0)

    # Define some metrics.
    #(metrics_to_updates, metric_note_precision, metric_note_recall,
    # metric_note_f1, metric_note_precision_with_offsets,
    # metric_note_recall_with_offsets, metric_note_f1_with_offsets,
    # metric_note_precision_with_offsets_velocity,
    # metric_note_recall_with_offsets_velocity,
    # metric_note_f1_with_offsets_velocity, metric_frame_labels,
    # metric_frame_predictions) = infer_util.define_metrics(num_dims)

    summary_op = tf.summary.merge_all()

    if write_summary_every_step:
      global_step = tf.train.get_or_create_global_step()
      global_step_increment = global_step.assign_add(1)
    else:
      global_step = tf.constant(
          estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP))
      global_step_increment = global_step

    iterator = dataset.make_initializable_iterator()
    next_record = iterator.get_next()
    with tf.Session() as sess:
      sess.run([
          tf.initializers.global_variables(),
          tf.initializers.local_variables()
      ])

      infer_times = []
      num_frames = []

      sess.run(iterator.initializer)
      while True:
        try:
          record = sess.run(next_record)
        except tf.errors.OutOfRangeError:
          break

        def input_fn(params):
          del params
          return tf.data.Dataset.from_tensors(record)

        start_time = time.time()

        # TODO(fjord): This is a hack that allows us to keep using our existing
        # infer/scoring code with a tf.Estimator model. Ideally, we should
        # move things around so that we can use estimator.evaluate, which will
        # also be more efficient because it won't have to restore the checkpoint
        # for every example.
        prediction_list = list(
            estimator.predict(
                input_fn,
                checkpoint_path=checkpoint_path,
                yield_single_examples=False))
        assert len(prediction_list) == 1

        input_features = record[0]
        input_labels = record[1]

        filename = input_features.sequence_id[0]
        note_sequence = music_pb2.NoteSequence.FromString(
            input_labels.note_sequence[0])
        labels = input_labels.labels[0]
        frame_probs = None  # prediction_list[0]['frame_probs_flat']
        onset_probs = prediction_list[0]['onset_probs_flat']
        velocity_values = None  # prediction_list[0]['velocity_values_flat']
        offset_probs = None  # prediction_list[0]['offset_probs_flat']

        if FLAGS.require_frame:
          frame_predictions = frame_probs > FLAGS.frame_threshold
        else:
          frame_predictions = None
        if FLAGS.require_onset:
          onset_predictions = onset_probs > FLAGS.onset_threshold
        else:
          onset_predictions = None

        if FLAGS.use_offset:
          offset_predictions = offset_probs > FLAGS.offset_threshold
        else:
          offset_predictions = None

        sequence_prediction = pianoroll_to_note_sequence(
            frame_predictions,
            frames_per_second=data.hparams_frames_per_second(hparams),
            min_duration_ms=0,
            min_midi_pitch=constants.MIN_MIDI_PITCH,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=velocity_values)

        # end_time = time.time()
        # infer_time = end_time - start_time
        # infer_times.append(infer_time)
        # num_frames.append(frame_probs.shape[0])
        # tf.logging.info(
        #     'Infer time %f, frames %d, frames/sec %f, running average %f',
        #     infer_time, frame_probs.shape[0], frame_probs.shape[0] / infer_time,
        #     np.sum(num_frames) / np.sum(infer_times))
        #
        # tf.logging.info('Scoring sequence %s', filename)
        #
        def shift_notesequence(ns_time):
          return ns_time + hparams.backward_shift_amount_ms / 1000.

        sequence_label = sequences_lib.adjust_notesequence_times(
            note_sequence, shift_notesequence)[0]
        # infer_util.score_sequence(
        #     sess,
        #     global_step_increment,
        #     metrics_to_updates,
        #     metric_note_precision,
        #     metric_note_recall,
        #     metric_note_f1,
        #     metric_note_precision_with_offsets,
        #     metric_note_recall_with_offsets,
        #     metric_note_f1_with_offsets,
        #     metric_note_precision_with_offsets_velocity,
        #     metric_note_recall_with_offsets_velocity,
        #     metric_note_f1_with_offsets_velocity,
        #     metric_frame_labels,
        #     metric_frame_predictions,
        #     frame_labels=labels,
        #     sequence_prediction=sequence_prediction,
        #     frames_per_second=data.hparams_frames_per_second(hparams),
        #     sequence_label=sequence_label,
        #     sequence_id=filename)

        if write_summary_every_step:
          # Make filenames UNIX-friendly.
          filename_safe = filename.decode('utf-8').replace('/', '_').replace(
              ':', '.')
          output_file = os.path.join(output_dir, filename_safe + '.predicted.midi')
          tf.logging.info('Writing inferred midi file to %s', output_file)
          midi_io.sequence_proto_to_midi_file(sequence_prediction, output_file)

          label_output_file = os.path.join(output_dir,
                                           filename_safe + '.label.midi')
          tf.logging.info('Writing label midi file to %s', label_output_file)
          midi_io.sequence_proto_to_midi_file(sequence_label, label_output_file)

          # Also write a pianoroll showing acoustic model output vs labels.
          # pianoroll_output_file = os.path.join(output_dir,
          #                                      filename_safe + '_pianoroll.png')
          # tf.logging.info('Writing acoustic logit/label file to %s',
          #                 pianoroll_output_file)
          # with tf.gfile.GFile(pianoroll_output_file, mode='w') as f:
          #   scipy.misc.imsave(
          #       f,
          #       infer_util.posterior_pianoroll_image(
          #           frame_probs,
          #           sequence_prediction,
          #           labels,
          #           overlap=True,
          #           frames_per_second=data.hparams_frames_per_second(hparams)))

          #summary = sess.run(summary_op)
          #summary_writer.add_summary(summary, sess.run(global_step))
          #summary_writer.flush()

      if not write_summary_every_step:
        pass
        # Only write the summary variables for the final step.
        #summary = sess.run(summary_op)
        #summary_writer.add_summary(summary, sess.run(global_step))
        #summary_writer.flush()


def main(unused_argv):
  output_dir = os.path.expanduser(FLAGS.output_dir)

  tf.logging.info('---------------------------------')
  tf.logging.info(FLAGS.hparams)
  tf.logging.info('++++++++++++++++++++++++')

  config = configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams
  hparams.use_cudnn = tf.test.is_gpu_available()

  hparams.parse(FLAGS.hparams)

  # Batch size should always be 1 for inference.
  hparams.batch_size = 1

  if FLAGS.max_seconds_per_sequence:
    hparams.truncated_length_secs = FLAGS.max_seconds_per_sequence
  else:
    hparams.truncated_length_secs = 0

  tf.logging.info(hparams)

  tf.gfile.MakeDirs(output_dir)

  summary_writer = tf.summary.FileWriter(logdir=output_dir)

  with tf.Session():
    run_config = '\n\n'.join([
        'model_dir: ' + FLAGS.model_dir,
        'checkpoint_path: ' + str(FLAGS.checkpoint_path),
        'examples_path: ' + FLAGS.examples_path,
        str(hparams),
    ])
    run_config_summary = tf.summary.text(
        'run_config',
        tf.constant(run_config, name='run_config'),
        collections=[])
    summary_writer.add_summary(run_config_summary.eval())

  model_inference(
    model_fn=config.model_fn,
    model_dir=FLAGS.model_dir,
    checkpoint_path=FLAGS.checkpoint_path,
    hparams=hparams,
    examples_path=FLAGS.examples_path,
    output_dir=output_dir,
    summary_writer=summary_writer,
    master=FLAGS.master)


def console_entry_point():
  tf.app.flags.mark_flags_as_required(['model_dir', 'examples_path'])

  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
