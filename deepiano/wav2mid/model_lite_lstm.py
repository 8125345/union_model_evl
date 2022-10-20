"""Onset-focused model for piano transcription."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from deepiano.common.sequence_example_lib import flatten_maybe_padded_sequences
from deepiano.common import tf_utils
from deepiano.wav2mid import constants

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.lite.experimental.examples.lstm.rnn import bidirectional_dynamic_rnn


def conv_net(inputs, hparams):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
    [slim.conv2d, slim.fully_connected],
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
      factor=2.0, mode='FAN_AVG', uniform=True)):

    net = inputs
    i = 0
    for (conv_temporal_size, conv_freq_size,
         num_filters, freq_pool_size, dropout_amt) in zip(
      hparams.temporal_sizes, hparams.freq_sizes, hparams.num_filters,
      hparams.pool_sizes, hparams.dropout_keep_amts):
      net = slim.conv2d(
        net,
        num_filters, [conv_temporal_size, conv_freq_size],
        scope='conv' + str(i),
        normalizer_fn=slim.batch_norm)
      if freq_pool_size > 1:
        net = slim.max_pool2d(
          net, [1, freq_pool_size],
          stride=[1, freq_pool_size],
          scope='pool' + str(i))
      if dropout_amt < 1:
        net = slim.dropout(net, dropout_amt, scope='dropout' + str(i))
      i += 1

    # Flatten while preserving batch and time dimensions.
    dims = tf.shape(net)
    net = tf.reshape(
      net, (dims[0], dims[1], net.shape[2].value * net.shape[3].value),
      'flatten_end')

    net = slim.fully_connected(net, hparams.fc_size, scope='fc_end')
    net = slim.dropout(net, hparams.fc_dropout_keep_amt, scope='dropout_end')

    return net


def lstm_layer(inputs,
               batch_size,
               num_units,
               lengths=None,
               stack_size=1,
               use_cudnn=False,
               rnn_dropout_drop_amt=0,
               is_training=True,
               bidirectional=True):
  """Create a LSTM layer using the specified backend."""
  assert rnn_dropout_drop_amt == 0
  cells_fw = tf.lite.experimental.nn.TFLiteLSTMCell(num_units)
  cells_bw = tf.lite.experimental.nn.TFLiteLSTMCell(num_units)
  with tf.variable_scope('cudnn_lstm'):
    if bidirectional:
        lstm_inputs = tf.transpose(inputs, [1, 0, 2])
        outputs, unused_state_f = bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths,
                parallel_iterations=1,
                time_major=True)
        fw_outputs, bw_outputs = outputs
        output_s = tf.concat([fw_outputs, bw_outputs], 2)
        output = tf.transpose(output_s, [1, 0, 2])
    else:
        lstm_inputs = tf.transpose(inputs, [1, 0, 2])
        with tf.variable_scope("unidirectional_rnn"):
            with tf.variable_scope("fw") as fw_scope:
                output_fw, unused_output_state_fw = tf.lite.experimental.nn.dynamic_rnn(
                         cell=cells_fw,
                         inputs=lstm_inputs,
                         sequence_length=lengths,
                         dtype=tf.float32,
                         #parallel_iterations=1,
                         time_major=True,
                         scope=fw_scope)
                output = tf.transpose(output_fw, [1, 0, 2])
    return output


def acoustic_model(inputs, hparams, lstm_units, lengths, is_training=True):
  """Acoustic model that handles all specs for a sequence in one window."""
  conv_output = conv_net(inputs, hparams)

  if lstm_units:
    return lstm_layer(
      conv_output,
      hparams.batch_size,
      lstm_units,
      lengths=lengths if hparams.use_lengths else None,
      stack_size=hparams.acoustic_rnn_stack_size,
      use_cudnn=hparams.use_cudnn,
      is_training=is_training,
      bidirectional=hparams.bidirectional)

  else:
    return conv_output


def model_fn(features, labels, mode, params, config):
  """Builds the acoustic model."""
  del config
  hparams = params

  if isinstance(features, dict):
    length = features.get('length')
    spec = features.get('spec')
    params.add_hparam('batch_size', 1)
  else:
    length = features.length
    spec = features.spec

  is_training = mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL

  if is_training:
    onset_labels = labels.onsets
    offset_labels = labels.offsets
    velocity_labels = labels.velocities
    frame_labels = labels.labels
    frame_label_weights = labels.label_weights

  if hparams.stop_activation_gradient and not hparams.activation_loss:
    raise ValueError(
      'If stop_activation_gradient is true, activation_loss must be true.')

  losses = {}
  accuracies = {}
  eval_metric_ops = {}
  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
    with tf.variable_scope('onsets'):
      onset_outputs = acoustic_model(
        spec,
        hparams,
        lstm_units=hparams.onset_lstm_units,
        lengths=length,
        is_training=is_training)
      onset_probs = slim.fully_connected(
        onset_outputs,
        constants.MIDI_PITCHES,
        activation_fn=tf.sigmoid,
        scope='onset_probs')

      # onset_probs_flat is used during inference.
      onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, length, hparams.tflite)
      if is_training:
        onset_labels_flat = flatten_maybe_padded_sequences(onset_labels, length, hparams.tflite)
        onset_losses = tf_utils.log_loss(onset_labels_flat, onset_probs_flat)
        tf.losses.add_loss(tf.reduce_mean(onset_losses))
        losses['onset'] = onset_losses

        accuracy = tf.metrics.accuracy(labels=onset_labels_flat > 0.3,
                                       predictions=onset_probs_flat > 0.3,
                                       name='onset_accuracy')
        # onset_accuracy = (
        #   tf.reduce_sum(
        #     tf.to_float(tf.equal(onset_labels_flat > 0.3, onset_probs_flat > 0.3))) /
        #   tf.cast(tf.size(onset_labels_flat), tf.float32))
        accuracies['onset'] = accuracy[1]
        eval_metric_ops = {'accuracy': accuracy}

    # with tf.variable_scope('frame'):
    #   if not hparams.share_conv_features:
    #     # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
    #     activation_outputs = acoustic_model(
    #         spec,
    #         hparams,
    #         lstm_units=hparams.frame_lstm_units,
    #         lengths=length,
    #         is_training=is_training)
    #     activation_probs = slim.fully_connected(
    #         activation_outputs,
    #         constants.MIDI_PITCHES,
    #         activation_fn=tf.sigmoid,
    #         scope='activation_probs')
    #   else:
    #     activation_probs = slim.fully_connected(
    #         onset_outputs,
    #         constants.MIDI_PITCHES,
    #         activation_fn=tf.sigmoid,
    #         scope='activation_probs')
    #
    #   probs = []
    #   if hparams.stop_onset_gradient:
    #     probs.append(tf.stop_gradient(onset_probs))
    #   else:
    #     probs.append(onset_probs)
    #
    #   if hparams.stop_activation_gradient:
    #     probs.append(tf.stop_gradient(activation_probs))
    #   else:
    #     probs.append(activation_probs)
    #
    #   combined_probs = tf.concat(probs, 2)
    #
    #   if hparams.combined_lstm_units > 0:
    #     outputs = lstm_layer(
    #         combined_probs,
    #         hparams.batch_size,
    #         hparams.combined_lstm_units,
    #         lengths=length if hparams.use_lengths else None,
    #         stack_size=hparams.combined_rnn_stack_size,
    #         use_cudnn=hparams.use_cudnn,
    #         is_training=is_training,
    #         bidirectional=hparams.bidirectional)
    #   else:
    #     outputs = combined_probs
    #
    #   frame_probs = slim.fully_connected(
    #       outputs,
    #       constants.MIDI_PITCHES,
    #       activation_fn=tf.sigmoid,
    #       scope='frame_probs')
    #
    # frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, length)
    #
    # if is_training:
    #   frame_labels_flat = flatten_maybe_padded_sequences(frame_labels, length)
    #   frame_label_weights_flat = flatten_maybe_padded_sequences(
    #       frame_label_weights, length)
    #   if hparams.weight_frame_and_activation_loss:
    #     frame_loss_weights = frame_label_weights_flat
    #   else:
    #     frame_loss_weights = None
    #   frame_losses = tf_utils.log_loss(
    #       frame_labels_flat, frame_probs_flat, weights=frame_loss_weights)
    #   tf.losses.add_loss(tf.reduce_mean(frame_losses))
    #   losses['frame'] = frame_losses
    #
    #   if hparams.activation_loss:
    #     if hparams.weight_frame_and_activation_loss:
    #       activation_loss_weights = frame_label_weights
    #     else:
    #       activation_loss_weights = None
    #     activation_losses = tf_utils.log_loss(
    #         frame_labels_flat,
    #         flatten_maybe_padded_sequences(activation_probs, length),
    #         weights=activation_loss_weights)
    #     tf.losses.add_loss(tf.reduce_mean(activation_losses))
    #     losses['activation'] = activation_losses

  predictions = {
    'onset_probs_flat': onset_probs_flat,
    # 'frame_probs': frame_probs_flat,
  }

  train_op = None
  loss = None
  if is_training:
    # Creates a pianoroll labels in red and probs in green [minibatch, 88]
    images = {}
    onset_pianorolls = tf.concat([
      onset_labels[:, :, :, tf.newaxis], onset_probs[:, :, :, tf.newaxis],
      tf.zeros(tf.shape(onset_labels))[:, :, :, tf.newaxis]
    ],
      axis=3)
    images['OnsetPianorolls'] = onset_pianorolls

    # activation_pianorolls = tf.concat([
    #     frame_labels[:, :, :, tf.newaxis], frame_probs[:, :, :, tf.newaxis],
    #     tf.zeros(tf.shape(frame_labels))[:, :, :, tf.newaxis]
    # ],
    #                                   axis=3)
    # images['ActivationPianorolls'] = activation_pianorolls

    for name, image in images.items():
      tf.summary.image(name, image)

    loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)
    for label, loss_collection in losses.items():
      loss_label = 'losses/' + label
      tf.summary.scalar(loss_label, tf.reduce_mean(loss_collection))

    for label, accuracy_collection in accuracies.items():
      accuracy_label = 'accuracy/' + label
      tf.summary.scalar(accuracy_label, accuracy_collection)

    train_op = tf.contrib.layers.optimize_loss(
      name='training',
      loss=loss,
      global_step=tf.train.get_or_create_global_step(),
      learning_rate=hparams.learning_rate,
      learning_rate_decay_fn=functools.partial(
        tf.train.exponential_decay,
        decay_steps=hparams.decay_steps,
        decay_rate=hparams.decay_rate,
        staircase=True),
      clip_gradients=hparams.clip_norm,
      optimizer='Adam')

  return tf.estimator.EstimatorSpec(
    mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)


def get_default_hparams():
  """Returns the default hyperparameters.

  Returns:
    A tf.contrib.training.HParams object representing the default
    hyperparameters for the model.
  """
  return tf.contrib.training.HParams(
    batch_size=8,
    learning_rate=0.0006,
    decay_steps=10000,
    decay_rate=0.98,
    clip_norm=3.0,
    transform_audio=True,
    onset_lstm_units=256,
    frame_lstm_units=0,
    acoustic_rnn_stack_size=1,
    combined_rnn_stack_size=1,
    activation_loss=False,
    stop_activation_gradient=False,
    stop_onset_gradient=True,
    stop_offset_gradient=True,
    weight_frame_and_activation_loss=False,
    combined_lstm_units=256,
    share_conv_features=False,
    temporal_sizes=[3, 3, 3],
    freq_sizes=[3, 3, 3],
    #num_filters=[24, 24, 48],
    num_filters=[48, 48, 96],
    pool_sizes=[1, 2, 2],
    dropout_keep_amts=[1.0, 0.25, 0.25],
    fc_size=768,
    fc_dropout_keep_amt=0.5,
    use_lengths=False,
    rnn_dropout_drop_amt=0.0,
    bidirectional=True,
    tflite=True,
  )
