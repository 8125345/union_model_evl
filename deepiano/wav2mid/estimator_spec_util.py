# Lint as: python3
"""Utilities for creating EstimatorSpecs for Onsets and Frames models."""

from deepiano.wav2mid import metrics


def get_metrics(features, labels, frame_probs, onset_probs, frame_predictions,
                onset_predictions, offset_predictions, velocity_values,
                hparams):
  """Return metrics values ops."""
  return metrics.define_metrics(
    frame_probs=frame_probs,
    onset_probs=onset_probs,
    frame_predictions=frame_predictions,
    onset_predictions=onset_predictions,
    offset_predictions=offset_predictions,
    velocity_values=velocity_values,
    length=features.length,
    sequence_label=labels.note_sequence,
    frame_labels=labels.labels,
    sequence_id=features.sequence_id,
    hparams=hparams)
