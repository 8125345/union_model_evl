import librosa
import numpy as np
import tensorflow as tf
from deepiano.music import audio_io
from deepiano.wav2mid import configs


def mealspec(filename, hparams):
  wav_data = tf.gfile.Open(filename, 'rb').read()
  y = audio_io.wav_data_to_samples_librosa(wav_data, hparams.sample_rate)

  mel = librosa.feature.melspectrogram(
    y,
    hparams.sample_rate,
    hop_length=hparams.spec_hop_length,
    fmin=hparams.spec_fmin,
    n_mels=hparams.spec_n_bins,
    htk=hparams.spec_mel_htk).astype(np.float32)

  mel = mel.T
  return mel


if __name__ == '__main__':
  hparams = configs.CONFIG_MAP['onsets_frames'].hparams
  hparams.use_cudnn = False
  hparams.batch_size = 1

  mealspec('../../data/test/record.wav', hparams)
