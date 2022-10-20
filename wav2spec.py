import numpy as np
import librosa

SR = 16000

def resample(y, src_sr, dst_sr):
    print('RESAMPLING from {} to {}'.format(src_sr, dst_sr))
    if src_sr == dst_sr:
        return y

    if src_sr % dst_sr == 0:
        step = src_sr // dst_sr
        y = y[::step]
        return y
    # Warning, slow
    print('WARNING!!!!!!!!!!!!! SLOW RESAMPLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return librosa.resample(y, src_sr, dst_sr)


def wav2spec(fn, sr=SR, hop_length=512, fmin=30.0, n_mels=229, htk=True, spec_log_amplitude=True):
    y, file_sr = librosa.load(fn, mono=True, sr=None)
    y = resample(y, file_sr, sr)
    y = np.concatenate((y, np.zeros(hop_length * 2, dtype=y.dtype)))

    mel = librosa.feature.melspectrogram(
        y,
        sr,
        hop_length=hop_length,
        fmin=fmin,
        n_mels=n_mels,
        htk=htk).astype(np.float32)

    # Transpose so that the data is in [frame, bins] format.
    spec = mel.T
    if spec_log_amplitude:
        spec = librosa.power_to_db(spec)
    return spec


def split_spec(spec, length=1875, pad=10):
    # 1875: 60 sec.
    slices = []
    pos = 0
    while True:
        left = max(pos - pad, 0)
        right = pos + length + pad + 1
        _slice = spec[left:right]
        end = right >= len(spec) + pad + 1
        slices.append({
            'pos': pos,
            'start_at': pos - left,
            'end_at': len(_slice) if end else len(_slice) - pad,
            'spec': _slice,
        })
        if end:
            break
        pos += length
    return slices
