import numpy as np

def encode_array(a):
    return {
        'raw': a.tobytes(),
        'dtype': str(a.dtype),
        'shape': a.shape
    }

def decode_array(v):
    return np.frombuffer(v['raw'], dtype=v['dtype']).reshape(v['shape'])


def to_sparse_onsets(onsets, min_onset_value=0.01):
    sparse_idx = np.argwhere(onsets.T > min_onset_value).tolist()

    sparse_onsets = []
    last_idx = (None, None)
    for idx in sparse_idx:
        idx = tuple(idx[::-1])
        if (idx[1] == last_idx[1]) and (idx[0] == last_idx[0] + 1):
            sparse_onsets[-1][1].append(float(onsets[idx]))
        else:
            sparse_onsets.append([idx, [float(onsets[idx])]])

        last_idx = idx

    return {
        'shape': onsets.shape,
        'data': sorted(sparse_onsets, key=lambda x: x[0][0]),
    }
