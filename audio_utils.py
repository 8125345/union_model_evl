import numpy as np
# import soundfile as sf
import librosa
import time
import os
import subprocess

SR = samplerate = 16000


def resampling_file1(file, dst_path):
    """
    acc转wav 16k
    :param file:
    :return:
    """

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'fatal', '-i', f'{file}', '-ac', '1', '-ar', '16000', f"{dst_path}"],
            # cwd='/home/user/Desktop/myProject/',   # current working directory
            # timeout=5,                             # timeout
            check=True  # check for errors
        )
    except Exception:
        print(f"出错跳过，path: {file}")


def resampling_file(file, dst_folder):
    """
    acc转wav 16k
    :param file:
    :return:
    """
    folder, filename = os.path.split(file)
    name, ext = os.path.splitext(filename)
    wav_path = os.path.join(dst_folder, name + "_16k" + ".wav")
    if not os.path.exists(wav_path):
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'fatal', '-i', f'{file}', '-ac', '1', '-ar', '16000', f"{wav_path}"],
                # cwd='/home/user/Desktop/myProject/',   # current working directory
                # timeout=5,                             # timeout
                check=True  # check for errors
            )
        except Exception:
            print(f"出错跳过，path: {file}")


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


def file2arr(fn, sr=SR):
    y, file_sr = librosa.load(fn, mono=True, sr=None)
    y = resample(y, file_sr, sr)
    return y


def wav2spec(point_data, sr=SR, hop_length=512, fmin=30.0, n_mels=229, htk=True, spec_log_amplitude=True):
    y = np.concatenate((point_data, np.zeros(hop_length * 2, dtype=point_data.dtype)))
    mel = librosa.feature.melspectrogram(
        y,
        sr,
        hop_length=hop_length,
        fmin=fmin,
        n_mels=n_mels,
        htk=htk).astype(np.float32)
    # todo 验证此处转置是否会引发错误
    # Transpose so that the data is in [frame, bins] format.
    spec = mel.T
    if spec_log_amplitude:
        spec = librosa.power_to_db(spec)
    return spec


def spec2mel(data):
    spec = librosa.db_to_power(data)
    mel = spec.T

    return mel


def mel2audio(fn, sr=SR, hop_length=512, fmin=30.0, htk=True):
    mel = fn
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        # n_mels=n_mels,  # n_mels的特征librosa后续模块会从shape反向导出，无需填写
        htk=htk)
    return audio


# 反向拼接音乐
def merge_blocks(song, block_type="b", crop=0.1, fmin=30.0):
    """
    拼接处理后的训练数据，用于检验数据处理结果
    :param song:
    :param block_type:
    :param crop:
    :return:
    """
    # b：背景，m：混合，f:前景音
    song_list = list()
    song_len = len(song)
    print("曲子长度", song_len)
    max_cnt = int(song_len * crop)
    print("裁剪后曲子长度", max_cnt)

    for i in range(max_cnt):
        block_dict = song[i]

        block_data = block_dict["data"][block_type]

        mel = spec2mel(block_data)
        audio = mel2audio(mel, fmin=fmin)
        song_list.append(audio)

    return song_list


# 反向拼接音乐
def merge_blocks_predict(song, model, crop=0.1, fmin=30.0, head=False):
    """
    拼接模型推理结果，用于检验单曲推理结果
    :param song:
    :param model:
    :param crop:
    :return:
    """
    # b：背景，m：混合，f:前景音
    song_list = list()
    song_len = len(song)
    print("曲子长度", song_len)
    max_cnt = int(song_len * crop)
    print("裁剪后曲子长度", max_cnt)

    # 遍历一个曲子中的全部block
    for i in range(max_cnt):
        print(i)
        block_dict = song[i]

        input_0 = block_dict["data"]["b"]
        input_1 = block_dict["data"]["m"]
        t1 = time.time()
        input_data = np.stack((input_0, input_1), axis=-1)

        input_data = np.array([input_data])
        output_data = model.predict(input_data)
        if head:
            flag = output_data[1][0][0]
            print(flag)
            output_data = output_data[0]
        output_data = np.squeeze(output_data)
        t2 = time.time()
        # print(f"推理耗时：{t2-t1}")

        mel = spec2mel(output_data)
        t3 = time.time()
        # print(f"反转耗时：{t3- t2}")

        audio = mel2audio(mel, fmin=fmin)
        song_list.append(audio)
        t4 = time.time()
        # print(f"转谱耗时：{t4-t3}")

    rtn = np.concatenate(song_list, axis=0)
    print(f"歌曲的shape: {rtn.shape}")
    return rtn
