import io
import json
import numpy as np
import base64
import tempfile
import logging
import uuid
import time
import redis
import pickle
import mido
from flask import Flask
from flask import request
from flask import jsonify
from flask_request_id import RequestID
from deepiano.server.wav2spec import wav2spec as wav2spec_impl
from deepiano.server.wav2spec import split_spec
from deepiano.server import config as server_config
from deepiano.server import utils
from deepiano.music import midi_io

TIMEOUT = 200
TASK_KEY = 'transcribe_tasks_v5'

app = Flask(__name__)
RequestID(app)

redis_pool = redis.ConnectionPool.from_url(server_config.redis_uri)

formatter = logging.Formatter('[%(asctime)s:%(thread)s:%(levelname)s:%(request_id)s] %(message)s')
app.logger.setLevel(logging.INFO)
for handler in app.logger.handlers:
    handler.setFormatter(formatter)
extra = {'request_id': '-'}
app.logger = logging.LoggerAdapter(app.logger, extra)


def seq_to_midi(seq):
    midi = mido.MidiFile()
    midi.ticks_per_beat = 125  # 0.5 / (512 / 16000) * 8
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for note in seq.get('notes', []):
        note_on = mido.Message('note_on', note=note['pitch'], velocity=note['velocity'], time=round(note['startTime'] * 250))
        note_off = mido.Message('note_off', note=note['pitch'], velocity=0, time=round(note['endTime'] * 250))
        track.append(note_on)
        track.append(note_off)

    track.sort(key=lambda msg: msg.time)

    last_time = 0
    for msg in track:
        last_time, msg.time = msg.time, msg.time - last_time

    return midi


def merge_results(results):
    app.logger.info('merging results')
    sec_per_frame = 512 / 16000
    onsets = np.zeros((0, 88), dtype='float32')

    sequence = results[0]['result']['sequence']
    notes = []
    app.logger.info(sequence)

    for chunk in results:
        result = chunk.pop('result')
        start_at = chunk['start_at']
        end_at = chunk['end_at']
        pos = chunk['pos']

        chunk_onsets = utils.decode_array(result['preds']['onsets'])[start_at:end_at]
        onsets = np.concatenate((onsets, chunk_onsets), axis=0)

        seq = result['sequence']

        for note in seq.get('notes', []):
            if 'startTime' not in note:
                continue

            if 'endTime' not in note:
                note['endTime'] = note['startTime'] + 0.5

            if note['startTime'] - start_at * sec_per_frame < 1e-6:
                continue
            note['startTime'] += ((pos - start_at) * sec_per_frame)
            note['endTime'] += ((pos - start_at) * sec_per_frame)
            notes.append(note)

    sequence['notes'] = notes
    if not notes:
        sequence['totalTime'] = 0
    else:
        sequence['totalTime'] = max([n['endTime'] for n in sequence['notes']])

    midi = seq_to_midi(sequence)
    midi_b64 = ''
    with io.BytesIO() as buf:
      midi.save(file=buf)
      midi_b64 = base64.b64encode(buf.getvalue()).decode()

    results = {
        'midi': midi_b64,
        # 'sequence': sequence,
        'preds' : {'onsets': utils.to_sparse_onsets(onsets)},
    }

    app.logger.info('merge results done')
    return results


@app.route('/pt/1.0/wav2mid', methods=['POST'])
def wav2mid():
    request_id = request.environ.get('FLASK_REQUEST_ID')
    app.logger.extra['request_id'] = request_id
    is_test = request.args.get('test')

    task_key = TASK_KEY
    if is_test and (is_test != '0'):
        task_key = '{}_test'.format(TASK_KEY)

    t1 = time.time()
    app.logger.info('wav2mid start')
    with tempfile.NamedTemporaryFile(dir='/tmp', delete=True) as wav_file:
        app.logger.info('writing audio file: %s', wav_file.name)
        wav_file.write(request.data)
        wav_file.flush()
        try:
            r = redis.Redis(connection_pool=redis_pool)
            app.logger.info('convert to spec')
            spec = wav2spec_impl(wav_file.name)
            specs = split_spec(spec)
            app.logger.info('split to %s chunks', len(specs))

            group_id = str(uuid.uuid4())
            for idx, _slice in enumerate(specs):
                key = str(uuid.uuid4())
                task = {
                    'request_id': request_id,
                    'group_id': group_id,
                    'spec_index': idx,
                    'key': key,
                    'pos': _slice['pos'],
                    'start_at': _slice['start_at'],
                    'end_at': _slice['end_at'],
                    'timestamp': time.time(),
                    'spec': utils.encode_array(_slice['spec']),
                }
                app.logger.info('enqueue task: %s', key)
                r.lpush(task_key, pickle.dumps(task))
                r.expire(task_key, 600)
                app.logger.info('waiting for result: %s', key)

            results = []
            for i in specs:
                result = r.brpop(group_id, TIMEOUT)
                if result is None:
                    app.logger.error('Timeout: %s', key)
                    raise Exception('Transcribe Timeout')
                key, value = result
                app.logger.info('result get: %s', key.decode())
                value = pickle.loads(value)
                results.append(value)

            results.sort(key=lambda t: t['spec_index'])

            ret = jsonify(merge_results(results))
            app.logger.info('wav2mid finish, COST %s', time.time() - t1)
            return ret
        except Exception:
            app.logger.error('Error when processing file %s', wav_file.name)
            raise


@app.route('/pt/1.0/wav2spec', methods=['POST'])
def wav2spec():
    with tempfile.NamedTemporaryFile(dir='/tmp', delete=True) as wav_file:
        wav_file.write(request.data)
        wav_file.flush()
        try:
            spec = wav2spec_impl(wav_file.name)
            return jsonify(spec.tolist())
        except Exception:
            app.logger.error('Error when processing file %s', wav_file.name)
            raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086)
