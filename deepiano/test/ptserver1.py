import base64
import tempfile
from flask import Flask
from flask import request
from flask import jsonify

from deepiano.test.transcribe import init_transcription_session, transcribe

app = Flask(__name__)


class Session:
  def __init__(self):
    self.session = init_transcription_session()
    self.busy = False

  def transcribe(self, file):
    self.busy = True
    try:
      result = transcribe(file, self.session)
    finally:
      self.busy = False
    return result


class TranscriberPool:
  def __init__(self, init_workers, max_workers=10):
    self.sessions = []
    self.max_workers = max_workers
    for i in range(init_workers):
      print('init worker: {}'.format(i))
      self.sessions.append(Session())

  def transcribe(self, file):
    for session in self.sessions:
      if not session.busy:
        return session.transcribe(file)

    if len(self.sessions) < self.max_workers:
      session = Session()
      self.sessions.append(session)
      print('add worker')
      return session.transcribe(file)
    raise Exception('Max workers reached')


pool = TranscriberPool(5)


@app.route('/pt/1.0/wav2mid', methods=['POST'])
def wav2mid():
  with tempfile.NamedTemporaryFile(dir='/tmp', delete=False) as wav_file:
    wav_file.write(request.data)
    wav_file.flush()
    try:
      onset_logits = pool.transcribe(wav_file.name)
    except Exception:
      print('Error when processing file', wav_file.name)
      raise
    midi_path = wav_file.name + '.midi'

    with open(midi_path, 'rb') as fp:
      midi_binary = fp.read()
      midi_b64 = base64.b64encode(midi_binary).decode('ascii')
      ret = {
        'midi': midi_b64,
        'preds': {'onsets': onset_logits.tolist()},
      }

      return jsonify(ret)


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=9999, debug=True)
