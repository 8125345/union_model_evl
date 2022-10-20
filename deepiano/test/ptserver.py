import socket, time, os, json, subprocess, base64
from http.server import BaseHTTPRequestHandler, HTTPServer

from deepiano.test.transcribe import init_transcription_session, transcribe

hostName = ""
hostPort = 8089

SESSION = None

WAV_PATH = '/tmp/transcribe.wav'
MID_PATH = '/tmp/transcribe.wav.midi'


class MyServer(BaseHTTPRequestHandler):
    #   GET is for clients geting the predi
    def do_GET(self):
        self.send_response(200)
        self.wfile.write(bytes("<p>You accessed path: %s</p>" % self.path, "utf-8"))

    #   POST is for submitting data.
    def do_POST(self):
        print("incomming http: ", self.path)

        if self.path == '/pt/1.0/wav2mid':
            content_length = int(self.headers['Content-Length'])
            wav_binary = self.rfile.read(content_length)

            with open(WAV_PATH, 'wb') as fp:
                fp.write(wav_binary)

            onset_logits = transcribe(WAV_PATH, SESSION)

            midi_binary = None
            with open(MID_PATH, 'rb') as fp:
                midi_binary = fp.read()

            midi_b64 = base64.b64encode(midi_binary).decode('ascii')
            ret = {
                'midi': midi_b64,
                'preds' : {'onsets': onset_logits.tolist()},
            };

            binary = bytes(json.dumps(ret), 'utf-8')

            os.remove(WAV_PATH)
            os.remove(MID_PATH)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            # self.send_header('Content-Disposition', 'attachment; filename="transcribe.midi"')
            self.send_header('Content-Length', len(binary))
            self.end_headers()
            self.wfile.write(binary)


if __name__ == '__main__':
    SESSION = init_transcription_session()

    myServer = HTTPServer((hostName, hostPort), MyServer)
    print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

    try:
        myServer.serve_forever()
    except KeyboardInterrupt:
        pass

    myServer.server_close()
    print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))
