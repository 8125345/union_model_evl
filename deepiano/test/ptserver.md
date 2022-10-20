Simple wav2mid server
DO NOT use it in production environment


## Run

```bash
python ptserver.py
```

## Test

Request
```bash
curl --header "Content-Type:application/octet-stream" --data-binary @input.wav http://127.0.0.1:8088/pt/1.0/wav2mid > output
```

Response
```javascript
{
    midi: base64MidiBuffer,
    logits: onsetLogits,
}
```
