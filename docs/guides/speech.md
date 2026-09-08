# Transcribe and synthesise speech

Needs a build with the `audio` feature, which the default build has.

## Transcription

```sh
curl -s localhost:11435/v1/audio/transcriptions -F file=@clip.wav -F model=whisper-1
```

`whisper-1` resolves to `openai/whisper-small`; the larger Whisper, distil and faster-whisper
checkpoints are named by their repository, see [`../MODELS.md`](../MODELS.md#audio).
`language`, `prompt`, `temperature` and `response_format` are read as the API documents
them; `/v1/audio/translations` answers in English.

## Speech

```sh
curl -s localhost:11435/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "tts-1",
  "input": "The lighthouse keeper lit the lamp.",
  "voice": "alloy"
}' -o out.wav
```

`tts-1` resolves to a Parler-TTS model; `piper/<voice>`, `pocket-tts` and `kyutai` name the
other families. `response_format: "mp3"` encodes the output; a streamed request answers one
chunk per sentence. `GET /v1/audio/voices` lists what the resident model offers, and
`voice_description` steers a Parler voice in words.

## Speech in, speech out

`POST /voice` takes audio and answers audio through the chat model, transcription and synthesis
included; `POST /conversation` keeps the turns. Both are described in
[`../API.md`](../API.md#media).
