# Transcribe and synthesise speech

Send audio to `/v1/audio/transcriptions` to get text, or text to `/v1/audio/speech` to get audio.

## Before you start

<!-- site: widget checklist id=speech-prereq -->
- [ ] The server runs a build with the `audio` feature, which the default build has. Without it these routes answer 404.
- [ ] The audio clip is a WAV, MP3, FLAC or OGG file.
- [ ] The model is listed in [the models page](../MODELS.md#audio), or named by its repository.
<!-- site: end -->

## Smallest configuration

None. Both routes load the model on the first request that names it.

## Transcribe

<!-- site: widget steps id=speech-transcribe -->
1. Post the clip as a multipart form with `file` and `model`.

   ```sh
   curl -s localhost:11435/v1/audio/transcriptions -F file=@clip.wav -F model=whisper-1
   ```

2. Read the `text` of the answer.
3. Add `language`, `prompt`, `temperature` or `response_format` to steer it. `response_format` is `json`, `text`, `verbose_json`, `srt` or `vtt`.
<!-- site: end -->

`whisper-1` resolves to `openai/whisper-small`.
A larger Whisper, a distil or a faster-whisper checkpoint is named by its repository.
`/v1/audio/translations` takes the same form and answers in English.

## Synthesise

<!-- site: widget steps id=speech-synthesise -->
1. Post `model`, `input` and `voice` as JSON and save the body.

   ```sh
   curl -s localhost:11435/v1/audio/speech -H 'Content-Type: application/json' -d '{
     "model": "tts-1",
     "input": "The lighthouse keeper lit the lamp.",
     "voice": "alloy"
   }' -o out.wav
   ```

2. Set `response_format` to `mp3` for an encoded output. The default is `wav`, and `pcm` is also accepted.
3. Set `stream` to `true` to receive the audio as 1 chunk per sentence.
<!-- site: end -->

| Option | Effect |
|---|---|
| `model` | `tts-1` resolves to a Parler-TTS model. `piper/<voice>`, `pocket-tts` and `kyutai` name the other families. |
| `voice` | An OpenAI preset name. `GET /v1/audio/voices` lists what the resident model offers. |
| `voice_description` | Steers a Parler voice in words, in place of `voice`. |

## Expected result

The first command prints JSON with the transcript:

```json
{"text": "The lighthouse keeper lit the lamp."}
```

The second command writes `out.wav` and prints nothing.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 404 on a route above | The build has no `audio` feature. | Run a build with it. |
| 400 `missing file field` | The form has no `file` part. | Send the clip as `-F file=@clip.wav`. |
| 400 `the audio is empty` | The clip has 0 bytes. | Send a non-empty file. |
| 413 `audio file too large` | The clip exceeds the size cap. | Compress it as MP3, FLAC or OGG, or cut it. |
| 400 `response_format 'x' not supported` | Unknown format. | Use the formats listed above. |
| 400 `voice 'x' is unknown` | `voice` is not a preset. | Pick a name from `GET /v1/audio/voices`, or pass `voice_description`. |

## Next

- [`POST /voice` and `POST /conversation`](../API.md#media): audio in, audio out through the chat model.
- [Audio models](../MODELS.md#audio)
