# persian-voice

Generate a small Persian word list, synthesize those words across many TTS providers/models, and serve a static comparison table (rows = words, columns = model+variant).

## Quick start (OpenAI-only baseline)

### 1) Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Optional: install local/open-weight model dependencies (large):

```bash
pip install -r backend/requirements-local-models.txt
```

### 1.5) (Optional) Put secrets in `.env`

Create a local `.env` (ignored by git) using `.env.example` as a template.

### 2) Generate (or reset) the word list

Writes `web/public/data/words.json`.

```bash
python backend/scripts/generate_words.py
```

To generate via OpenAI (one-time) instead of the baked-in starter list:

```bash
export OPENAI_API_KEY=...
python backend/scripts/generate_words.py --llm --count 10
```

### 3) Render audio clips (idempotent)

Writes audio to `web/public/audio/...` and updates `web/public/data/models.json` + `web/public/data/clips.json`.

```bash
export OPENAI_API_KEY=...
python backend/scripts/render_audio.py --providers openai
```

To run multiple providers:

```bash
export PERSIAN_VOICE_PROVIDERS=openai,azure_speech,elevenlabs,google_cloud_tts
python backend/scripts/render_audio.py
```

### 4) Run the web UI

```bash
cd web
npm install
npm run dev
```

To build a static export into `web/out` and preview it:

```bash
cd web
npm run build
npm run preview
```

## Environment variables

- `OPENAI_API_KEY`: required for OpenAI synthesis.
- `OPENAI_TTS_MODEL`: default `gpt-4o-mini-tts`.
- `OPENAI_TTS_VOICES`: default `alloy` (comma-separated).
- `PERSIAN_VOICE_PUBLIC_DIR`: where to write `data/` + `audio/` (default `web/public`).
- `PERSIAN_VOICE_PROVIDERS`: comma-separated provider ids (default `all`).
- `PERSIAN_VOICE_OPENAI_TIMEOUT`: request timeout seconds (default `20`).
- `PERSIAN_VOICE_OPENAI_MAX_RETRIES`: OpenAI SDK retries (default `2`).

Azure Speech (provider id `azure_speech`):

- `AZURE_SPEECH_KEY`: subscription key.
- `AZURE_SPEECH_REGION`: Azure region (e.g. `eastus`) or set `AZURE_SPEECH_ENDPOINT`.
- `AZURE_SPEECH_ENDPOINT`: full REST endpoint (optional).
- `AZURE_SPEECH_LANG`: default `fa-IR`.
- `AZURE_SPEECH_VOICES`: default `fa-IR-DilaraNeural,fa-IR-FaridNeural` (comma-separated).
- `AZURE_SPEECH_OUTPUT_FORMAT`: default `audio-24khz-48kbitrate-mono-mp3`.
- `PERSIAN_VOICE_AZURE_TIMEOUT`: request timeout seconds (default `20`).
- `PERSIAN_VOICE_AZURE_MAX_RETRIES`: retries (default `2`).

ElevenLabs (provider id `elevenlabs`):

- `ELEVENLABS_API_KEY` (or `XI_API_KEY`): API key.
- `ELEVENLABS_MODEL_ID`: default `eleven_multilingual_v2`.
- `ELEVENLABS_VOICE_IDS`: comma-separated voice ids (optional; otherwise tries to fetch voices).
- `ELEVENLABS_OUTPUT_FORMAT`: default `mp3_44100_128`.
- `ELEVENLABS_MAX_VOICES`: when auto-fetching, default `2`.
- `PERSIAN_VOICE_ELEVENLABS_TIMEOUT`: request timeout seconds (default `20`).
- `PERSIAN_VOICE_ELEVENLABS_MAX_RETRIES`: retries (default `2`).

Google Cloud Text-to-Speech (provider id `google_cloud_tts`):

- `GOOGLE_CLOUD_TTS_API_KEY` (or `GOOGLE_API_KEY`): API key.
- `GOOGLE_CLOUD_TTS_LANGUAGE_CODE`: default `fa-IR`.
- `GOOGLE_CLOUD_TTS_VOICE_NAMES`: comma-separated voice names (optional; otherwise tries to list voices).
- `GOOGLE_CLOUD_TTS_MAX_VOICES`: when auto-fetching, default `2`.
- `GOOGLE_CLOUD_TTS_SPEAKING_RATE`, `GOOGLE_CLOUD_TTS_PITCH`, `GOOGLE_CLOUD_TTS_SAMPLE_RATE_HZ`: optional audio config knobs.
- `PERSIAN_VOICE_GOOGLE_CLOUD_TIMEOUT`: request timeout seconds (default `20`).
- `PERSIAN_VOICE_GOOGLE_CLOUD_MAX_RETRIES`: retries (default `2`).

Local open-weight models (disabled by default):

- `HF_TOKEN` (or `HUGGINGFACE_API_TOKEN`): optional Hugging Face token for faster downloads / gated models.

Meta MMS TTS (provider id `mms`):

- `MMS_TTS_ENABLED`: set to `1` to enable (downloads the `facebook/mms-tts-fas` weights on first run).
- `MMS_TTS_DEVICE`: optional torch device (e.g. `cpu`, `cuda`, `mps`).

OuteTTS (provider id `outetts`):

- `OUTETTS_ENABLED`: set to `1` to enable (downloads the `OuteAI/Llama-OuteTTS-1.0-1B` weights on first run).
- `OUTETTS_SPEAKERS`: comma-separated default speaker ids (default `EN-FEMALE-1-NEUTRAL`).
- `OUTETTS_BACKEND`: `hf` (default) or `llamacpp`.
- `OUTETTS_LLAMACPP_QUANTIZATION`: e.g. `FP16` (only used when `OUTETTS_BACKEND=llamacpp`).
- `OUTETTS_TEMPERATURE`: default `0.4`.

Amazon Polly (provider id `aws_polly`):

- Requires AWS CLI (`aws`) + configured credentials.
- `AWS_POLLY_ENABLED`: set to `1` to enable (disabled by default to avoid accidental API usage).
- `AWS_POLLY_VOICE_IDS`: comma-separated voice ids (default `Joanna`).
- `AWS_POLLY_ENGINE`: optional `standard|neural|generative`.
- `AWS_POLLY_TEXT_TYPE`: optional `text|ssml`.
- Uses `AWS_POLLY_REGION` or `AWS_REGION` / `AWS_DEFAULT_REGION` if set (otherwise defaults to `us-east-1`).

IBM Watson TTS (provider id `ibm_watson`):

- `IBM_WATSON_TTS_API_KEY`: API key.
- `IBM_WATSON_TTS_URL`: service URL (base; e.g. `https://.../instances/<id>`).
- `IBM_WATSON_TTS_VOICES`: comma-separated voice ids (default `en-US_AllisonV3Voice`).
- `IBM_WATSON_TTS_ACCEPT`: `audio/mp3` (default) or `audio/wav`.
- `PERSIAN_VOICE_IBM_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_IBM_MAX_RETRIES`: retries (default `2`).

Resemble AI (provider id `resemble`):

- `RESEMBLE_API_KEY`: API key.
- `RESEMBLE_VOICE_UUIDS`: comma-separated `voice_uuid` values.
- `RESEMBLE_OUTPUT_FORMAT`: `wav` (default) or `mp3`.
- `RESEMBLE_SAMPLE_RATE`: default `48000`.
- `PERSIAN_VOICE_RESEMBLE_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_RESEMBLE_MAX_RETRIES`: retries (default `2`).

Narakeet (provider id `narakeet`):

- `NARAKEET_API_KEY`: API key.
- `NARAKEET_VOICES`: comma-separated voice names (optional; uses provider default if unset).
- `NARAKEET_VOICE_SPEED`, `NARAKEET_VOICE_VOLUME`: optional query params.
- `PERSIAN_VOICE_NARAKEET_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_NARAKEET_MAX_RETRIES`: retries (default `2`).

Speechify (provider id `speechify`):

- `SPEECHIFY_API_KEY`: API key (sent as `Authorization: Bearer ...`).
- `SPEECHIFY_VOICE_IDS`: comma-separated voice ids (default `cliff`).
- `SPEECHIFY_BASE_URL`: default `https://api.sws.speechify.com`.
- `PERSIAN_VOICE_SPEECHIFY_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_SPEECHIFY_MAX_RETRIES`: retries (default `2`).

LOVO (provider id `lovo`):

- `LOVO_API_KEY`: API key (sent as `X-API-KEY`).
- `LOVO_SPEAKER_IDS`: comma-separated speaker ids (optional; otherwise tries to list speakers).
- `LOVO_LOCALE_FILTER`: when auto-listing speakers, default `fa-IR`.
- `LOVO_SPEAKER_STYLE_ID`: optional speaker style id.
- `LOVO_SPEED`: default `1.0`.
- `LOVO_BASE_URL`: default `https://api.genny.lovo.ai`.
- `LOVO_MAX_SPEAKERS`: when auto-fetching, default `2`.
- `PERSIAN_VOICE_LOVO_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_LOVO_MAX_RETRIES`: retries (default `2`).

SpeechGen (provider id `speechgen`):

- `SPEECHGEN_TOKEN`: API token from your SpeechGen profile.
- `SPEECHGEN_EMAIL`: your SpeechGen account email.
- `SPEECHGEN_VOICES`: comma-separated voice names (optional; otherwise auto-picks from the public `Persian` voice list).
- `SPEECHGEN_FORMAT`: `mp3` (default) or `wav`.
- `SPEECHGEN_BASE_URL`: default `https://speechgen.io`.
- `SPEECHGEN_SPEED`, `SPEECHGEN_PITCH`, `SPEECHGEN_EMOTION`: optional knobs.
- `PERSIAN_VOICE_SPEECHGEN_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_SPEECHGEN_MAX_RETRIES`: retries (default `2`).
- `PERSIAN_VOICE_SPEECHGEN_TOTAL_TIMEOUT`: poll timeout seconds (default `60`).
- `PERSIAN_VOICE_SPEECHGEN_POLL_INTERVAL`: poll interval seconds (default `1`).

AiVOOV (provider id `aivoov`):

- `AIVOOV_API_KEY`: API key (sent as `X-API-KEY`).
- `AIVOOV_VOICE_IDS`: comma-separated voice ids (optional; otherwise tries to list voices).
- `AIVOOV_LANGUAGE_CODE`: used when listing voices, default `fa-IR`.
- `AIVOOV_BASE_URL`: default `https://aivoov.com/api/v8`.
- `AIVOOV_MAX_VOICES`: when auto-fetching, default `2`.
- `AIVOOV_PITCH`, `AIVOOV_SPEAKING_RATE`, `AIVOOV_VOLUME`: optional SSML-style knobs (`default` keeps provider defaults).
- `PERSIAN_VOICE_AIVOOV_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_AIVOOV_MAX_RETRIES`: retries (default `2`).

WellSaid (provider id `wellsaid`):

- `WELLSAID_API_KEY`: API key (sent as `X-API-KEY`).
- `WELLSAID_SPEAKER_IDS`: comma-separated integer speaker ids.
- `WELLSAID_MODEL`: default `caruso`.
- `WELLSAID_SAMPLE_RATE`: default `44100`.
- `PERSIAN_VOICE_WELLSAID_TIMEOUT`: request timeout seconds (default `30`).
- `PERSIAN_VOICE_WELLSAID_MAX_RETRIES`: retries (default `2`).

CAMB.AI (provider id `cambai`):

- `CAMB_API_KEY`: API key.
- `CAMB_VOICE_IDS`: comma-separated integer voice ids.
- `CAMB_LANGUAGE_ID`: optional override (otherwise inferred from voice list when possible; defaults to `1`).
- `CAMB_MAX_VOICES`: when auto-fetching, default `2`.
- `PERSIAN_VOICE_CAMB_TIMEOUT`: per-request timeout seconds (default `30`).
- `PERSIAN_VOICE_CAMB_TOTAL_TIMEOUT`: total poll timeout seconds (default `120`).
- `PERSIAN_VOICE_CAMB_POLL_INTERVAL`: poll interval seconds (default `1`).
- Note: CAMB currently saves raw bytes as `flac`.

## Adding a provider (pattern)

- Implement `Provider` in `backend/persian_voice/providers/` (see `backend/persian_voice/providers/openai_tts.py`).
- Register it in `backend/persian_voice/providers/registry.py`.
- Re-run `python backend/scripts/render_audio.py --providers all`.

## Deployment

The web UI is deployed to Kubernetes at `tts-shootout.farsi.school`.

### Syncing audio files to S3

Audio files are stored in S3 and served separately from the static site. After generating new audio clips locally, sync them:

```bash
AWS_PROFILE=new aws s3 sync web/public/audio/ s3://persian-tts-shootout-audio/audio/
```

### Deploying code changes

1. Push changes to `master` (changes in `web/` trigger the CI)
2. GitHub Actions builds and pushes the Docker image to ECR
3. Restart the deployment to pull the new image:

```bash
kubectl rollout restart deployment/persian-tts-shootout -n languagetool --context=k6.srvpl.de
```

### GitHub Secrets Required

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: For pushing Docker images to ECR
- `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN`: For ratings database (baked into the static build)
