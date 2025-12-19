# persian-voice

Generate a small Persian word list, synthesize those words across many TTS providers/models, and serve a static comparison table (rows = words, columns = model+variant).

## Quick start (OpenAI-only baseline)

### 1) Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

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
- `PERSIAN_VOICE_PROVIDERS`: comma-separated provider ids (default `openai`).
- `PERSIAN_VOICE_OPENAI_TIMEOUT`: request timeout seconds (default `20`).
- `PERSIAN_VOICE_OPENAI_MAX_RETRIES`: OpenAI SDK retries (default `2`).

## Adding a provider (pattern)

- Implement `Provider` in `backend/persian_voice/providers/` (see `backend/persian_voice/providers/openai_tts.py`).
- Register it in `backend/persian_voice/providers/registry.py`.
- Re-run `python backend/scripts/render_audio.py --providers all`.
