# Notes → Video (local-first)

Turn messy study notes into a narrated, captioned educational video — running
**entirely on your own machine**. No cloud LLM, no external TTS, no API keys
required: a local language model structures the notes, a local diffusion model
paints each scene, a local neural voice narrates it, and everything is
assembled into a 1080p MP4.

Built by William West as an exercise in standing up a full generative-AI stack
end to end without renting one.

## How it works

A FastAPI service runs the work in four stages, each of which can be re-run on
its own:

```
notes ─▶ [A] organize ─▶ [B] script + image prompts ─▶ [C] narrate ─▶ [D] assemble ─▶ video.mp4
            Llama 3.1        Llama 3.1                   XTTS v2        MoviePy
                                   │
                                   └─▶ Stable Diffusion paints each scene
```

- **Stage A — organize.** A local LLM (Llama 3.1 8B via [Ollama](https://ollama.com))
  turns unstructured notes into clean, titled sections. Output is forced into
  strict JSON and validated; a deterministic stub keeps the pipeline working
  even if no model is loaded.
- **Stage B — script + imagery.** The LLM writes a narration script per screen
  and an image prompt; **Stable Diffusion 2.1** renders each background locally.
- **Stage C — narrate.** **XTTS v2** generates neural voiceover on-device.
- **Stage D — assemble.** MoviePy composites backgrounds, captions, and audio
  into a 1920×1080 / 30 fps MP4.

## Why local-first

Every model runs on-device, so notes never leave the machine and the whole
thing works offline. The code is written so each model is swappable from
`config.py`, and optional hosted providers (OpenAI, etc.) can be dropped in via
environment variables if you'd rather trade privacy for speed.

## Stack

`Python` · `FastAPI` · `Ollama (Llama 3.1 8B)` · `Stable Diffusion 2.1
(diffusers)` · `XTTS v2 (Coqui TTS)` · `MoviePy`

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# local models (recommended):
ollama pull llama3.1:8b        # LLM
# Stable Diffusion + XTTS download on first run

python app.py                  # http://localhost:8000  (Swagger at /docs)
```

Generate a video:

```bash
curl -X POST http://localhost:8000/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{"notes": "Newton'\''s three laws of motion...", "style": "study"}'
```

Model choices (LLM, diffusion, TTS, GPU on/off) all live in `config.py`.

## Layout

```
app.py              FastAPI entrypoint + routes
pipeline.py         the four-stage orchestration
llm_client.py       local/remote LLM calls, JSON-mode + fallbacks
image_generator.py  Stable Diffusion scene rendering
tts_client.py       XTTS v2 narration
video_assembler.py  MoviePy compositing
config.py           models, resolution, styles — all tunable here
```
