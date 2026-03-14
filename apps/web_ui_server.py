"""
AdsupVoice Web UI Server — EverAI-style FastAPI backend
Serves the custom HTML frontend and exposes REST API for the VieNeu TTS engine.
"""
import os, sys, gc, time, yaml, tempfile, threading
import numpy as np
import soundfile as sf
from typing import Optional, List
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

print("⏳ Đang khởi động AdsupVoice Web UI...")

import torch
from vieneu import VieNeuTTS, FastVieNeuTTS
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool
from sea_g2p import Normalizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f) or {}

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS    = _config.get("codec_configs", {})
TEXT_SETTINGS    = _config.get("text_settings", {})
MAX_CHARS        = TEXT_SETTINGS.get("max_chars_per_chunk", 256)

WEBAPP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "webapp")

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
tts              = None
model_loaded     = False
using_lmdeploy   = False
current_backbone = None
current_codec    = None
loading_lock     = threading.Lock()
_normalizer      = Normalizer()

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="AdsupVoice API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Static files (CSS, JS, images)
if os.path.isdir(WEBAPP_DIR):
    app.mount("/static", StaticFiles(directory=WEBAPP_DIR), name="static")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ─────────────────────────────────────────────
# ROUTES — UI
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(WEBAPP_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h2>Frontend not found. Please check webapp/index.html</h2>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ─────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    return {
        "model_loaded": model_loaded,
        "backbone": current_backbone,
        "codec": current_codec,
        "using_lmdeploy": using_lmdeploy,
    }

@app.get("/api/models")
async def get_models():
    return {
        "backbones": [
            {"id": k, "description": v.get("description", k), "streaming": v.get("supports_streaming", False)}
            for k, v in BACKBONE_CONFIGS.items()
        ],
        "codecs": [
            {"id": k, "description": v.get("description", k)}
            for k, v in CODEC_CONFIGS.items()
        ],
    }

class LoadModelRequest(BaseModel):
    backbone: str = "VieNeu-TTS-0.3B-q4-gguf"
    codec: str    = "NeuCodec ONNX (Fast CPU)"
    device: str   = "Auto"
    use_lmdeploy: bool = False

@app.post("/api/load_model")
async def load_model(req: LoadModelRequest, background_tasks: BackgroundTasks):
    """Trigger model loading in the background; poll /api/status."""
    global tts, model_loaded, using_lmdeploy, current_backbone, current_codec

    if req.backbone not in BACKBONE_CONFIGS:
        raise HTTPException(400, f"Unknown backbone: {req.backbone}")
    if req.codec not in CODEC_CONFIGS:
        raise HTTPException(400, f"Unknown codec: {req.codec}")

    def _load():
        global tts, model_loaded, using_lmdeploy, current_backbone, current_codec
        with loading_lock:
            try:
                model_loaded = False
                if tts is not None:
                    tts = None
                    cleanup_gpu()

                backbone_cfg = BACKBONE_CONFIGS[req.backbone]
                codec_cfg    = CODEC_CONFIGS[req.codec]

                # Device resolution
                dev = req.device.lower()
                if dev == "auto":
                    bb_dev = "cuda" if torch.cuda.is_available() else "cpu"
                elif dev == "cuda":
                    bb_dev = "cuda"
                else:
                    bb_dev = "cpu"

                if "gguf" in backbone_cfg["repo"].lower() and bb_dev == "cuda":
                    bb_dev = "gpu"

                cc_dev = "cpu" if codec_cfg.get("use_preencoded") else bb_dev
                if "onnx" in codec_cfg["repo"].lower():
                    cc_dev = "cpu"

                # Load
                tts = VieNeuTTS(
                    backbone_repo=backbone_cfg["repo"],
                    backbone_device=bb_dev,
                    codec_repo=codec_cfg["repo"],
                    codec_device=cc_dev,
                )
                current_backbone  = req.backbone
                current_codec     = req.codec
                model_loaded      = True
                using_lmdeploy    = False
                print(f"✅ Model loaded: {req.backbone}")
            except Exception as e:
                print(f"❌ Model load failed: {e}")
                model_loaded = False

    background_tasks.add_task(_load)
    return {"status": "loading", "message": f"Loading {req.backbone}…"}

@app.get("/api/voices")
async def get_voices():
    if not model_loaded or tts is None:
        return []
    try:
        voices = tts.list_preset_voices()
        if not voices:
            return []
        if isinstance(voices[0], tuple):
            return [{"id": vid, "name": desc} for desc, vid in voices]
        return [{"id": v, "name": v} for v in voices]
    except Exception as e:
        print(f"Voice list error: {e}")
        return []

class GenerateRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    temperature: float = 1.0
    max_chars_chunk: int = 256
    use_batch: bool = False
    max_batch_size: int = 4

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Generate speech, return path to temp .wav file."""
    if not model_loaded or tts is None:
        raise HTTPException(503, "Model not loaded yet. Please load a model first.")
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text is empty.")

    try:
        # Resolve voice
        ref_codes, ref_text = None, ""
        if req.voice_id:
            voice_data = tts.get_preset_voice(req.voice_id)
            ref_codes  = voice_data["codes"]
            ref_text   = voice_data["text"]
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

        normalized = _normalizer.normalize(req.text.strip())
        chunks     = split_text_into_chunks(normalized, max_chars=req.max_chars_chunk)
        sr         = 24000
        all_wavs   = []

        for chunk in chunks:
            wav = tts.infer(
                chunk,
                ref_codes=ref_codes,
                ref_text=ref_text,
                temperature=req.temperature,
                max_chars=req.max_chars_chunk,
                skip_normalize=True,
            )
            if wav is not None and len(wav) > 0:
                all_wavs.append(wav)

        if not all_wavs:
            raise HTTPException(500, "No audio generated.")

        final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=0.15)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=tempfile.gettempdir())
        sf.write(tmp.name, final_wav, sr)
        tmp.close()

        cleanup_gpu()
        return FileResponse(tmp.name, media_type="audio/wav", filename="output.wav")

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/api/generate_clone")
async def generate_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    temperature: float = Form(1.0),
    max_chars_chunk: int = Form(256),
):
    """Generate speech using a reference audio file for voice cloning."""
    print(f"🎤 [CLONE] Text length: {len(text)}, Ref Text: '{ref_text}', Temp: {temperature}")
    if not model_loaded or tts is None:
        raise HTTPException(503, "Model not loaded yet.")

    # Save uploaded reference audio to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_tmp:
        ref_tmp.write(await ref_audio.read())
        ref_path = ref_tmp.name

    try:
        ref_codes = tts.encode_reference(ref_path)
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

        normalized = _normalizer.normalize(text.strip())
        chunks     = split_text_into_chunks(normalized, max_chars=max_chars_chunk)
        sr         = 24000
        all_wavs   = []

        for chunk in chunks:
            wav = tts.infer(
                chunk,
                ref_codes=ref_codes,
                ref_text=ref_text,
                temperature=temperature,
                max_chars=max_chars_chunk,
                skip_normalize=True,
            )
            if wav is not None and len(wav) > 0:
                all_wavs.append(wav)

        if not all_wavs:
            raise HTTPException(500, "No audio generated.")

        final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=0.15)
        out_tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(out_tmp.name, final_wav, sr)
        out_tmp.close()

        cleanup_gpu()
        return FileResponse(out_tmp.name, media_type="audio/wav", filename="clone_output.wav")

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        os.unlink(ref_path)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    server_name = os.getenv("ADSUP_HOST", "127.0.0.1")
    server_port = int(os.getenv("ADSUP_PORT", "8080"))
    print(f"🌐 AdsupVoice running at http://{server_name}:{server_port}")
    uvicorn.run(app, host=server_name, port=server_port, log_level="warning")

if __name__ == "__main__":
    main()
