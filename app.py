import os
import re
import json
import hashlib
import tempfile
import base64
import wave
from pathlib import Path

import yaml
import requests
from flask import Flask, request, jsonify, send_file

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} references with the corresponding environment variable."""
    def _replacer(match):
        return os.environ.get(match.group(1), "")
    return re.sub(r"\$\{(\w+)\}", _replacer, str(value))


def _resolve_config(obj):
    """Recursively resolve environment variable references in config."""
    if isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_config(item) for item in obj]
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    return obj


CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = _resolve_config(yaml.safe_load(f))

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

CACHE_DIR = Path(__file__).parent / CFG["cache"]["dir"]
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(preset_id: str, text: str) -> Path:
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return CACHE_DIR / preset_id / f"{text_hash}.wav"


def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(buf.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        with open(buf.name, "rb") as f:
            return f.read()
    finally:
        buf.close()
        os.unlink(buf.name)


def _call_mimo_api(preset_id: str, text: str) -> bytes:
    """Call MiMo TTS API and return WAV bytes."""
    preset = CFG["voice_presets"][preset_id]

    payload = {
        "model": CFG["mimo"]["model"],
        "messages": [
            {"role": "user", "content": preset["description"]},
            {"role": "assistant", "content": preset["prefix"] + text},
        ],
        "audio": {
            "format": "pcm16",
            "voice": preset["voice"],
        },
        "stream": True,
    }

    headers = {
        "api-key": CFG["mimo"]["api_key"],
        "Content-Type": "application/json",
    }

    resp = requests.post(
        CFG["mimo"]["api_url"],
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    pcm_chunks: list[bytes] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            audio = delta.get("audio") or {}
            audio_data = audio.get("data")
            if audio_data:
                pcm_chunks.append(base64.b64decode(audio_data))
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            continue

    if not pcm_chunks:
        raise RuntimeError("No audio data received from MiMo API")

    pcm_bytes = b"".join(pcm_chunks)
    return _pcm16_to_wav(pcm_bytes)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/presets")
def presets():
    return jsonify({
        pid: {"voice": p["voice"], "description": p["description"]}
        for pid, p in CFG["voice_presets"].items()
    })


@app.route("/tts")
def tts():
    preset_id = request.args.get("id", "").strip()
    text = request.args.get("text", "").strip()

    if not preset_id:
        return jsonify({"error": "missing parameter: id"}), 400
    if not text:
        return jsonify({"error": "missing parameter: text"}), 400
    if preset_id not in CFG["voice_presets"]:
        available = list(CFG["voice_presets"].keys())
        return jsonify({"error": f"unknown preset id: {preset_id}", "available": available}), 400

    cache_file = _cache_path(preset_id, text)

    # Serve from cache if available
    if cache_file.exists():
        return send_file(cache_file, mimetype="audio/wav")

    # Call MiMo API
    try:
        wav_bytes = _call_mimo_api(preset_id, text)
    except requests.HTTPError as e:
        return jsonify({"error": f"MiMo API error: {e.response.status_code}", "detail": e.response.text}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save to cache (write to tmp then rename for atomicity)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_file.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(wav_bytes)
        os.replace(tmp_path, str(cache_file))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return send_file(cache_file, mimetype="audio/wav")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = CFG["server"]["host"]
    port = CFG["server"]["port"]
    print(f"Starting MiMo TTS API on {host}:{port}")
    app.run(host=host, port=port, debug=True)
