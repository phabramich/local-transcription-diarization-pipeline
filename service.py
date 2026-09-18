"""Diarized-transcription HTTP service.

Topology: client -> this FastAPI app -> audiocpp_server (resident models).
The app owns everything that is not GPU inference: audio decode/resample,
CAM++ speaker-embedding refinement, word-to-speaker alignment.

Run:
    audiocpp_server --config server.json --no-ui        # inference engine
    uvicorn service:app --host 0.0.0.0 --port 8100      # this orchestrator

POST /transcribe  (multipart file upload) -> transcript JSON
GET  /health
"""

import json
import os
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile

from pipeline import (SR, assign_speakers, build_lines, fmt_ts, http_post_json,
                      probe_segments, speaker_of, to_16k_wav)

PIPELINE_DIR = Path(__file__).resolve().parent
AUDIOCPP = os.environ.get("AUDIOCPP_SERVER", "http://127.0.0.1:8091")
DIAR_ID = os.environ.get("DIAR_MODEL_ID", "sortformer")
ASR_ID = os.environ.get("ASR_MODEL_ID", "parakeet")
SPK_MODEL = Path(os.environ.get("SPK_MODEL", PIPELINE_DIR / "models" / "campplus.onnx"))
CHUNK_SEC = float(os.environ.get("ASR_CHUNK_SEC", "15"))  # 15s -> ~0.5GB encoder graph, coexists with other GPU tenants
NO_SPK = os.environ.get("NO_SPK", "").lower() in ("1", "true", "yes")

app = FastAPI(title="diar-pipeline", version="1.0")

_lock = threading.Lock()  # one pipeline at a time; engine serializes per model anyway
_extractor = None


def get_extractor():
    global _extractor
    if _extractor is None and not NO_SPK:
        import sherpa_onnx
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(SPK_MODEL), num_threads=8))
    return _extractor


@app.get("/health")
def health():
    try:
        with urllib.request.urlopen(f"{AUDIOCPP}/health", timeout=5) as resp:
            engine = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        engine = {"error": str(e)}
    return {"ok": True, "engine": engine, "speaker_embeddings": not NO_SPK}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), no_spk: bool = False):
    t_all = time.perf_counter()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    outdir = PIPELINE_DIR / "output"
    outdir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=outdir, suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        src = Path(tmp.name)
    wav16 = src.with_suffix(".16k.wav")
    try:
        with _lock:
            dur = to_16k_wav(src, wav16)

            t0 = time.perf_counter()
            turns = http_post_json(
                f"{AUDIOCPP}/v1/tasks/run",
                {"model": DIAR_ID, "request": {"audio": str(wav16)}}).get("speaker_turns") or []
            t_diar = time.perf_counter() - t0

            t0 = time.perf_counter()
            asr = http_post_json(
                f"{AUDIOCPP}/v1/tasks/run",
                {"model": ASR_ID, "request": {"audio": str(wav16),
                                              "audio_chunk_mode": "fixed",
                                              "audio_chunk_duration_sec": CHUNK_SEC}})
            words = asr.get("words") or []
            t_asr = time.perf_counter() - t0

            t0 = time.perf_counter()
            ex = None if no_spk else get_extractor()
            if ex is not None:
                samples, _ = sf.read(str(wav16), dtype="float32")
                segs = assign_speakers(samples, words, turns, ex)
            else:
                segs = probe_segments(words, turns)
                for s in segs:
                    s["speaker"] = speaker_of(s["words"][0], turns)
            t_spk = time.perf_counter() - t0

            lines = build_lines(segs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"pipeline failed: {e}")
    finally:
        src.unlink(missing_ok=True)
        wav16.unlink(missing_ok=True)

    out_lines = [{"speaker": l["speaker"],
                  "start": fmt_ts(l["start_sample"]), "end": fmt_ts(l["end_sample"]),
                  "start_s": round(l["start_sample"] / SR, 2),
                  "end_s": round(l["end_sample"] / SR, 2),
                  "text": l["text"]} for l in lines]
    return {
        "duration_s": round(dur, 1),
        "speakers": sorted({l["speaker"] for l in out_lines}),
        "lines": out_lines,
        "transcript": "\n".join(f"[{l['start']}] {l['speaker']}: {l['text']}" for l in out_lines),
        "timing": {"diar_s": round(t_diar, 2), "asr_s": round(t_asr, 2),
                   "spk_s": round(t_spk, 2), "total_s": round(time.perf_counter() - t_all, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8100")))
