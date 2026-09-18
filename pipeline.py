# -*- coding: utf-8 -*-
"""Diarized transcription pipeline on audio.cpp — single ASR pass + word->speaker alignment.

  input audio (wav/mp3) -> 16k mono wav
    -> sortformer_diar_v2 : speaker turns  (--turns-out)
    -> parakeet_tdt      : word timestamps (--words-out), one pass over whole file
    -> align words to turns -> [hh:mm:ss] SPEAKER_xx: text

usage: python pipeline.py <audio> [--backend cpu] [--threads 8] [--device 1]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import wave
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CLI_DIR = PIPELINE_DIR / "bin" / "audio.cpp-vulkan"
DEFAULT_DIAR_MODEL = PIPELINE_DIR / "models" / "sortformer-v2.1-f16-mixed.gguf"
DEFAULT_ASR_MODEL = PIPELINE_DIR / "models" / "parakeet-tdt-0.6b-v3-q8_0.gguf"

SR = 16000


def run_cli(cli: Path, args: list[str]) -> tuple[str, float]:
    cmd = [str(cli), *args]
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cli.parent), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"audiocpp_cli failed ({proc.returncode}):\n{proc.stdout[-3000:]}")
    return proc.stdout, dt


def to_16k_wav(src: Path, dst: Path) -> float:
    """Decode wav/mp3/etc via soundfile -> 16kHz mono s16 wav. Returns duration s."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # mono
    if sr != SR:
        n_out = int(round(len(data) * SR / sr))
        data = np.interp(np.linspace(0, len(data) - 1, n_out), np.arange(len(data)), data).astype("float32")
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    o = wave.open(str(dst), "wb")
    o.setnchannels(1)
    o.setsampwidth(2)
    o.setframerate(SR)
    o.writeframes(pcm)
    o.close()
    return len(data) / SR


def http_post_json(url: str, payload: dict, timeout: float = 900.0) -> dict:
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def diarize(wav16: Path, args, turns_out: Path) -> tuple[list[dict], float]:
    if getattr(args, "server", None):
        t0 = time.perf_counter()
        resp = http_post_json(f"{args.server}/v1/tasks/run",
                              {"model": args.diar_id, "request": {"audio": str(wav16)}})
        turns = resp.get("speaker_turns") or []
        turns_out.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")
        return turns, time.perf_counter() - t0
    # streaming mode = bounded memory on long files (offline builds O(len^2) graph)
    _, dt = run_cli(args.cli, [
        "--task", "diar", "--family", "sortformer_diar_v2", "--model", str(args.diar_model),
        "--backend", args.backend, "--device", args.device, "--threads", str(args.threads),
        "--mode", "streaming",
        "--audio", str(wav16), "--turns-out", str(turns_out),
    ])
    return json.loads(turns_out.read_text(encoding="utf-8")), dt


def transcribe_words(wav16: Path, args, text_out: Path) -> tuple[list[dict], float]:
    if getattr(args, "server", None):
        t0 = time.perf_counter()
        resp = http_post_json(f"{args.server}/v1/tasks/run",
                              {"model": args.asr_id,
                               "request": {"audio": str(wav16),
                                           "audio_chunk_mode": "fixed",
                                           "audio_chunk_duration_sec": args.chunk_sec}})
        text_out.write_text(resp.get("text") or "", encoding="utf-8")
        return resp.get("words") or [], time.perf_counter() - t0
    # fixed chunking keeps encoder graph bounded; word_timestamps come in global coords
    out, dt = run_cli(args.cli, [
        "--task", "asr", "--family", "parakeet_tdt", "--model", str(args.asr_model),
        "--backend", args.backend, "--device", args.device, "--threads", str(args.threads),
        "--audio", str(wav16), "--text-out", str(text_out), "--metrics",
        "--request-option", "audio_chunk_mode=fixed",
        "--request-option", f"audio_chunk_duration_sec={args.chunk_sec:g}",
    ])
    words: list[dict] = []
    for line in out.splitlines():
        if line.startswith("word_timestamps="):
            words += json.loads(line.split("=", 1)[1])
    return words, dt


def fix_word_ends(words: list[dict]) -> list[dict]:
    """TDT inflates the last word's end with following silence; clamp end to next word's start."""
    ws = sorted(words, key=lambda x: x["start_sample"])
    for i, w in enumerate(ws[:-1]):
        nxt = ws[i + 1]["start_sample"]
        if w["end_sample"] > nxt:
            w["end_sample"] = nxt
    return ws


def resolve_turn(word: dict, turns: list[dict]) -> int | None:
    """Best-matching turn index for a word. Word start decides (TDT word ends are inflated
    into following silence, so overlap is biased); gap words go to whichever side of the
    gap midpoint the word center falls on; else max overlap."""
    ws, we = word["start_sample"], word["end_sample"]
    for i, t in enumerate(turns):
        if t["start_sample"] <= ws < t["end_sample"]:
            return i
    prev_i = next_i = None
    for i, t in enumerate(turns):
        if t["end_sample"] <= ws:
            prev_i = i
        elif t["start_sample"] > ws:
            next_i = i
            break
    if prev_i is not None and next_i is not None:
        mid = (turns[prev_i]["end_sample"] + turns[next_i]["start_sample"]) / 2
        return prev_i if (ws + we) / 2 < mid else next_i
    if prev_i is not None or next_i is not None:
        return prev_i if prev_i is not None else next_i
    best_i, best_ov = None, 0
    for i, t in enumerate(turns):
        ov = min(we, t["end_sample"]) - max(ws, t["start_sample"])
        if ov > best_ov:
            best_i, best_ov = i, ov
    return best_i


def speaker_of(word: dict, turns: list[dict]) -> str:
    ti = resolve_turn(word, turns)
    return turns[ti]["speaker_id"] if ti is not None else "SPEAKER_00"


def probe_segments(words: list[dict], turns: list[dict], gap_s: float = 0.4) -> list[dict]:
    """Phrase units: split at pauses > gap_s or whenever the resolved sortformer turn changes
    (turn gaps are likely speaker switches even when the speaker label stays the same)."""
    ws = fix_word_ends(words)
    if not ws:
        return []
    gap = int(gap_s * SR)
    segs: list[dict] = []
    cur = [ws[0]]
    prev_ti = resolve_turn(ws[0], turns)
    for w in ws[1:]:
        ti = resolve_turn(w, turns)
        boundary = (w["start_sample"] - cur[-1]["end_sample"] > gap
                    or (ti is not None and prev_ti is not None and ti != prev_ti))
        if boundary:
            segs.append({"start_sample": cur[0]["start_sample"], "end_sample": cur[-1]["end_sample"], "words": cur})
            cur = [w]
        else:
            cur.append(w)
        if ti is not None:
            prev_ti = ti
    segs.append({"start_sample": cur[0]["start_sample"], "end_sample": cur[-1]["end_sample"], "words": cur})
    return segs


def embed_seg(extractor, samples, s: int, e: int):
    import numpy as np
    st = extractor.create_stream()
    st.accept_waveform(SR, samples[s:e])
    if not extractor.is_ready(st):
        return None
    v = np.asarray(extractor.compute(st), dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n else None


def assign_speakers(samples, words: list[dict], turns: list[dict], extractor,
                    min_seg_s: float = 0.5, min_sim: float = 0.30, min_margin: float = 0.02) -> list[dict]:
    """Re-attribute probe segments via CAM++ embeddings against sortformer speaker centroids."""
    import numpy as np

    segs = probe_segments(words, turns)
    # centroids: weighted mean of embeddings over each speaker's confident turns
    by_spk: dict[str, list] = {}
    for t in turns:
        if t["end_sample"] - t["start_sample"] >= int(0.6 * SR):
            e = embed_seg(extractor, samples, t["start_sample"], t["end_sample"])
            if e is not None:
                w = (t["end_sample"] - t["start_sample"]) * max(t.get("confidence", 1.0), 0.1)
                by_spk.setdefault(t["speaker_id"], []).append((e, w))
    centroids = {}
    for spk, lst in by_spk.items():
        embs = np.stack([e for e, _ in lst])
        wgt = np.array([w for _, w in lst])
        m = np.average(embs, axis=0, weights=wgt)
        m /= np.linalg.norm(m)
        # drop contaminated outliers (e.g. wrong-voice turns sortformer folded in), then recompute
        if len(lst) >= 4:
            sims = embs @ m
            keep = sims >= np.quantile(sims, 0.25)
            if keep.sum() >= 2:
                m = np.average(embs[keep], axis=0, weights=wgt[keep])
                m /= np.linalg.norm(m)
        centroids[spk] = m

    for s in segs:
        dur = s["end_sample"] - s["start_sample"]
        s["turn_spk"] = speaker_of(s["words"][0], turns)
        if dur < int(min_seg_s * SR) or len(centroids) < 2:
            s["speaker"] = s["turn_spk"]
            continue
        e = embed_seg(extractor, samples, s["start_sample"], s["end_sample"])
        if e is None:
            s["speaker"] = s["turn_spk"]
            continue
        sims = sorted(((float(e @ c), spk) for spk, c in centroids.items()), reverse=True)
        s["sims"] = {spk: round(v, 3) for v, spk in sims}
        if sims[0][0] >= min_sim and sims[0][0] - sims[1][0] >= min_margin:
            s["speaker"] = sims[0][1]
        else:
            s["speaker"] = s["turn_spk"]
    return refine_edges(samples, segs, centroids, extractor)


def refine_edges(samples, segs: list[dict], centroids: dict, extractor,
                 edge_words: int = 2, min_word_s: float = 0.5,
                 edge_margin: float = 0.08) -> list[dict]:
    """Word-level pass: around each speaker-label transition, re-embed the boundary words.
    A word may only flip between the two speakers adjacent to that transition, and only
    with a clear margin — single-word embeddings are too noisy for open-set decisions."""
    import numpy as np
    flat = [w for s in segs for w in s["words"]]
    if not flat or len(centroids) < 2:
        for s in segs:
            s.setdefault("speaker", s.get("turn_spk"))
        return segs
    for s in segs:
        for w in s["words"]:
            w["spk"] = s["speaker"]
            w["_seg"] = s

    for i in range(len(flat) - 1):
        a_spk, b_spk = flat[i]["spk"], flat[i + 1]["spk"]
        if a_spk == b_spk or a_spk not in centroids or b_spk not in centroids:
            continue
        pair = np.stack([centroids[a_spk], centroids[b_spk]])
        for j in range(max(0, i + 1 - edge_words), min(len(flat), i + 1 + edge_words)):
            w = flat[j]
            if w["spk"] not in (a_spk, b_spk):
                continue
            if w["end_sample"] - w["start_sample"] < int(min_word_s * SR):
                continue
            # a sentence-final word at its segment edge belongs to that sentence —
            # embeddings on boundary words are too noisy to detach it
            seg = w.get("_seg")
            if seg is not None and w is seg["words"][-1] and w["word"].rstrip().endswith((".", "?", "!", "…")):
                continue
            e = embed_seg(extractor, samples, w["start_sample"], w["end_sample"])
            if e is None:
                continue
            sa, sb = float(pair[0] @ e), float(pair[1] @ e)
            new = a_spk if sa > sb else b_spk
            if new != w["spk"] and abs(sa - sb) >= edge_margin:
                w["spk"] = new
                w["spk_src"] = "emb"

    for w in flat:
        w.pop("_seg", None)
    out: list[dict] = []
    for w in flat:
        if out and out[-1]["speaker"] == w["spk"] and w["start_sample"] - out[-1]["end_sample"] <= int(0.4 * SR):
            out[-1]["words"].append(w)
            out[-1]["end_sample"] = w["end_sample"]
        else:
            out.append({"speaker": w["spk"], "turn_spk": w["spk"],
                        "start_sample": w["start_sample"], "end_sample": w["end_sample"], "words": [w]})
    return out


def build_lines(segs: list[dict], min_gap_s: float = 1.0) -> list[dict]:
    """Group consecutive same-speaker probe segments into utterance lines."""
    min_gap = int(min_gap_s * SR)
    lines: list[dict] = []
    for s in segs:
        words_txt = [w["word"].strip() for w in s["words"] if w["word"].strip()]
        if not words_txt:
            continue
        if (lines and lines[-1]["speaker"] == s["speaker"]
                and s["start_sample"] - lines[-1]["end_sample"] <= min_gap):
            lines[-1]["words"].extend(words_txt)
            lines[-1]["end_sample"] = s["end_sample"]
        else:
            lines.append({"speaker": s["speaker"], "start_sample": s["start_sample"],
                          "end_sample": s["end_sample"], "words": words_txt})
    for l in lines:
        l["text"] = " ".join(l.pop("words"))
    return lines


def fmt_ts(samples: int) -> str:
    s = samples / SR
    return f"{int(s // 3600):02d}:{int(s // 60) % 60:02d}:{s % 60:05.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--device", default="0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--server", default=None, help="audiocpp_server base URL; HTTP mode instead of CLI")
    ap.add_argument("--diar-id", default="sortformer", help="server model id for diarization")
    ap.add_argument("--asr-id", default="parakeet", help="server model id for ASR")
    ap.add_argument("--cli", type=Path, default=DEFAULT_CLI_DIR / "audiocpp_cli.exe")
    ap.add_argument("--diar-model", type=Path, default=DEFAULT_DIAR_MODEL)
    ap.add_argument("--asr-model", type=Path, default=DEFAULT_ASR_MODEL)
    ap.add_argument("--spk-model", type=Path, default=PIPELINE_DIR / "models" / "campplus.onnx")
    ap.add_argument("--no-spk", action="store_true", help="skip speaker-embedding refinement")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    work = PIPELINE_DIR / "output" / (args.tag or args.audio.stem)
    work.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    wav16 = work / "input.16k.wav"
    dur = to_16k_wav(args.audio, wav16)
    print(f"input: {args.audio} -> {wav16} ({dur:.1f}s)")

    turns, t_diar = diarize(wav16, args, work / "turns.json")
    spk = sorted({t["speaker_id"] for t in turns})
    print(f"diar: {len(turns)} turns, {len(spk)} speakers, {t_diar:.1f}s")

    words, t_asr = transcribe_words(wav16, args, work / "text.txt")
    print(f"asr: {len(words)} words, {t_asr:.1f}s")
    (work / "words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")

    t0 = time.perf_counter()
    if args.no_spk or len(spk) < 2:
        segs = probe_segments(words, turns)
        for s in segs:
            s["speaker"] = speaker_of(s["words"][0], turns)
        t_spk = 0.0
    else:
        import soundfile as sf
        import sherpa_onnx
        samples, _ = sf.read(str(wav16), dtype="float32")
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(args.spk_model), num_threads=args.threads))
        segs = assign_speakers(samples, words, turns, extractor)
        t_spk = time.perf_counter() - t0
    print(f"spk-embed: {len(segs)} segments, {t_spk:.1f}s")
    (work / "segments.json").write_text(json.dumps(
        [{"speaker": s["speaker"], "turn_spk": s.get("turn_spk"), "sims": s.get("sims"),
          "start_s": s["start_sample"] / SR, "end_s": s["end_sample"] / SR,
          "text": " ".join(w["word"] for w in s["words"])} for s in segs],
        ensure_ascii=False, indent=1), encoding="utf-8")

    lines = build_lines(segs)
    transcript = "\n".join(f"[{fmt_ts(l['start_sample'])}] {l['speaker']}: {l['text']}" for l in lines)
    print("\n" + transcript)
    (work / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    (work / "transcript.json").write_text(json.dumps(
        [{"speaker": l["speaker"], "start_s": l["start_sample"] / SR,
          "end_s": l["end_sample"] / SR, "text": l["text"]} for l in lines],
        ensure_ascii=False, indent=1), encoding="utf-8")

    total = time.perf_counter() - t_all
    print(f"\ntiming: diar={t_diar:.1f}s asr={t_asr:.1f}s total={total:.1f}s "
          f"(audio {dur:.1f}s, rtf_total={total / dur:.3f})")
    print(f"written: {work / 'transcript.txt'}")


if __name__ == "__main__":
    main()
