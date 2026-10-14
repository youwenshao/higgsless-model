"""Mix and master. GPL-3.0: run only as a separate process (see LICENSE).

Usage: python worker/gpl/mix.py --params-file P --output-dir D --result R
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=1)
    return audio


def _match_lufs(vocal: np.ndarray, rate: int, target: np.ndarray, target_rate: int) -> np.ndarray:
    import pyloudnorm as pyln

    meter = pyln.Meter(rate)
    ref = pyln.Meter(target_rate)
    loud = meter.integrated_loudness(_to_stereo(vocal))
    want = ref.integrated_loudness(_to_stereo(target))
    if not np.isfinite(loud) or not np.isfinite(want):
        return vocal
    return pyln.normalize.loudness(vocal, loud, want)


def mix(params: dict, out: Path) -> dict:
    import matchering as mg
    from pedalboard import Delay, Pedalboard, Reverb

    lead, rate = sf.read(params["lead"], always_2d=True)
    instrumental, i_rate = sf.read(params["instrumental"], always_2d=True)
    if rate != i_rate:
        raise RuntimeError(f"lead is {rate} Hz and the instrumental is {i_rate} Hz; resample first")
    if params.get("reference_vocal"):
        ref, ref_rate = sf.read(params["reference_vocal"], always_2d=True)
        lead = _match_lufs(lead, rate, ref, ref_rate)
    board = Pedalboard([
        Reverb(room_size=float(params.get("room", 0.35)), damping=0.5, wet_level=float(params.get("wet", 0.18)),
               dry_level=1.0, width=1.0),
        Delay(delay_seconds=float(params.get("delay", 0.18)), feedback=0.15, mix=float(params.get("delay_mix", 0.08))),
    ])
    wet = board(lead.T, rate).T
    n = min(len(wet), len(instrumental))
    mixdown = wet[:n] + instrumental[:n]
    peak = np.max(np.abs(mixdown))
    if peak > 0.98:
        mixdown *= 0.98 / peak
    raw = out / "mix_unmastered.wav"
    sf.write(raw, mixdown, rate, subtype="PCM_24")
    final = out / "final_mix.wav"
    if params.get("reference_track"):
        mg.process(str(raw), params["reference_track"], [mg.pcm24(str(final))])
    else:
        final = raw
    lead_out = out / "lead_wet.wav"
    sf.write(lead_out, wet, rate, subtype="PCM_24")
    return {"outputs": {"final_mix": [str(final)], "lead_wet": [str(lead_out)]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = mix(json.loads(Path(args.params_file).read_text(encoding="utf-8")), out)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported through the result file
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
