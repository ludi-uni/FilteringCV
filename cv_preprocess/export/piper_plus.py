"""Export materialize output to piper_plus preprocess layout."""

from __future__ import annotations

from pathlib import Path

from cv_preprocess.config.dataset_builder import PiperPlusExportConfig
from cv_preprocess.export.common import place_audio, sanitize_pipe_text
from cv_preprocess.export.protocol import ExportResult, PlaceMode, UtteranceRow

PIPER_README = """# piper_plus dataset export (FilteringCV)

Layout expected by ``python -m piper_train.preprocess``:

- ``wav/`` — utterance WAV files
- ``metadata.csv`` — no header; ``id|text`` (single speaker) or ``id|speaker|text`` (multi)

Example preprocess:

```bash
python -m piper_train.preprocess \\
  --language ja \\
  --input-dir . \\
  --output-dir ./training \\
  --dataset-format ljspeech \\
  --sample-rate 22050
# add --single-speaker when metadata.csv has only id|text
```
"""


def export_piper_plus(
    utterances: list[UtteranceRow],
    output_dir: Path,
    *,
    config: PiperPlusExportConfig,
    mode: PlaceMode,
    resample: bool,
) -> ExportResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / config.wav_dirname
    wav_dir.mkdir(parents=True, exist_ok=True)

    speakers = {u.speaker_id for u in utterances}
    multi = len(speakers) > 1
    target_sr = int(config.sample_rate) if resample else None

    lines: list[str] = []
    files: list[str] = []
    for utt in utterances:
        stem = utt.clip_id
        dst = wav_dir / f"{stem}.wav"
        place_audio(utt.source_audio, dst, mode, resample_hz=target_sr)
        files.append(str(dst.relative_to(output_dir)).replace("\\", "/"))
        text = sanitize_pipe_text(utt.text)
        if multi:
            lines.append(f"{stem}|{sanitize_pipe_text(utt.speaker_id)}|{text}")
        else:
            lines.append(f"{stem}|{text}")

    csv_path = output_dir / "metadata.csv"
    csv_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    files.append("metadata.csv")

    readme = output_dir / "README.txt"
    readme.write_text(PIPER_README, encoding="utf-8")
    files.append("README.txt")

    return ExportResult(
        format="piper_plus",
        output_dir=output_dir,
        utterance_count=len(utterances),
        files=files,
    )
