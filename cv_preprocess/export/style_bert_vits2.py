"""Export materialize output to Style-Bert-VITS2 Data/ layout."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cv_preprocess.config.dataset_builder import StyleBertVits2ExportConfig
from cv_preprocess.export.common import place_audio, sanitize_pipe_text
from cv_preprocess.export.protocol import ExportResult, PlaceMode, UtteranceRow


def export_style_bert_vits2(
    utterances: list[UtteranceRow],
    output_dir: Path,
    *,
    config: StyleBertVits2ExportConfig,
    mode: PlaceMode,
    resample: bool,
) -> ExportResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_sr = int(config.sample_rate) if resample else None
    language = sanitize_pipe_text(config.language) or "JP"
    files: list[str] = []

    if config.packaging == "per_speaker":
        by_speaker: dict[str, list[UtteranceRow]] = defaultdict(list)
        for utt in utterances:
            by_speaker[utt.speaker_id].append(utt)
        for speaker, group in sorted(by_speaker.items()):
            model_dir = output_dir / "Data" / speaker
            files.extend(
                _write_model_pack(
                    group,
                    model_dir,
                    language=language,
                    mode=mode,
                    resample_hz=target_sr,
                    root=output_dir,
                )
            )
    else:
        model_dir = output_dir / "Data" / config.model_name
        files.extend(
            _write_model_pack(
                utterances,
                model_dir,
                language=language,
                mode=mode,
                resample_hz=target_sr,
                root=output_dir,
            )
        )

    return ExportResult(
        format="style_bert_vits2",
        output_dir=output_dir,
        utterance_count=len(utterances),
        files=files,
    )


def _write_model_pack(
    utterances: list[UtteranceRow],
    model_dir: Path,
    *,
    language: str,
    mode: PlaceMode,
    resample_hz: int | None,
    root: Path,
) -> list[str]:
    raw_dir = model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    files: list[str] = []
    for utt in utterances:
        filename = f"{utt.clip_id}.wav"
        dst = raw_dir / filename
        place_audio(utt.source_audio, dst, mode, resample_hz=resample_hz)
        files.append(str(dst.relative_to(root)).replace("\\", "/"))
        text = sanitize_pipe_text(utt.text)
        speaker = sanitize_pipe_text(utt.speaker_id)
        lines.append(f"{filename}|{speaker}|{language}|{text}")

    esd_path = model_dir / "esd.list"
    # UTF-8 without BOM
    esd_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    files.append(str(esd_path.relative_to(root)).replace("\\", "/"))
    return files
