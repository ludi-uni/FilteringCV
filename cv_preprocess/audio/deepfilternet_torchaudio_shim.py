"""``torchaudio`` 2.9+ 向け互換: DeepFilterNet の ``df.io`` が期待する ``torchaudio.backend.common`` / ``torchaudio.info`` を補う。

DeepFilterNet 0.5.x は古い torchaudio API に依存する。本プロジェクトの cu128 torchaudio では
``AudioMetaData`` と ``info()`` が無いため、import ``df`` の前にこのモジュールを読み込む。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass


def ensure_deepfilternet_torchaudio_shim() -> None:
    if "torchaudio.backend.common" in sys.modules:
        return

    import torchaudio as ta

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int = -1
        num_channels: int = 1
        bits_per_sample: int = 16
        encoding: str = ""

    backend_pkg = types.ModuleType("torchaudio.backend")
    common_mod = types.ModuleType("torchaudio.backend.common")
    common_mod.AudioMetaData = AudioMetaData
    backend_pkg.common = common_mod
    sys.modules["torchaudio.backend"] = backend_pkg
    sys.modules["torchaudio.backend.common"] = common_mod

    if not hasattr(ta, "info"):
        import soundfile as sf

        def _info(path: str, **kwargs):  # type: ignore[no-untyped-def]
            inf = sf.info(path)
            return AudioMetaData(
                sample_rate=int(inf.samplerate),
                num_frames=int(inf.frames),
                num_channels=int(inf.channels),
                bits_per_sample=16,
                encoding="",
            )

        ta.info = _info  # type: ignore[attr-defined]
