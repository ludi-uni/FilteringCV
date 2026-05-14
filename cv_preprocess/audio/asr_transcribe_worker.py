# SPDX-License-Identifier: Apache-2.0
"""NeMo ASR 常駐ワーカー（``NFA_PYTHON`` / NeMo venv で起動）。stdin 1 行 JSON 1 応答。

* ``{"op":"init",...}`` -> ``{"ok":true}`` / ``{"ok":false,"detail":"..."}``
* ``{"op":"transcribe","paths":["/abs/a.wav",...]}`` -> ``{"ok":true,"texts":[...]}`` （失敗時 ``ok:false``）
* ``{"op":"shutdown"}`` -> 終了

stdout に NeMo のログが混ざる場合があるため、親は先頭が ``{`` の JSON 行を拾う。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model


def _reply(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _hyp_to_text(h: Any) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h.strip()
    t = getattr(h, "text", None)
    if isinstance(t, str):
        return t.strip()
    return str(h).strip()


def _load_model(init: dict[str, Any]) -> tuple[Any, torch.device]:
    device = torch.device(init.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_setup = OmegaConf.create(
        {
            "pretrained_name": init.get("pretrained_name"),
            "model_path": init.get("model_path"),
        }
    )
    if cfg_setup.model_path not in (None, "None", "") and str(cfg_setup.model_path).strip():
        pass
    elif cfg_setup.pretrained_name:
        cfg_setup.model_path = None
    else:
        raise ValueError("init requires pretrained_name or model_path")

    model, _ = setup_model(cfg_setup, device)
    model.eval()
    # Hybrid のみ CTC に寄せる（NFA 足切りと同系）。純 RNNT/TDT（例: parakeet-tdt-0.6b-v3）は既定デコードのまま。
    if isinstance(model, EncDecHybridRNNTCTCModel):
        try:
            model.change_decoding_strategy(decoder_type="ctc")
        except Exception:
            pass
    if bool(init.get("use_local_attention", True)):
        try:
            model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])
        except Exception:
            pass
    return model, device


def main() -> None:
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    try:
        model, _device = _load_model(init)
    except Exception as e:
        _reply({"ok": False, "detail": f"{type(e).__name__}: {e}"})
        return
    _reply({"ok": True})

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        msg = json.loads(line)
        op = msg.get("op")
        if op == "shutdown":
            break
        if op != "transcribe":
            _reply({"ok": False, "detail": f"unknown op: {op!r}"})
            continue
        paths = msg.get("paths") or []
        if not paths:
            _reply({"ok": True, "texts": []})
            continue
        try:
            raw = model.transcribe(audio=list(paths), batch_size=int(msg.get("batch_size", len(paths))))
        except Exception as e:
            _reply({"ok": False, "detail": f"{type(e).__name__}: {e}"})
            continue
        if not isinstance(raw, list):
            raw = [raw]
        texts = [_hyp_to_text(h) for h in raw]
        if len(texts) != len(paths):
            _reply(
                {
                    "ok": False,
                    "detail": f"transcribe length mismatch: paths={len(paths)} texts={len(texts)}",
                }
            )
            continue
        _reply({"ok": True, "texts": texts})


if __name__ == "__main__":
    main()
