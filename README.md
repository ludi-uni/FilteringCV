# FilteringCV / cv-preprocess

Common Voice 向けの音声・テキスト前処理パッケージです。ライセンスは [**Apache License 2.0**](LICENSE)（`pyproject.toml` の `license` と一致）。

## セットアップ

- **Windows**: Dev Container（`.devcontainer`）の利用を推奨します。作成時に **`uv sync --extra sidon --extra dev`** が走り、[`config/default.yaml`](config/default.yaml) の **enhance での Sidon**（`sidon_restore`・`sidon_after_enhance_split`）に必要な依存が `.venv` に入ります。**NeMo Forced Aligner 用の別 venv**（`/opt/nfa-venv`）と NeMo の `align.py` ツリーもイメージに含まれます。**MFA（conda）は `.venv` に入れていません**（NFA と SGMSE の protobuf 要件が衝突するため `.venv` に NeMo を同居させない）。
- **Linux**: リポジトリをクローンし、既定構成に合わせるなら次で足ります。

  ```bash
  uv sync --extra sidon --extra dev
  ```

  **Dasheng**（`denoise.method: dasheng`）や **SGMSE**、**HiFi-GAN** を設定に含める場合は、それぞれ **`--extra dasheng`** / **`--extra sgmse`** / **`--extra hifigan`** を追加。**NARA-WPE＋DeepFilterNet** は **`--extra wpe_dfn`**（`deepfilterlib` のビルドに Rust / `cargo` が要ることがあります）。詳細は [docs/開発環境.md](docs/開発環境.md) を参照。

PyTorch は **CUDA 12.x（公式ホイール `cu128`）** 向けに `pyproject.toml` で指定しています。

詳細は [docs/開発環境.md](docs/開発環境.md) を参照してください。

### MFA（Montreal Forced Aligner）と `mfa_gate`（ホスト）

`config` の **`mfa_gate.enabled: true`** で、ノイズ除去などの音声チェーンの直後に **`mfa align`** による強制アライメント足切りが走ります（[docs/仕様.md](docs/仕様.md) §5.2）。

**Dev Container には MFA を入れていません。** ホストで使う場合は [conda-forge の `montreal-forced-aligner`](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) 等で `mfa` を用意し、辞書・音響モデル（例: `japanese_mfa`）を取得してください。日本語は **spaCy + Sudachi** が必要になることがあります。G2P 比較は **`mfa_gate.mfa_to_g2p_token_map_path`**（YAML）を推奨。草案集計は **`cv-preprocess suggest-mfa-g2p-map`**、形式例は [`config/mfa_to_openjtalk_phones.example.yaml`](config/mfa_to_openjtalk_phones.example.yaml)、手順は [docs/音素照合マニフェスト.md](docs/音素照合マニフェスト.md) を参照。

### NeMo Forced Aligner（NFA、Dev Container）

コンテナ内に **`NFA_PYTHON`**（`/opt/nfa-venv/bin/python`）と **`NFA_ALIGN_DIR`**（`align.py` 所在）が定義されています。前処理では **`nfa_gate.enabled: true`**（**`mfa_gate` と同時に true にしない**）で NeMo NFA による足切りが走ります。拒否理由は **`nfa_*` 専用**（[docs/仕様.md](docs/仕様.md) §5.3）。既定モデル例 **`nvidia/parakeet-tdt_ctc-0.6b-ja`**（16 kHz、`model_sample_rate_hz` でリサンプル）。**`nfa_gate.persistent_worker: true`（既定）** のときは NeMo を **1 subprocess 常駐**（`cv_preprocess/audio/nfa_align_worker.py`）で動かし **モデルは初回のみロード**する。従来の「バッチごとに `align.py` 起動」は `persistent_worker: false` または環境変数 **`CV_PREPROCESS_NFA_SUBPROCESS=1`**。音素マップを使わず **NeMo の `pred_text` と参照 `text_norm` をそれぞれ OpenJTalk G2P した音素列**で照合する場合は **`text.phonemize: true`** のうえ **`align_using_pred_text: true`** と **`compare_pred_text_to_norm: true`**（`compare_tokens_to_g2p` は `false`）。詳細は [docs/仕様.md](docs/仕様.md) §5.3。

**`phoneme_alignment_check`** 用の JSONLで MFA TextGrid を使う場合は、従来どおり `cv-preprocess phoneme-manifest --source mfa_textgrid …` とトークンマップ YAML が利用できます（MFA をホストに入れたとき）。

### HiFi-GAN（帯域補完、`bandwidth_extension`、任意）

**二次パイプライン**（`cv-preprocess secondary`、[docs/追加仕様.md](docs/追加仕様.md) §11）の `config.secondary.audio_pipeline.steps` に **`type: bandwidth_extension`** を置くと、[jik876/hifi-gan](https://github.com/jik876/hifi-gan) 互換の **Generator** でメル→波形を生成し帯域を補います。YAML の例は [`config/example.yaml`](config/example.yaml) の `secondary` コメント、フィールド定義は `cv_preprocess/config/audio_steps.py` の **`BandwidthExtensionStep`** を参照してください。

- **依存**: `uv sync --extra hifigan`（詳細は [docs/開発環境.md](docs/開発環境.md)）。
- **重みと設定**: 学習済み Generator のチェックポイントと、同リポジトリの **`config.json`** を手元に用意し、`generator_checkpoint`（必須）と `config_json`（チェックポイントと同じディレクトリに `config.json` がある場合は省略可）を設定します。
- **自動ダウンロードはしません**。公式配布が Drive 等のリンクになりがちで安定 URL にしにくいこと、利用するチェックポイントの選び方・ライセンス・再配布の扱いが利用者側で決まるためです。必要なファイルは各自で取得してください（方針は [docs/追加仕様.md](docs/追加仕様.md) §12）。

## ドキュメント

| 文書 | 内容 |
|------|------|
| [docs/仕様.md](docs/仕様.md) | パイプライン・品質ゲート・MFA/NFA・設定キーの正。**§18（末尾付録）**は CPU 側パフォーマンスの実装メモ |
| [docs/開発環境.md](docs/開発環境.md) | Dev Container、`uv` / GPU、optional extra（**Sidon** が既定構成。Dasheng・SGMSE・HiFi-GAN・WPE+DFN は設定に応じて追加） |
| [docs/音素照合マニフェスト.md](docs/音素照合マニフェスト.md) | `phoneme_alignment_check` 用 JSONL と `phoneme-manifest` |
| [docs/追加仕様.md](docs/追加仕様.md) | 多話者データセット論点・二次パイプライン・HiFi-GAN（§10–§12） |

## 使い方

エントリポイントは **`cv-preprocess`**（`python -m cv_preprocess` でも可）です。ヘルプは `cv-preprocess --help`、各サブコマンドは `cv-preprocess <command> --help` で確認できます。

### 設定ファイル

- 既定の雛形: [`config/default.yaml`](config/default.yaml) をコピーして `input.corpus_root` や `speakers.include_client_ids` などを自分の Common Voice 展開先に合わせて編集するか、[config/example.yaml](config/example.yaml) を参考にします。コミットしたくない差分だけを分けたい場合は `config/default.local.yaml` のように別名にし（`.gitignore` に含まれています）、`cv-preprocess … -c config/default.local.yaml` のように指定してください。
- パイプラインや品質ゲートの意味は上表の [docs/仕様.md](docs/仕様.md) を参照してください。

### 注意: `validated.tsv` の「行」と話者 ID（`client_id`）

Common Voice の `validated.tsv` は **タブ区切りのテキスト**ですが、文中に **ダブルクォート（`"`）** などがあり **RFC 風のクォート付きフィールド**になると、**1 レコードが複数の物理行**（ファイル上の改行）にまたがることがあります。

- **本パッケージ**は Python 標準の **`csv` モジュール**で読み込みます。**論理行（レコード）**の数は、エクスプローラや `wc -l`、**行単位の grep** で数えた物理行数より少なくなることがあります。
- **LibreOffice Calc** などで開く場合、テキストインポートの **「文字列の区切り」** を **`"`** にするか **空**にするかで、行の切り方が変わります。区切りを空に近い設定にすると **改行のたびに行が分かれ**、ある物理行の先頭に **本当の `client_id` 列ではない**文字列（他レコードのクォート内の続き）が **1 列目に見える**ことがあります。スプレッドシートや grep で「見つかった」からといって、パーサ上の話者 ID として存在するとは限りません。
- **`speakers.include_client_ids`** に使う値は、**本ツールの解釈（`cv-preprocess scan` の件数・警告、`load_validated_tsv` の結果）を基準**にしてください。`scan` では物理行と論理行が一致しないとき、説明付きの警告が出ます。

### よく使うコマンド

| コマンド | 説明 |
|----------|------|
| `cv-preprocess scan -c <設定.yaml>` | コーパスを走査し、件数・パスなどの概要を JSON で標準出力に出します（本処理の前に確認用）。 |
| `cv-preprocess preprocess -c <設定.yaml>` | 設定に従い前処理を実行し、終了時にレポートを JSON で標準出力に出します。CI やログ保存向けに進捗バーを止める場合は `--no-progress` を付けます。 |
| `cv-preprocess phoneme-manifest -c <設定.yaml> …` | **`phoneme_alignment_check` 用 JSONL** を生成。`--source g2p_text`（既定）で **OpenJTalk G2P（preprocess と同一）**。`mfa_textgrid` とトークンマップ YAML も可。 [docs/音素照合マニフェスト.md](docs/音素照合マニフェスト.md) |
| `cv-preprocess suggest-mfa-g2p-map -c <設定.yaml> --mfa-textgrid-root <dir> -o <out.yaml>` | TextGrid の phones と G2P を **同じ TSV 行**で突き合わせ、**`mfa_to_g2p_token_map_path` 用 YAML の草案**（投票・閾値）と `*_report.json` を出す。近似なので人手レビュー必須。`--help` で戦略・閾値。 |
| `cv-preprocess suggest-nfa-g2p-map -c <設定.yaml> … -o <out.yaml>` | NFA（CTM）トークンと G2P を突き合わせ、**`nfa_to_g2p_token_map_path` 用 YAML の草案**を出す。人手レビュー前提。 |
| `cv-preprocess secondary -c <設定.yaml>` | 一次 `preprocess` の出力に対し二次音声チェーンと再品質ゲートを適用（`config.secondary` が必要。[docs/追加仕様.md](docs/追加仕様.md) §11）。 |
| `cv-preprocess metadata-jsonl-to-validated-tsv -m <metadata.jsonl> [-o <validated.tsv>]` | `metadata.jsonl` から LJSpeech 互換の `validated.tsv`（3 列・ヘッダなし）を生成します。 |
| `cv-preprocess dataset-partition -m <metadata.jsonl> -o <out_dir> …` | `--group-by` で `quality_tier`・`split`・`split_quality_tier` などに応じて WAV をバケット別サブフォルダへ集約。`--only-tiers`・`--min-quality-score` 等で好みの抽出。各バケットに `metadata.jsonl` と `validated.tsv`（既定 WAV はシンボリックリンク）。詳細は `--help`。 |
| `cv-preprocess text-normalize "<文>"` | TTS 向けに正規化したテキストを 1 行で出力します（デバッグ用）。 |
| `cv-preprocess phonemize "<文>"` | 正規化のうえ G2P した音素列を出力します。読みをカナで出す場合は `--kana` を付けます。 |

本番の一連の流れは、**設定を用意 → `scan` で確認 → `preprocess`** が基本です。
