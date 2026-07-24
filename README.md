# FilteringCV / cv-preprocess

Common Voice から **TTS 学習用コーパス**を作るツールです。ライセンスは [Apache License 2.0](LICENSE)。

対話操作は **GUI が前提**です。CLI は CI・自動化・上級者向けです。

## すぐ始める（GUI）

```bash
# Windows: Dev Container 推奨（作成時に依存が入る）
# Linux / コンテナ内:
uv sync --extra sidon --extra gui --extra dev
./scripts/start-gui.sh
```

ブラウザで **http://127.0.0.1:8765** を開きます。

| 手順 | 画面 | やること |
|------|------|----------|
| 1 | **Setup** | YAML を選ぶか、`config/example.yaml` から作成（既定: `config/default.yaml`） |
| 2 | **Config** | `input.corpus_root` や話者フィルタなどを編集して保存 |
| 3 | **Jobs** | **Build（推奨）** を開始（scan→…→audit を一括実行） |
| 4 | **Clips / Coverage** | 結果の確認・override・カバレッジ確認 |

詳細手順は [docs/gui.md](docs/gui.md)。環境・GPU・extra は [docs/開発環境.md](docs/開発環境.md)。

## GUI の画面

| 画面 | 役割 |
|------|------|
| Setup | 設定 YAML の選択・新規作成（未バインド時） |
| Dashboard | パス・最近のジョブ・run manifest |
| **Jobs** | ビルダー段階の実行・進捗・キャンセル（下表の順番） |
| Config | YAML の Form / テキスト編集と検証・保存 |
| Coverage | 特徴量カバレッジ |
| Clips | カタログ閲覧・再生・override |
| Compare | 2 つの `work/` または出力ディレクトリの比較 |

設定の切替はサイドバーの **Switch config**（実行中ジョブがあると不可）。

## Jobs の順番（ビルダー）

YAML で `dataset_builder.enabled: true` のとき、Jobs では次の順で進みます。**初めては Build だけ**で十分です。

| # | Job | 何をするか | 主な出力 |
|---|-----|------------|----------|
| 1 | `scan` | コーパス / TSV の件数・パスを確認 | 概要（ジョブ結果） |
| 2 | `analyze` | 解析・品質ゲート・音声キャッシュ・カタログ作成（重い） | `work/catalog/`、`audio_cache` |
| 3 | `plan-split` | train / val / test の分割計画（意味はプロトコル依存・下表） | `work/plans/split_plan.json` |
| 4 | `select` | カバレッジに沿ったクリップ選択（同上） | `work/plans/selection_plan.parquet` |
| 5 | `materialize` | WAV・メタデータ等を出力ディレクトリへ書き出し | 最終コーパス一式 |
| 6 | `audit` | 選択・分割・出力の整合性チェック | 監査結果 |
| ★ | **`build`（推奨）** | **1〜6 を順に実行**（途中成果物があれば再開） | 上記すべて + `run_manifest.json` |

個別ステージは「analyze だけやり直す」「select だけ再実行」など向けです。`Force` は既存成果物があっても再実行します。

### `plan-split` と `select` の関係（よくある疑問）

Jobs 上の順番は常に **`plan-split` → `select`** ですが、**中身の「どちらが先か」は `dataset_builder.split.protocol` で変わります**。

| プロトコル | 実質の流れ | なぜ |
|------------|------------|------|
| **`unseen_speaker`（既定でよく使う）** | **先に話者を train/val/test に割当 → 各バケット内で select** | 同一話者が train と val/test に出ないようにするため。全体を先に select すると、あとで話者を分けたときにカバレッジが崩れる |
| **`seen_speaker` / `single_speaker`** | **先に全体で select → 選ばれたクリップに split を付与** | 話者またぎを許す（または単話者）ので、クリップ割当は選択後でよい |

「select してから split した方がいいのでは？」は後者ではその感覚どおりです。**話者を分けたい `unseen_speaker` では、今の順（話者計画 → バケット内選択）が意図どおり**です。設定の `split.protocol` を確認してください。詳細は [docs/dataset-builder.md](docs/dataset-builder.md)。

アルゴリズム・列定義は [docs/dataset-builder.md](docs/dataset-builder.md)、[docs/selection-algorithm.md](docs/selection-algorithm.md)、[docs/catalog-schema.md](docs/catalog-schema.md)。

## 設定で最初に触る場所

Setup で作った YAML（または [`config/example.yaml`](config/example.yaml)）を Config 画面かエディタで編集します。

- **`input.corpus_root`** … Common Voice の言語ルート（例: `…/ja`）
- **`speakers.include_client_ids`** … 空なら全話者。絞るなら `client_id` を列挙
- **`dataset_builder.enabled: true`** … ビルダー（GUI Jobs）を使う場合は true
- 音声チェーン / 品質ゲート … [docs/仕様.md](docs/仕様.md)

`config/default.yaml` と `config/*.local.yaml` は gitignore 対象です。

### `validated.tsv` と話者 ID

TSV はクォート付きフィールドで **物理行 ≠ 論理レコード** になり得ます。話者 ID はスプレッドシートや `wc -l` ではなく、本ツールの **scan（Jobs）** やパーサ結果を基準にしてください。

## ドキュメント

| 文書 | 内容 |
|------|------|
| [docs/gui.md](docs/gui.md) | GUI 起動・Setup・画面 |
| [docs/開発環境.md](docs/開発環境.md) | Dev Container / `uv` / GPU / optional extra |
| [docs/仕様.md](docs/仕様.md) | パイプライン・ゲート・設定キーの正 |
| [docs/dataset-builder.md](docs/dataset-builder.md) | ビルダー段階・CLI 参照 |
| [docs/architecture.md](docs/architecture.md) | Core API 構成 |
| [docs/追加仕様.md](docs/追加仕様.md) | 二次パイプライン・HiFi-GAN など |
| [docs/音素照合マニフェスト.md](docs/音素照合マニフェスト.md) | 音素マニフェスト |
| [docs/coverage-automation.md](docs/coverage-automation.md) | 希少音素カバレッジ自動確保 |
| [docs/migration-v1-v2.md](docs/migration-v1-v2.md) | レガシー `preprocess` からの移行 |

## オプション機能（必要なときだけ）

セットアップやゲートの詳細は [docs/開発環境.md](docs/開発環境.md) / [docs/仕様.md](docs/仕様.md) へ。

| 機能 | 概要 |
|------|------|
| **Sidon**（既定例） | enhance 用。`uv sync --extra sidon` |
| **Dasheng / SGMSE / WPE+DFN / HiFi-GAN** | 設定に応じて対応 extra を追加 |
| **NFA**（Dev Container） | `nfa_gate`。コンテナに別 venv あり。`mfa_gate` と同時 true 不可 |
| **MFA**（ホスト） | Dev Container 非同梱。conda 等で `mfa` を用意 |
| **レガシー preprocess / secondary** | `dataset_builder.enabled: false` 時の逐次前処理など。CLI 向け |

## CLI（自動化・上級者向け）

エントリ: `cv-preprocess`（`python -m cv_preprocess`）。ヘルプは `cv-preprocess --help`。

GUI と同じ Core API を呼びます。Jobs と同じビルダー段階:

```text
scan → analyze → plan-split → select → materialize → audit
# 一括:
cv-preprocess build -c config/default.yaml
```

その他（レガシー・ユーティリティ）: `preprocess`, `secondary`, `phoneme-manifest`, `suggest-mfa-g2p-map`, `suggest-nfa-g2p-map`, `dataset-partition`, `compare-runs`, `benchmark-selection`, `text-normalize`, `phonemize` など。詳細は各 `--help` と上表のドキュメント。

### 希少音素カバレッジ自動化

目標件数に届かない音素・モーラ等を、全件重解析せずに補う機能です。

```bash
cv-preprocess coverage-index -c config/default.yaml -o output/coverage/clip-index.jsonl
cv-preprocess coverage-plan  -c config/default.yaml --index output/coverage/clip-index.jsonl -o output/coverage/plan.json
cv-preprocess coverage-run   -c config/default.yaml --index output/coverage/clip-index.jsonl -o output/coverage/run-001 --dry-run
```

詳細は [docs/coverage-automation.md](docs/coverage-automation.md)。

## 計算バックエンド

`compute.backend: auto`（既定）で Polars、不可時は Python にフォールバック。`run_manifest.json` にステージ時間などが記録されます。
