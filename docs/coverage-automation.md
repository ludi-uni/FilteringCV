# 希少音素カバレッジ自動確保（Coverage Automation）

## 目的

全候補を重い品質解析にかける前に、テキスト / G2P ベースの軽量インデックスで
**不足している音素・モーラ・バイフォン等を補える候補だけ**を選び、品質ゲート結果を反映しながら反復解析します。

## GUI からの実行

1. Config で `coverage.enabled: true` と目標特徴を設定して保存（`insert_before_analyze: true` が既定）
2. **Build** を実行 → `scan` → **coverage-index/run** → `analyze`（既解析は再利用）→ …
3. Coverage 画面で active-run の達成状況を確認

analyze の**前**に有望候補だけ品質解析するため、全件の重い解析待ちを避けられます。
残りのクリップは後続の analyze が埋めます（coverage 済みはスキップ）。

## なぜ全件重解析をしないのか

品質解析（デコード + パイプライン + ゲート）はコストが高い一方、希少特徴を含むクリップは全体のごく一部です。
軽量インデックスで期待効用の高い候補だけを解析することで、目標カバレッジ到達までの解析量を大幅に削減できます。

## 処理段階

| コマンド | 役割 |
|----------|------|
| `coverage-index` | 全候補の軽量インデックス（JSONL）を作成。ASR/MFA/ノイズ除去などは行わない |
| `coverage-plan` | 現在の合格カバレッジと不足量から、次バッチ候補とスコア内訳を計画 |
| `coverage-run` | 計画 → 品質解析 → 再計画を目標達成または停止条件まで反復 |
| `coverage-report` | 実行ディレクトリから JSON / CSV / Markdown / HTML を再生成 |

既存の `analyze` / `select` は変更せず併用します。`coverage-run` は内部で部分解析 API `analyze_clips()` を呼びます。

## 設定例

`coverage.enabled: false`（既定）なら既存動作に影響しません。

```yaml
coverage:
  enabled: true
  counting_mode: per_clip

  features:
    phoneme:
      enabled: true
      required: true
      default_target: 0
      targets:
        v: 5
        fy: 5
        dy: 5
        gw: 5
        py: 20
        by: 30
        my: 40
        ny: 50
        hy: 80
        gy: 100
        ry: 120
    mora:
      enabled: true
      required: false
      default_target: 0
      # 索引と同じカタカナ表記（ローマ字 kya ではない）
      targets:
        ニャ: 10
        ピュ: 5
        ヴァ: 5
    biphone:
      enabled: true
      required: false
      default_target: 0
      targets:
        "dy-u": 5
        "gw-a": 5
        "py-o": 10
    positioned_phoneme:
      enabled: true
      required: false
      default_target: 0
      targets:
        "word_initial:py": 10
        "word_initial:ny": 20

  required_features:
    - "phoneme:v"
    - "phoneme:fy"
    - "phoneme:dy"
    - "phoneme:gw"
    - "phoneme:py"
    - "phoneme:by"
    - "phoneme:my"
    - "phoneme:ny"
    - "phoneme:hy"
    - "phoneme:gy"
    - "phoneme:ry"
  optional_features:
    - "biphone:dy-u"
    - "biphone:gw-a"
    - "positioned_phoneme:word_initial:py"

  pass_probability:
    default: 0.5
    prior_strength: 10
    min_probability: 0.05
    max_probability: 0.95

  analysis_cost:
    base: 1.0
    duration_weight: 0.1

  diversity:
    speaker_penalty: 0.5
    duplicate_text_penalty: 1.0
    max_per_speaker_per_batch: 5

  batch:
    min_size: 20
    max_size: 500
    safety_factor: 1.3

  limits:
    max_iterations: 50
    max_analyzed_clips: 10000
    max_audio_hours: 100

  rare_rescue:
    enabled: true
    target_features:
      - "phoneme:v"
      - "phoneme:fy"
      - "phoneme:dy"
      - "phoneme:gw"
      - "phoneme:py"
      - "phoneme:by"
      - "phoneme:my"
      - "phoneme:ny"
    min_candidate_clips: 1
    min_candidate_speakers: 1
    stricter_quality_gate: true
```

網羅版の具体値は `config/default.yaml` / `config/example.yaml` を参照してください。

## スコアリング

```text
score(c) = expected_pass_probability(c)
         × marginal_coverage_gain(c)
         × diversity_factor(c)
         ÷ estimated_analysis_cost(c)
```

- **利得**: 不足特徴ごとに `target_weight × rarity_weight × deficit_ratio` を加算（複数特徴は合算）
- **希少度**: `1 / sqrt(pool_candidate_count + 1)`
- **通過率**: 話者別実績をグローバル率でベイズ平滑化 + 軽量品質特徴で補正
- **コスト**: `base + duration_sec × duration_weight`
- **多様性**: 同一話者・同一正規化テキストにペナルティ。`max_per_speaker_per_batch` で上限

バッチ選択は単純 Top-N ではなく、1件選ぶたびに仮不足量を更新して再スコアします。

## 実行例

```bash
# 1) 軽量インデックス
cv-preprocess coverage-index \
  -c config/default.yaml \
  --input /path/to/validated.tsv \
  -o output/coverage/clip-index.jsonl

# 2) 次バッチ計画のみ
cv-preprocess coverage-plan \
  -c config/default.yaml \
  --index output/coverage/clip-index.jsonl \
  --accepted-metadata work/catalog/clips.parquet \
  -o output/coverage/plan.json

# 3) dry-run（解析せず計画と推定のみ）
cv-preprocess coverage-run \
  -c config/default.yaml \
  --index output/coverage/clip-index.jsonl \
  --accepted-metadata work/catalog/clips.parquet \
  -o output/coverage/run-001 \
  --dry-run

# 4) 本実行
cv-preprocess coverage-run \
  -c config/default.yaml \
  --index output/coverage/clip-index.jsonl \
  --accepted-metadata work/catalog/clips.parquet \
  -o output/coverage/run-001

# 5) 再開
cv-preprocess coverage-run \
  -c config/default.yaml \
  --index output/coverage/clip-index.jsonl \
  --resume output/coverage/run-001

# 6) レポート再生成
cv-preprocess coverage-report --run-dir output/coverage/run-001
```

`--accepted-metadata` には dataset builder の `clips.parquet`、またはレガシー `metadata.jsonl` を渡せます。

## レポートの読み方

`output/coverage/run-001/` に以下が生成されます。

| ファイル | 内容 |
|----------|------|
| `run-state.json` | 反復状態・解析済み ID・不足量（再開用） |
| `coverage-summary.json` / `.csv` | 特徴ごとの目標・採用数・不足・到達見込み |
| `iteration-history.jsonl` | 反復ごとの合格率・補えた特徴 |
| `selected-batches.jsonl` | 候補ごとのスコア内訳（選定理由追跡） |
| `unreachable-features.csv` | 到達不能 / 候補枯渇特徴 |
| `report.md` / `report.html` | 人間向け要約 |

## 希少音素救済

`rare_rescue.enabled: true` のとき、指定（または必須）希少特徴を含む候補をレポート上 `rare_rescue` として区別します。
候補除外の下限は緩和しますが、**品質基準は緩和しません**（`stricter_quality_gate` は将来の追加確認用フラグ）。

通常の `select` における `feature_support.min_utterances` 等の除外ロジックは、coverage 経路では適用しません。

## 停止理由

| reason | 意味 |
|--------|------|
| `complete` | 必須特徴がすべて目標達成 |
| `candidate_exhausted` | 不足特徴を含む未解析候補が無い |
| `unreachable` | 候補数から見て目標達成不能 |
| `likely_unreachable` | 推定通過率込みで到達が厳しい（レポート上） |
| `analysis_budget_exceeded` | 解析件数 / 音声時間上限 |
| `iteration_limit_reached` | 反復上限 |
| `dry_run` | `--dry-run` 完了 |
| `cancelled` / `failed` | 中断・失敗 |

## 達成不能時の対応

1. `unreachable-features.csv` で不足特徴と残候補数を確認
2. 目標値を下げる、またはコーパスを追加して `coverage-index --incremental` を再実行
3. `coverage-plan` で推定必要解析数を見直す

## 既存 analyze / select との関係

- **analyze**: 全件カタログ作成。coverage は `analyze_clips()` で部分解析し、既存カタログへマージ
- **select（従来の限界効用）**: 時間目標に対する貪欲選択は継続
- **select（coverage-aware・既定オン）**: 同じ `coverage.features` 目標を最終セットの必須制約として予約選択する
- 成果物の再利用: 既に catalog にある `clip_id` は再解析しない（`reuse_existing=true`）
- 設定変更や index fingerprint 不一致時は resume を拒否します

## coverage-aware select

Force Build（coverage-run）は希少特徴クリップを eligible に入れる役割、**最終保証は select 側**で行います。

**既定で有効**です（`selection.coverage_constraints.enabled: true`）。Build / 単独 `select` のどちらでも同じ。

```text
coverage-index → coverage-run → eligible
  → Phase A: 必須カバレッジ予約（貪欲集合被覆）
  → Phase B: 既存 greedy で残り時間充填（initial_selected 連携）
  → Phase C: coverage 保護 local search
  → Phase D: coverage audit / missing-features
  → materialize
```

### 使い方（最短）

1. YAML の `coverage.features` に目標を書く（`config/default.yaml` に例あり）
2. GUI なら **Jobs → Build**、または CLI で `cv-preprocess build -c config/default.yaml`
3. select 後に `work/reports/selection/coverage-audit.csv` を見る

| status 例 | 意味 |
|-----------|------|
| `configured_target_satisfied` | 設定どおり達成 |
| `corpus_limit_satisfied` | 設定は高いが eligible 上限まで取れた |
| `selection_constraint_conflict` | 候補はあるが話者/時間上限などで選べない |
| `not_present_in_index` / `not_present_in_eligible` | 索引または eligible に特徴が無い → `missing-features.json` |

### 目標の解釈

| YAML | 解釈 |
|------|------|
| `v: 5` | `minimum = 5`, `desired = 5`（必須条件として扱う） |
| `v: {minimum: 2, desired: 5}` | 最低 2 / 希望 5 |
| `effective_*` | `min(configured_*, eligible_clip_count)`（1 クリップ 1 カウント） |

### 設定例

```yaml
selection:
  coverage_constraints:
    enabled: true                 # 既定。切るときだけ false
    violation_policy: best_effort   # fail | warn | best_effort
    use_coverage_targets: true
    required_families: [phoneme, mora]
    optional_families: [biphone]
    preserve_during_local_search: true
  acoustic_diversity:
    enabled: true                 # 既定。切るときだけ false
    backend: lightweight
    weight: 0.15
```

### CLI

通常はフラグ不要（既定オン）:

```bash
cv-preprocess select -c config/default.yaml
```

上書き用途:

```bash
cv-preprocess select \
  --config config/default.yaml \
  --coverage-aware \
  --coverage-policy best_effort \
  --coverage-audit-output work/reports/selection \
  --disable-acoustic-diversity
```

出力（既定: `{work_dir}/reports/selection/`）:

- `coverage-audit.json` / `.csv`
- `coverage-contributions.jsonl`
- `missing-features.json`
- `acoustic-diversity-summary.json`
