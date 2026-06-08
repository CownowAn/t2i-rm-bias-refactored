# Counterfactual Debiasing — Implementation Plan

## Context

`search/` 알고리즘은 주어진 reward model의 undesirable bias attribute을 발견합니다
(예: `outputs/search/20260601-154143/`). 이 plan은 그 결과를 활용해 counterfactual pair
기반으로 reward model을 debias하는 파이프라인을 `debias/counterfactual/` 아래에 구축하는
방법을 정의합니다. 참고: [OVERVIEW.md](OVERVIEW.md).

핵심 아이디어:
- 각 prompt $x$에 대해 $\mathcal{B}_x = \{k : W_{x,k} > \tau \;\land\; \text{undesirable}(k)\}$를
  per-prompt $W_x$ matrix에서 선정 (top-N).
- 각 (prompt, attr)에 대해 detection cache로 그 attr이 있는 baseline 이미지를 찾는다.
- FLUX-Kontext로 **해당 attribute만** 제거한 counterfactual image 생성.
- (winner = edited, loser = original) BT loss로 reward model finetune.

**왜 PoC 먼저**: editor의 attr별 capability에 결과가 90% 좌우됩니다. main을 돌리기 전에
어떤 attr가 깔끔히 toggle되는지 알아야 budget을 합리적으로 배분할 수 있습니다.

---

## Goals

### PoC — Editor Capability Survey
주어진 `per_prompt_W_step{N}_topic{T}.json` 파일에 대해:
1. 각 attr별로 $\mathcal{B}_x$ 기준을 만족하는 prompt들을 식별.
2. attr마다 `--n_prompts_per_attr`개 prompt × `--n_images_per_prompt`개 source image 샘플링.
3. FLUX-Kontext로 edit → detector로 verification (attr이 진짜 제거됐는지).
4. **per-attr success rate 표 + 시각 점검용 thumbnail grid** 생성.

→ "editable한 attr subset"이 식별되면 그것만으로 main pipeline 진행.

### Main pipeline (PoC 이후)
1. Full-scale로 PoC modules 재활용해 모든 valid (prompt, attr) pair에 대해 counterfactual 생성.
2. 외부 preference data (Pick-a-Pic 등)와 mixing.
3. Reward model에 LoRA 부착 (또는 ImageReward는 full) + BT loss로 finetune.
4. Held-out CFPair val + standard benchmark (Pick-a-Pic test, HPSv2.1)에서 평가.

> Main pipeline의 자세한 결정은 PoC 결과 (editable subset, success rate)를 보고 정합니다.
> 이 plan에서는 module skeleton만 정의합니다.

---

## Directory layout

```
debias/counterfactual/
├── OVERVIEW.md                  # 기존; 보존
├── PLAN.md                      # 이 파일
├── README.md                    # 사용법 + flow diagram
├── __init__.py
├── types.py                     # 공용 dataclass (단일 source of truth)
├── io_utils.py                  # path helpers, run_dir 레이아웃
│
├── selection/                   # Stage 1: (prompt, attr, image_id) 선정
│   ├── __init__.py
│   ├── per_prompt_w_loader.py
│   ├── humanness.py             # 옵션 re-check
│   ├── attr_selector.py         # B_x, top-N per prompt
│   ├── detection_lookup.py
│   └── source_image_sampler.py
│
├── edit/                        # Stage 2: FLUX-Kontext edit
│   ├── __init__.py
│   ├── instruction_builder.py
│   ├── editor_runner.py
│   └── output_paths.py
│
├── verify/                      # Stage 3: detector verification
│   ├── __init__.py
│   ├── pair_validator.py
│   ├── side_effect_check.py     # 옵션: 다른 attr drift 체크
│   └── thumbnail_grid.py
│
├── poc/                         # PoC entrypoint (capability survey)
│   ├── __init__.py
│   ├── survey.py                # orchestration
│   ├── report.py                # JSON + thumbnail report writers
│   ├── run_poc.py               # Python CLI
│   └── run_poc.sh               # bash wrapper
│
├── pairs/                       # Stage 4 (main): CFPair 구성
│   ├── __init__.py
│   ├── pair_builder.py
│   ├── pref_mixer.py            # CF + external pref mixing
│   └── pair_store.py            # jsonl I/O
│
├── finetune/                    # Stage 5 (main): BT-loss finetune
│   ├── __init__.py
│   ├── bt_loss.py
│   ├── data_module.py
│   ├── lora_adapters.py
│   ├── trainer.py
│   ├── run_finetune.py
│   └── run_finetune.sh
│
└── tests/
    ├── __init__.py
    └── test_selection_schemas.py
```

### Output 경로 규약 (`io_utils.py` 상수)

| 산출물 | 경로 |
|---|---|
| Edited images | `/nfs/data/sohyun/projects/t2i-rm-bias/counterfactuals/<run_id>/topic_<T>/<attr_hash>/edited_<hash>.png` |
| PoC 리포트 | `outputs/counterfactual_poc/<run_id>/{selected_pairs,edit_results,survey_report}.json` |
| PoC thumbnails | `outputs/counterfactual_poc/<run_id>/thumbs/<attr_hash>.png` |
| Main pairs | `outputs/counterfactual/<run_id>/pairs.jsonl` |
| Finetune ckpts | `outputs/counterfactual/<run_id>/finetune/best.pt` |

---

## Critical reuse map

| 새 모듈 | 재사용 대상 (file:line) |
|---|---|
| `edit/editor_runner.py` | [`FluxKontextApplier`](../../search/models/editor/flux_kontext.py#L8) — sync `.apply(image_path, instruction, output_path)`, lazy-load, idempotent (output_path 존재 시 skip). 사용 예: [`evolution.py:94`](../../search/pipeline/evolution.py#L94). |
| `edit/output_paths.py` | [`edited_image_filename`](../../search/utils/io.py#L12) — `md5(attr|prompt|baseline_stem)[:12]` 기반 deterministic 파일명. |
| `edit/instruction_builder.py` | 기본은 `f"Remove {attr.lower().rstrip('.')}."`. VLM 보강은 옵션. |
| `selection/humanness.py` | [`AttributeUndesirabilityFilter.filter_by_humanness`](../../search/pipeline/attribute_filter.py#L37) — async, YES/NO 응답, parallel. |
| `selection/detection_lookup.py` | Detection cache 포맷: `{model_key: {image_id: {attr: 0|1}}}` (e.g. `Qwen/Qwen3.5-9B::auto`). 참고: [`_load_detection_cache`](../../search/pipeline/bon_amplified_evo.py#L777). |
| `selection/source_image_sampler.py` | [`load_baselines_from_manifest`](../../search/pipeline/baselines.py#L53), [`BaselineImage`](../../search/data/types.py#L12). `baseline_root` 처리에 주의 — manifest에 절대경로면 `""` 전달. |
| `verify/pair_validator.py` | [`build_detector`](../../search/models/detector/__init__.py#L11) → [`VisionLLMDetector.detect(image_paths, prompts, attribute)`](../../search/models/judge/vlm_judge.py#L165) — **attribute는 단일 문자열**. K attrs면 K번 호출. |
| `pairs/pair_store.py`, `poc/report.py` | [`save_json`, `load_json`](../../search/utils/io.py#L19). |
| `finetune/data_module.py` | [`search/models/reward/{imagereward,pickscore,hpsv3}.py`](../../search/models/reward/) — 기본 backbone load. |
| `poc/run_poc.sh`, `finetune/run_finetune.sh` | [`baselines/generate_mjhq.sh`](../../baselines/generate_mjhq.sh#L1) 패턴 (arg parsing, log/PID 관리). |

### 중요 정정 (Plan agent 가정 → 실제)

- `ba_expand_step{N}_topic{T}.json`의 `humanness_rejected` field는 **desirable로 분류되어 search에서 제외된** attr 리스트입니다 (즉 우리가 원하는 게 *아닌* attr). 진짜 undesirable set은 `acc_pool` (또는 per_prompt_W의 `attrs`) 그 자체 — 이미 search 단계에서 humanness filter를 통과한 상태.
- 따라서 **default**: `attrs` list 전체를 undesirable로 사용.
- **`--humanness_recheck`** 플래그: `attrs`에 대해 한 번 더 `filter_by_humanness` 돌려 더 strict한 set으로 좁힘 (안전장치).

---

## Public APIs (per-module signatures)

### `types.py` (단일 source of truth)

```python
@dataclass(frozen=True)
class PerPromptW:
    step_idx: int
    topic_id: int
    attrs: list[str]
    per_prompt_W: dict[str, list[float]]      # prompt_text → [W_0..W_K-1]
    per_prompt_r2: dict[str, float]

@dataclass(frozen=True)
class PromptAttrSelection:
    prompt_text: str
    topic_id: int
    attr: str
    w_value: float
    rank_in_prompt: int                       # 0 = top weight
    is_undesirable: bool

@dataclass(frozen=True)
class SourceImage:
    image_id: str
    image_path: Path
    prompt_text: str
    detected_attrs_snapshot: dict[str, int]   # cache 그대로

@dataclass(frozen=True)
class EditTask:
    selection: PromptAttrSelection
    source: SourceImage
    instruction: str
    edited_output_path: Path

@dataclass(frozen=True)
class EditResult:
    task: EditTask
    success: bool
    edited_attr_detected: int | None          # 0/1 on edited (0이어야 success)
    original_attr_detected: int | None        # sanity (1이어야 정상)
    side_effect_drift: dict[str, tuple[int, int]] | None
    error: str | None

@dataclass(frozen=True)
class CFPair:
    prompt_text: str
    topic_id: int
    attr: str
    winner_path: Path                         # edited
    loser_path: Path                          # original
    winner_image_id: str
    loser_image_id: str
    meta: dict                                # w_value, instruction, run_id

@dataclass
class AttrSurveyRow:
    attr: str
    n_attempted: int
    n_success: int
    success_rate: float
    sample_success: list[tuple[Path, Path]]   # (original, edited)
    sample_fail: list[tuple[Path, Path]]
    thumbnail_path: Path | None
```

### Selection layer

```python
# per_prompt_w_loader.py
def load_per_prompt_w(path: Path | str) -> PerPromptW
def iter_attr_columns(ppw: PerPromptW) -> Iterator[tuple[str, dict[str, float]]]

# humanness.py
def load_undesirable_from_search(ba_expand_path: Path, attrs: list[str]) -> set[str]
    # acc_pool / attrs 그대로 반환. (humanness_rejected는 *제외*하지 않음 — 이미 빠진 것들임)
async def recheck_humanness(attrs: list[str], cfg: AttrFilterConfig,
                            cache_cfg: CacheConfig | None) -> set[str]
async def resolve_undesirable_set(
    attrs: list[str],
    ba_expand_path: Path | None,
    *,
    recheck: bool,
    cfg: AttrFilterConfig,
    cache_cfg: CacheConfig | None,
) -> set[str]

# attr_selector.py
def select_per_prompt(
    ppw: PerPromptW,
    undesirable: set[str],
    tau: float,
    top_n: int,
) -> list[PromptAttrSelection]
def group_by_attr(
    selections: list[PromptAttrSelection],
) -> dict[str, list[PromptAttrSelection]]

# detection_lookup.py
def load_detection_cache(path: Path, detector_key: str) -> dict[str, dict[str, int]]
def find_image_ids_with_attr(
    detection: dict[str, dict[str, int]], attr: str, value: int = 1
) -> set[str]

# source_image_sampler.py
def sample_source_images(
    sel: PromptAttrSelection,
    baselines_for_prompt: list[BaselineImage],
    detection: dict[str, dict[str, int]],
    k_img: int,
    rng_seed: int,
) -> list[SourceImage]
```

### Edit layer

```python
# instruction_builder.py
def build_remove_instruction(attr: str) -> str
    # 기본: f"Remove the following from this image: {attr}"
    # 향후 VLM 보강 옵션 (instruction_gen 모듈) 추가 가능

# output_paths.py
def cf_root() -> Path                                  # NFS root
def edited_image_path(run_id: str, topic_id: int,
                      attr: str, image_id: str) -> Path

# editor_runner.py
class EditorRunner:
    def __init__(self, model_name: str, device: str,
                 guidance_scale: float, hf_cache_dir: str)
    def edit_one(self, task: EditTask) -> Path        # FluxKontextApplier 위임, idempotent
    def edit_many(self, tasks: list[EditTask]) -> list[Path]
```

### Verify layer

```python
# pair_validator.py
async def validate_edits(
    tasks: list[EditTask],
    detector: DetectorModel,
    *,
    check_original: bool = True,
    batch_size: int = 32,
) -> list[EditResult]
    # attr별로 묶어서 detect를 attr 단위로 batched. K attrs × N images per attr.

# side_effect_check.py
async def detect_side_effect_drift(
    task: EditTask,
    other_present_attrs: list[str],         # 그 source image에 detection=1이었던 다른 attrs
    detector: DetectorModel,
) -> dict[str, tuple[int, int]]            # attr → (before=1, after=0/1)

# thumbnail_grid.py
def build_grid(
    pairs: list[tuple[Path, Path]],          # (original, edited)
    out_path: Path,
    cols: int = 4,
    thumb_px: int = 256,
    title: str = "",
) -> Path                                    # PIL grid만, LPIPS 등 외부 dep 없음
```

### PoC entrypoint

```python
# poc/survey.py
@dataclass
class PoCConfig:
    tau: float
    top_n_per_prompt: int
    n_prompts_per_attr: int
    n_images_per_prompt: int
    humanness_recheck: bool
    side_effect_check: bool
    make_thumbnails: bool
    seed: int
    run_id: str

async def run_survey(
    ppw_path: Path,
    ba_expand_path: Path | None,
    detection_cache_path: Path,
    detector_key: str,
    baseline_manifest_path: Path,
    baseline_root: str,
    topic_id: int,
    cfg: PoCConfig,
    editor: EditorRunner,
    detector: DetectorModel,
    attr_filter: AttributeUndesirabilityFilter | None,
) -> SurveyResult
    # SurveyResult: dataclass(selections, edit_results, rows)

# poc/report.py
def write_selected_pairs(...) -> None       # → selected_pairs.json
def write_edit_results(...) -> None         # → edit_results.json
def build_survey_report(...) -> dict        # 메모리 dict
def write_survey_report(...) -> None        # → survey_report.json

# poc/run_poc.py
def parse_args() -> argparse.Namespace
async def main() -> None
```

### Main pipeline (skeleton — PoC 후 확정)

```python
# pairs/pair_builder.py
def build_pairs_from_edits(results: list[EditResult]) -> list[CFPair]

# pairs/pref_mixer.py
def load_external_pref_pairs(path: Path) -> list[CFPair]
def mix(cf: list[CFPair], external: list[CFPair],
        external_ratio: float, seed: int) -> list[CFPair]

# pairs/pair_store.py
def write_pairs_jsonl(pairs: list[CFPair], out_path: Path) -> None
def read_pairs_jsonl(path: Path) -> list[CFPair]

# finetune/bt_loss.py
def bt_loss(r_winner: Tensor, r_loser: Tensor,
            beta: float = 1.0, margin: float = 0.0) -> Tensor
def pair_accuracy(r_winner: Tensor, r_loser: Tensor) -> Tensor

# finetune/data_module.py
class PairDataset(Dataset)
def make_loader(ds, batch_size, shuffle) -> DataLoader

# finetune/lora_adapters.py
def wrap_with_lora(model, target_modules, r, alpha) -> nn.Module
def freeze_base(model) -> None

# finetune/trainer.py
@dataclass
class FinetuneConfig:
    lr: float; batch: int; steps: int
    val_every: int; lora_r: int; lora_alpha: int
    bt_beta: float; kl_gamma: float
def train(model_orig, lora_model, train_loader, val_loader,
          cfg: FinetuneConfig, out_dir: Path) -> Path
def evaluate(model, val_loader) -> dict[str, float]
```

---

## JSON schemas

### `selected_pairs.json`
```json
{
  "run_id": "smoke_test_20260601",
  "topic_id": 0,
  "step_idx": 12,
  "tau": 0.05,
  "top_n_per_prompt": 3,
  "humanness_source": "search" | "recheck",
  "n_undesirable_attrs": 28,
  "selections": [
    {
      "prompt_text": "...",
      "attr": "Eyes are exaggerated and high-contrast...",
      "w_value": 0.114,
      "rank_in_prompt": 0,
      "source_image_ids": ["c8530f3f18_07", ...],
      "source_image_paths": ["/nfs/.../baseline_c8530f3f18_07.png", ...]
    }
  ]
}
```

### `edit_results.json`
```json
{
  "run_id": "...",
  "topic_id": 0,
  "results": [
    {
      "prompt_text": "...",
      "attr": "...",
      "source_image_id": "c8530f3f18_07",
      "source_image_path": "/nfs/.../baseline_c8530f3f18_07.png",
      "edited_image_path": "/nfs/.../counterfactuals/.../edited_a3f9b2.png",
      "instruction": "Remove the following from this image: ...",
      "success": true,
      "edited_attr_detected": 0,
      "original_attr_detected": 1,
      "side_effect_drift": null,
      "error": null
    }
  ]
}
```

### `survey_report.json`
```json
{
  "run_id": "...",
  "topic_id": 0,
  "config": {
    "tau": 0.05, "top_n_per_prompt": 3,
    "n_prompts_per_attr": 5, "n_images_per_prompt": 4,
    "humanness_recheck": false, "side_effect_check": false
  },
  "per_attr": [
    {
      "attr": "...",
      "n_attempted": 20,
      "n_success": 14,
      "success_rate": 0.70,
      "thumbnail_path": "outputs/counterfactual_poc/.../thumbs/a3f9b2.png",
      "sample_success_paths": [["orig.png", "edited.png"], ...],
      "sample_fail_paths": [["orig.png", "edited.png"], ...]
    }
  ],
  "global": {
    "n_attrs": 28, "n_attempted_total": 560,
    "mean_success_rate": 0.51,
    "editable_subset": ["attr_X", "attr_Y", ...]
  }
}
```

---

## PoC CLI

### `debias/counterfactual/poc/run_poc.sh`
```bash
bash debias/counterfactual/poc/run_poc.sh \
  --gpus 0 \
  --per_prompt_W_path outputs/search/<RUN>/per_prompt_W_step<N>_topic<T>.json \
  [--ba_expand_path outputs/search/<RUN>/ba_expand_step<N>_topic<T>.json] \
  --topic_id 0 \
  [--run_id smoke_test] \
  --detection_cache_path outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json \
  --detector_key "Qwen/Qwen3.5-9B::auto" \
  --baseline_manifest /nfs/.../topic_0/black-forest-labs-FLUX.1-dev/manifest.json \
  --baseline_root "" \
  --prompts_dir clustering/output/mjhq \
  --tau 0.0 \
  --top_n_per_prompt 3 \
  --n_prompts_per_attr 5 \
  --n_images_per_prompt 4 \
  [--humanness_recheck] \
  [--humanness_model openai/gpt-5] \
  --flux_model black-forest-labs/FLUX.1-Kontext-dev \
  --editor_device cuda:0 \
  --guidance_scale 2.5 \
  --hf_cache_dir /nfs/data/sohyun/models \
  --detector_model "Qwen/Qwen3.5-9B" \
  [--detector_vllm_base_url http://localhost:8000/v1] \
  --seed 42 \
  --cf_root /nfs/data/sohyun/projects/t2i-rm-bias/counterfactuals \
  --report_root outputs/counterfactual_poc \
  [--side_effect_check] \
  [--make_thumbnails]
```

`--ba_expand_path` 미지정 시 `per_prompt_W_path`에서 자동 추론 (`per_prompt_W_step` → `ba_expand_step`).

### PoC end-to-end flow

```
run_poc.sh → run_poc.py main():
  1. types.PerPromptW = load_per_prompt_w(--per_prompt_W_path)
  2. undesirable = await resolve_undesirable_set(
         ppw.attrs, --ba_expand_path,
         recheck=--humanness_recheck, cfg=AttrFilterConfig(...)
     )
     # default: undesirable = set(ppw.attrs)
     # recheck: filter_by_humanness로 다시 좁힘
  3. selections = select_per_prompt(ppw, undesirable, --tau, --top_n_per_prompt)
  4. by_attr = group_by_attr(selections)
  5. 각 attr별 --n_prompts_per_attr 개 prompt 샘플 (rng-seeded)
  6. detection = load_detection_cache(--detection_cache_path, --detector_key)
  7. baselines = load_baselines_from_manifest(...)
  8. tasks: list[EditTask] = []
     for sel in sampled_selections:
         sources = sample_source_images(sel, baselines[sel.prompt_text],
                                        detection, k_img=--n_images_per_prompt)
         for src in sources:
             instr = build_remove_instruction(sel.attr)
             out  = edited_image_path(--run_id, sel.topic_id, sel.attr, src.image_id)
             tasks.append(EditTask(sel, src, instr, out))
     write_selected_pairs(tasks → selected_pairs.json)
  9. editor = EditorRunner(...)
     for task in tasks: editor.edit_one(task)           # idempotent
 10. detector = build_detector(DetectorConfig(...))
     results = await validate_edits(tasks, detector,
                                    check_original=True)
     write_edit_results(results → edit_results.json)
 11. if --side_effect_check:
         await detect_side_effect_drift(...) on top-residual attrs
 12. report = build_survey_report(results, thumbs_root)
     write_survey_report(report → survey_report.json)
 13. if --make_thumbnails:
         for row in report.per_attr: build_grid(...)
```

---

## Main pipeline end-to-end (skeleton)

PoC 결과를 보고 결정:
- `editable_subset` = success_rate ≥ 0.5인 attr만 main에 사용.
- 그 외 attr는 별도 방법 (Tier C2/C3 등) 으로 나중에.

```
debias/counterfactual/finetune/run_finetune.sh →
  Phase A (full-scale pair gen — PoC modules 재사용):
    1. PoC flow 1-12와 동일, but --n_prompts_per_attr ALL, K_img 크게.
    2. build_pairs_from_edits(EditResult[success]) → CFPair[]
    3. (옵션) load_external_pref_pairs(Pick-a-Pic subset)
    4. mix(cf, external, ratio=0.5) → final pairs
    5. write_pairs_jsonl → outputs/counterfactual/<run_id>/pairs.jsonl
  Phase B (finetune):
    6. load base reward model (PickScore / HPSv3 / ImageReward)
    7. wrap_with_lora (PickScore/HPSv3) or freeze nothing (ImageReward)
    8. PairDataset + DataLoader (val split: 10%)
    9. trainer.train(...) with bt_loss + (R - R_ref)² KL term
   10. evaluate on (a) CF val (b) Pick-a-Pic test (c) D1 search recurrence
   11. save best ckpt
```

---

## Verification protocol

### Smoke run (5분 budget)

```bash
bash debias/counterfactual/poc/run_poc.sh \
  --gpus 0 \
  --per_prompt_W_path outputs/search/20260601-154143/per_prompt_W_step12_topic0.json \
  --topic_id 0 \
  --run_id smoke_test \
  --tau 0.05 \
  --top_n_per_prompt 2 \
  --n_prompts_per_attr 2 \
  --n_images_per_prompt 1
```

### Acceptance criteria

1. **Files exist**:
   - `outputs/counterfactual_poc/smoke_test/selected_pairs.json` — `selections` non-empty, 각 entry `len(source_image_ids) == 1`.
   - 각 task별 edited PNG가 `/nfs/data/sohyun/.../counterfactuals/smoke_test/...`에 존재.
   - `edit_results.json` — task 수만큼 row.
   - `survey_report.json` — attr별 row + `thumbnail_path` non-null.

2. **Sanity**: `edit_results.json`의 모든 row에서 `original_attr_detected == 1` (sampler가 attr=1만 골라야 정상).

3. **Idempotence**: 같은 command 재실행 시
   - FluxKontextApplier가 cache hit log 출력하며 skip
   - Detector는 CacheConfig 통해 동일 응답 재사용
   - 전체 wall time 첫 실행 대비 < 10%

4. **Negative test**: `--tau 999.0` → 빈 `selections` + clean exit + informative log.

5. **Visual spot-check**: top-1 success-rate attr와 bottom-1의 thumbnail grid 열어보고
   editor가 진짜 그 attr를 제거했는지 사람 눈으로 확인.

### PoC 후 go/no-go 결정

- **Go**: `editable_subset`의 attr 수 ≥ 10, 그리고 mean success rate ≥ 0.5 → main pipeline 착수.
- **No-go**: editor가 대부분의 attr에 실패 → instruction engineering 또는 다른 editor (InstructPix2Pix 등) 시도 또는 Tier C3 (DPO with prompt-engineered counter-pair) 로 우회.

---

## Open decisions (deferred)

다음은 PoC 시작 후 또는 결과 보고 결정:

1. **Instruction phrasing**: `f"Remove {attr}"` vs LLM이 attr → editing instruction 보강. 우선 default로 시작.
2. **Side-effect threshold**: 다른 attr drift 허용량. 우선 logging만, filter는 main pipeline에서.
3. **CFPair ↔ external pref ratio**: 0.5? 0.3? PoC 결과의 attr 수에 따라.
4. **LoRA rank / target_modules**: reward model architecture별로 다름. finetune phase에서 검증.
5. **Quality control human spot-check**: PoC budget의 일부를 사람 검수에 배정할지 (10% sample).