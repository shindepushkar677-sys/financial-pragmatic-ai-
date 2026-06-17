# AGENT_HANDOFF.md

## 1. Snapshot (Verified)

- Repository: `NLP_Proj`
- Branch: `main`
- Latest commit (checked locally): `5cb423f` (`lock production dependencies`)
- Date of this handoff update: 2026-04-12

This document reflects the **current code in the repository**. Nothing below is inferred from prior chat context.

---

## 2. Project Overview

The project is a financial transcript analysis system.

Primary backend flow:
- Transcript text/file -> parsing + segmentation -> intent prediction per segment -> aggregate score/signal/prediction/drivers -> JSON response.

There are **two frontends** in the repo:
- `frontend/` (CRA): stateless dashboard directly calling backend APIs.
- `frontend_v2/` (Vite + Tailwind + Supabase): richer UI with auth/history stored in Supabase, while inference still calls backend `/analyze`.

---

## 3. Verified Project Structure

```text
NLP_Proj/
├── AGENT_HANDOFF.md
├── README.md
├── backend/
│   ├── api/
│   │   ├── schemas.py
│   │   └── server.py                    # Active FastAPI entrypoint
│   ├── requirements.txt
│   ├── runtime.txt                      # Railway Python runtime pin
│   └── financial_pragmatic_ai/
│       ├── __init__.py
│       ├── analysis/
│       │   ├── conversation_vectorizer.py
│       │   ├── earnings_call_analyzer.py
│       │   ├── financial_insight_generator.py
│       │   ├── financial_signal_engine.py
│       │   ├── insight_engine.py
│       │   ├── market_predictor.py
│       │   ├── signal_statistics.py
│       │   ├── timeline_builder.py
│       │   ├── timeline_signal_analyzer.py
│       │   ├── transcript_analyzer.py
│       │   └── transcript_parser.py
│       ├── api/
│       │   └── server.py                # Legacy second API server module (not uvicorn target)
│       ├── data/
│       │   ├── pragmatic_intent_dataset_clean.csv
│       │   ├── pragmatic_intent_dataset.csv
│       │   ├── conversation_signal_dataset.csv
│       │   ├── intent_dataset.csv
│       │   └── ...
│       ├── evaluation/
│       │   └── better_than_fin/
│       │       ├── evaluate.py
│       │       ├── metrics.py
│       │       ├── utils.py
│       │       ├── visualize.py
│       │       └── results/
│       ├── inference/
│       │   ├── decision_engine.py
│       │   └── signal_extractor.py
│       ├── models/
│       │   ├── conversation_attention_model.py
│       │   ├── conversation_interaction_model.py
│       │   ├── financial_pragmatic_transformer.py
│       │   ├── financial_pragmatic_transformer_v2.py
│       │   ├── finbert_base.py
│       │   ├── finbert_intent_model.py
│       │   ├── intent_classifier.py
│       │   ├── pragmatic_attention.py
│       │   ├── pragmatic_input_layer.py
│       │   └── speaker_embedding.py
│       ├── testing/
│       │   ├── evaluate_model.py
│       │   ├── test_earnings_call_analyzer.py
│       │   ├── test_trained_model.py
│       │   └── test_transcript_analyzer.py
│       ├── training/
│       │   ├── train_conversation_model.py
│       │   ├── train_finbert_intent_v2.py
│       │   ├── train_intent_classifier.py
│       │   ├── train_pragmatic_transformer.py
│       │   ├── train_pragmatic_transformwer.py
│       │   └── train_v2_pipeline.py
│       └── utils/
├── frontend/                              # CRA frontend
│   ├── package.json
│   └── src/
│       ├── App.js
│       ├── api/client.js
│       ├── components/
│       └── pages/DashboardPage.js
└── frontend_v2/                           # Vite frontend with Supabase
    ├── package.json
    └── src/
        ├── App.jsx
        ├── supabaseClient.js
        └── components/
            ├── Auth.jsx
            ├── Compare.jsx
            ├── Insights.jsx
            ├── Navbar.jsx
            ├── Overview.jsx
            ├── Sidebar.jsx
            ├── Tabs.jsx
            └── TimelineChart.jsx
```

---

## 4. Backend Runtime Architecture (Active)

Active entrypoint:
- `backend/api/server.py`

FastAPI endpoints implemented:
- `POST /analyze`
- `POST /upload`
- `POST /compare`

Request schemas (`backend/api/schemas.py`):
- `TranscriptRequest`: `{ "transcript": str }`
- `CompareRequest`: `{ "transcript_1": str, "transcript_2": str }`

CORS in active server:
- `allow_origins=["*"]`
- `allow_credentials=True`
- `allow_methods=["*"]`
- `allow_headers=["*"]`

No Mongo/JWT/auth usage in active backend routes.
- No `backend/api/auth.py`
- No `backend/api/database.py`

---

## 5. Core Inference Flow (Current Code)

`/analyze` -> `_run_analysis(transcript)` in `backend/api/server.py`

1. Instantiate `EarningsCallAnalyzer` lazily.
2. `EarningsCallAnalyzer.analyze()`:
   - calls `TranscriptAnalyzer.analyze()` for segments+intents
   - computes timeline window signals (`TimelineSignalAnalyzer`)
   - returns `segments`, `timeline_signals`, `aggregation`, `insight`
3. `_run_analysis` recomputes main outputs from `segments` using:
   - `compute_risk_score`
   - `derive_signal`
   - `compute_confidence`
   - `detect_volatility`
   - `compute_signal_std`
   - `compute_intent_distribution`
   - `compute_signal_distribution`
   - `predict_market_outlook`
   - `generate_insight`
   - `extract_key_drivers`
4. Builds timeline points from intents.
5. Returns response JSON.

Returned keys from `/analyze` currently:
- `score`
- `signal`
- `prediction`
- `prediction_explanation`
- `confidence`
- `volatility`
- `volatility_std`
- `intent_distribution`
- `insight`
- `segments`
- `growth_drivers`
- `risk_drivers`
- `drivers`
- `timeline`
- `fallback_used`

`/upload`:
- Supports `.txt` and `.pdf`.
- Parses text, then calls same `_run_analysis`.

`/compare`:
- Runs `_run_analysis` on two transcripts.
- Returns both analyses plus `risk_delta`, `confidence_delta`, `trend`, and `signal_difference`.

---

## 6. Transcript Segmentation + Intent Path

File: `backend/financial_pragmatic_ai/analysis/transcript_analyzer.py`

Segmentation behavior:
- First tries speaker cue splitting (`CEO:`, `CFO:`, `ANALYST:`, `EXECUTIVE:`, `OPERATOR:`).
- Sentence splitting and chunking used internally (`chunk_size` dynamically 1/2/3).
- Falls back to `parse_transcript()` in `analysis/transcript_parser.py`.
- If still <=1 segment, fallback chunking on cleaned full text.
- Filters out very short segments (`<10` chars).

Runtime logging currently present:
- Uses `logging` module rather than `print()` in `TranscriptAnalyzer`.
- Logs parsed segment counts, fallback load state, fallback failures, and per-request fallback usage.

Intent smoothing exists as function `smooth_intents(...)` but is currently **disabled**:
- `# results = smooth_intents(results)` is commented.

---

## 7. Model Loading and Usage (Current)

### 7.1 FinBERT intent model wrapper
File: `backend/financial_pragmatic_ai/models/finbert_intent_model.py`

- `FinBERTIntentModel` loads from HuggingFace repo (hardcoded):
  - `HF_INTENT_REPO = "SarcoNarco/finbert_intent_v3"`
- Uses `AutoModelForSequenceClassification.from_pretrained(...)`.
- Predict returns:
  - `intent`, `logits`, `embedding` (CLS), `confidence`.
- Label map in predict:
  - `0 -> EXPANSION`
  - `1 -> COST_PRESSURE`
  - `2 -> STRATEGIC_PROBING`
  - `3 -> GENERAL_UPDATE`

Important verified behavior:
- Constructor parameter `model_dir` is accepted but not used to load local model; load path is HF repo.

### 7.2 TranscriptAnalyzer intent fallback
If FinBERT wrapper fails:
- falls back to `FinancialPragmaticTransformer` weights downloaded from HF repo:
  - repo: `SarcoNarco/financial-models`
  - file: `pragmatic_transformer_trained.pt`
- Fallback loading is production-hardened:
  - lazy-loaded only when needed
  - CPU-only load via `torch.load(..., map_location="cpu")`
  - best-effort `weights_only=True` and `mmap=True`
  - guarded by `threading.Lock`
  - timeout-protected (`20` seconds)
  - circuit breaker disables fallback for the process after first load/inference failure

### 7.3 Conversation attention model
- Architecture exists in `models/conversation_attention_model.py`.
- `TranscriptAnalyzer` attempts to load local `backend/financial_pragmatic_ai/models/conversation_attention.pt`.
- If file missing -> warning + rule-based signal fallback.

### 7.4 Local model artifacts state
Verified locally:
- No `.pt` or `.safetensors` files present under `backend/`.
- Deploy-time model weights have been refactored off GitHub and onto Hugging Face-hosted repos.
- Runtime depends on downloading model artifacts from Hugging Face instead of loading committed binaries from this repository.

### 7.5 Deployment/runtime constraints reflected in code
- Railway runtime pin: `backend/runtime.txt` -> `python-3.11.9`
- CPU-only PyTorch install is pinned in `backend/requirements.txt`:
  - `torch==2.2.2+cpu`
- Dependency set is pinned for deployment stability:
  - `transformers==4.40.2`
  - `huggingface_hub==0.23.2`
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `datasets==2.20.0`
  - `python-multipart==0.0.9`
  - `pdfplumber==0.11.1`
  - `protobuf==4.25.3`

---

## 8. Signal/Score/Prediction Logic (Current)

File: `backend/financial_pragmatic_ai/analysis/financial_signal_engine.py`

Intent-to-score mapping (`compute_risk_score`):
- `EXPANSION: +1.0`
- `COST_PRESSURE: -1.0`
- `STRATEGIC_PROBING: +0.2`
- `GENERAL_UPDATE: 0.0`

Returned score is average over segments.

Signal derivation (`derive_signal`):
- `score > 0.2 -> growth`
- `score < -0.2 -> risk`
- else `neutral`

Additional outputs:
- `compute_confidence`: dominant signal percentage (0–100)
- `compute_intent_distribution`: percentages by intent (0–100)
- `detect_volatility`: LOW/MEDIUM/HIGH via std of mapped signal values

File: `analysis/market_predictor.py`
- `predict_market_outlook(signal, risk_score, volatility, intent_distribution)` returns prediction + explanation.

File: `analysis/insight_engine.py`
- `extract_key_drivers(...)` performs sentence selection, compression, quality gate, semantic dedup.
- Outputs `growth_drivers` and `risk_drivers`.

---

## 9. Training Pipelines (Present in Repo)

### FinBERT intent training
- `backend/financial_pragmatic_ai/training/train_finbert_intent_v2.py`
  - Calls `train_finbert_intent_model(...)`.
- Main trainer implementation in `models/finbert_intent_model.py`:
  - Transcript-level deterministic split via hash (`<80` train, else eval).
  - Duplicate normalized text overlap warning across splits.
  - 4-class mapping and balancing.
  - HuggingFace `Trainer` training.

### Unified V2 pipeline
- `backend/financial_pragmatic_ai/training/train_v2_pipeline.py`
  - Trains FinBERT intent model.
  - Builds 3-step conversation sequences.
  - Streams embeddings to disk (`./embeddings/*.pt`).
  - Trains conversation attention model from embedding files.

### Other legacy trainers
- `train_pragmatic_transformer.py`
- `train_conversation_model.py`
- `train_intent_classifier.py`
- `train_pragmatic_transformwer.py` (filename typo retained in repo)

---

## 10. Datasets (Verified)

### `backend/financial_pragmatic_ai/data/pragmatic_intent_dataset_clean.csv`
- Columns: `text`, `speaker`, `intent`
- Row count: `99,926`

### `backend/financial_pragmatic_ai/data/pragmatic_intent_dataset.csv`
- Columns: `text`, `speaker`, `intent`
- Row count: `140,303`

### `backend/financial_pragmatic_ai/data/conversation_signal_dataset.csv`
- Columns: `CEO_intent`, `CFO_intent`, `Analyst_intent`, `signal`
- Row count: `64`

### `backend/financial_pragmatic_ai/data/intent_dataset.csv`
- Columns: `text`, `intent`
- Row count: `8`

---

## 11. Evaluation Pipeline and Latest Saved Results

Evaluation module:
- `backend/financial_pragmatic_ai/evaluation/better_than_fin/evaluate.py`
- Entrypoint:
  - `python -m financial_pragmatic_ai.evaluation.better_than_fin.evaluate`

Behavior:
- Loads full clean dataset.
- Builds balanced sample by signal (`per_class_target`, default `80`).
- Runs:
  - FinBERT baseline sentiment -> signal
  - Custom system pipeline -> signal
- Saves metrics/artifacts in `evaluation/better_than_fin/results/`.

Latest saved metrics from `results/metrics_summary.json`:
- Last rerun verified locally on `2026-04-12`
- Sample size: `240` (80 per class)
- FinBERT:
  - Accuracy: `0.5041666666666667`
  - Macro F1: `0.4523600209314495`
- Our system:
  - Accuracy: `0.8291666666666667`
  - Macro F1: `0.8302953273507985`
- Improvement deltas:
  - Accuracy delta: `0.32500000000000007`
  - Macro F1 delta: `0.37793530641934897`
- Prediction distributions from latest rerun:
  - FinBERT: `{'neutral': 120, 'growth': 102, 'risk': 18}`
  - Our system: `{'growth': 61, 'risk': 67, 'neutral': 112}`

Saved artifacts currently present:
- `metrics_summary.json`
- `classification_report.txt`
- `disagreement_cases.csv`
- `ours_correct_finbert_wrong.csv`
- `confusion_matrix_ours_normalized.png`
- `confusion_matrix_finbert_normalized.png`
- `model_comparison.png`
- `per_class_f1.png`
- `agreement_bar.png`
- `class_distribution.png`

---

## 12. Frontend State (Two Apps)

### 12.1 `frontend/` (CRA)
- Stateless dashboard.
- Uses backend APIs directly:
  - `/analyze`, `/upload`, `/compare`
- `src/api/client.js` sends `{ transcript }` to `/analyze`.
- Contains Analyze + Compare tabs and dashboard components.

### 12.2 `frontend_v2/` (Vite + Tailwind)
- Uses Supabase auth and Supabase table `analyses` for history/compare selection.
- `App.jsx` gates app via session (`if !session return <Auth />`).
- Analyze action calls backend `/analyze` and maps response.
- File upload in `frontend_v2` reads local `.txt` into textarea; it does **not** call backend `/upload`.
- Compare tab uses two selected history records from Supabase (not backend `/compare` endpoint).

Implication:
- Backend is stateless, but `frontend_v2` introduces external persistence/auth through Supabase.

---

## 13. Known Inconsistencies / Risks (Verified in Code)

1. **Two API server modules exist**
   - Active: `backend/api/server.py`
   - Legacy: `backend/financial_pragmatic_ai/api/server.py`
   - Can cause confusion if wrong uvicorn target is used.

2. **Score semantic mismatch in insight/prediction thresholds**
   - `compute_risk_score` returns small signed average around `[-1, 1]`.
   - `generate_insight` and parts of `predict_market_outlook` still use thresholds like `35/45/55/60/65`.
   - This creates logically inconsistent behavior.

3. **`train_v2_pipeline.py` references missing attribute**
   - Uses `finbert_wrapper.encoder_device`.
   - `FinBERTIntentModel` does not define `encoder_device`.
   - This is a likely runtime error path in V2 pipeline training.

4. **Very verbose debug logging remains active in production paths**
   - `TranscriptAnalyzer` now uses structured `logging`, but `financial_signal_engine.py` still emits direct debug prints for intents/scores/signals.

5. **No local model binaries in repo**
   - Runtime depends on HuggingFace downloads (`SarcoNarco/*` repos).

6. **README is stale vs current implementation**
   - Still describes older minimal response and only `frontend/` flow.

---

## 14. How To Run (Verified commands)

### Backend
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/backend
pip install -r requirements.txt
uvicorn api.server:app --reload
```

### Frontend (CRA)
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/frontend
npm install
npm start
```

### Frontend V2 (Vite)
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/frontend_v2
npm install
npm run dev
```

`frontend_v2` additionally requires env vars for Supabase:
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

---

## 15. Important Notes for Next Agent

- Do not assume the system is fully stateless if you are working on `frontend_v2`; it depends on Supabase.
- Use `backend/api/server.py` as the backend source of truth.
- Treat `metrics_summary.json` as the latest saved benchmark unless you rerun evaluation.
- If fixing model-performance collapse, inspect:
  - segmentation output count in `TranscriptAnalyzer`
  - intent distribution from `predict_intent`
  - score-threshold consistency across `financial_signal_engine.py` and `market_predictor.py`
- If working on V2 training, resolve `encoder_device` mismatch before execution.
