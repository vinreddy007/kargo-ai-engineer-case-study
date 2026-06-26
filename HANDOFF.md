# Kargo Media Recommender Handoff

## Current State

This repo contains a working Streamlit chatbot for the Kargo AI Engineer case study.

The app accepts a plain-text client brief, uses an OpenAI-backed parser to extract campaign requirements and planning intent, asks clarification questions when required details are missing, and then calls a deterministic recommendation engine to recommend a Kargo product or small bundle.

The app is intended to run locally with:

```bash
python3 -m streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

## Environment

Python 3.11+ is required.

Install dependencies:

```bash
python3 -m pip install -e '.[dev]'
```

The app needs an OpenAI API key for the LLM-backed parser. The local secrets file is:

```text
.streamlit/secrets.toml
```

It should contain:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

Do not commit `.streamlit/secrets.toml`; it is ignored by `.gitignore`.

## Architecture

```text
Streamlit app.py
  -> RecommendationWorkflow
    -> OpenAIRequirementParser
    -> clarification routing
    -> RecommendationEngine
      -> CSV data loaders
      -> benchmark calculations
      -> inventory checks
      -> ranking / bundle logic
```

LLM responsibilities:

- Parse messy client briefs into structured `ClientRequirements`.
- Infer whether the user wants one clear product, allows a bundle, or wants maximum budget delivery.
- Merge follow-up answers with prior requirements.

Deterministic Python responsibilities:

- CTR and in-view benchmark calculations.
- Budget-to-impressions math.
- Inventory confidence handling.
- Product budget-fit and inventory caveat checks.
- Product ranking.
- Bundle planning by default when a bundle improves KPI performance and budget delivery.
- Rejected-alternative explanations.

## Key Files

- `app.py`: Streamlit chatbot UI.
- `src/kargo_media_recommender/workflow.py`: LangGraph workflow for parse, clarify, recommend, format.
- `src/kargo_media_recommender/llm_parser.py`: OpenAI structured-output parser.
- `src/kargo_media_recommender/recommender.py`: deterministic recommendation engine.
- `src/kargo_media_recommender/benchmarks.py`: CTR, in-view rate, and CPM/impression math.
- `src/kargo_media_recommender/data.py`: CSV loaders.
- `src/kargo_media_recommender/schemas.py`: dataclass models used across the app.
- `src/kargo_media_recommender/ui_helpers.py`: UI formatting helpers.
- `client_briefs.json`: sample briefs shown in the sidebar.
- `product_catalog.csv`: product names/descriptions/CPMs.
- `campaign_history.csv`: historical line-item performance data.
- `inventory_forecaster.csv`: available impressions and confidence scores.
- `README.md`: setup, run, test, and architecture notes.

## Recommendation Logic

Required fields before recommendation:

- `vertical`: Retail, Finance, Travel, QSR, or Entertainment
- `primary_kpi`: click-through rate or in-view rate
- `geo`: US, EMEA, or APAC
- `budget`: dollar amount

Optional fields:

- advertiser/client name
- impression goal
- recommendation style: `single_product_preferred`, `bundle_allowed`, or `maximize_budget_delivery`

Core formulas:

```text
CTR = clicks / impressions
In-view rate = viewable_impressions / impressions
Estimated impressions = budget / CPM * 1000
Usable inventory = available_imps * inventory_confidence
```

The engine defaults to bundle-capable planning, so it can recommend a small bundle when that improves KPI performance and budget delivery. It pivots to one product only when the brief asks for a simple, straightforward, single-product, or otherwise clearly one-product recommendation. If that selected product cannot absorb the full budget, it is still recommended with a budget-fit caveat. `no_viable_option` is used only when the available data cannot support a recommendation.

## Tests

Run:

```bash
python3 -m pytest -q
```

Last verified result:

```text
24 passed, 1 warning
```

The warning comes from LangChain/Pydantic under Python 3.14 and has not blocked the app.

Tests do not call the OpenAI API. Parser/workflow tests use fakes.

## Manual UI Checks Performed

Verified in browser automation:

- App loads at `http://localhost:8501`.
- Sample 1 returns a `Commerce Connect` + `Attention Builder` bundle.
- Sample 3 returns `Attention Builder` with a budget-fit caveat.
- Sample 7 asks the KPI clarification.
- Answering `click-through rate` after Sample 7 may return a product or bundle based on the inferred recommendation style.
- No `could not process` errors appeared during those checks.

## Current UI Behavior

- Sidebar shows sample briefs.
- Clicking a sample starts a fresh conversation automatically.
- User messages render as right-aligned chat bubbles.
- Submitted prompts are enqueued so the user bubble renders before recommendation processing starts.
- The chat accepts clarification answers only until a final recommendation is produced; after that the input is replaced by a start-over action.
- Assistant responses include recommendation details inside the chat turn.
- Current parsed brief appears in the sidebar.
- Rejected alternatives are expandable.
- A collapsed decision summary shows each evaluated product's final decision state.

## Known Notes

- The LLM parser defaults to `gpt-5.5` with low reasoning effort because it only extracts structured fields.
- The recommendation engine is deterministic and should remain the source of truth for math/ranking.
- `.streamlit/secrets.toml` should stay local only.
- The repo may not be initialized as a git repository in this local workspace.

## Good Next Steps

- Add a slide deck outline or `presentation.md`.
- Add a lightweight architecture diagram.
- Add a few screenshots to the README if needed.
- Consider adding an integration test with a mocked full Streamlit session if the UI grows.
- Consider adding a configurable model name via environment variable for easier demos.
