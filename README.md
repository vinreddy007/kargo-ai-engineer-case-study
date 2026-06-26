# Kargo Media Recommendation Assistant

Chatbot-style recommendation assistant for the Kargo AI Engineer case study.

The app accepts a plain-text client brief, extracts campaign requirements and planning intent with an OpenAI-backed parser, asks clarifying questions when required fields are missing, and uses deterministic Python logic to recommend a Kargo product or small bundle.

## Architecture

```text
Streamlit chat UI
  -> LangGraph recommendation workflow
  -> OpenAI structured-output parser
  -> deterministic recommendation engine
  -> CSV data files
```

The LLM is used for language understanding only, including whether the brief implies one clear product or a bundle/full-budget plan. Product benchmarks, CPM math, inventory checks, ranking, and rejection logic are deterministic and covered by tests.

## Setup

Use Python 3.11 or newer.

```bash
python3 -m pip install -e '.[dev]'
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key'
```

If you use Streamlit secrets instead, create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-api-key"
```

## Run The App

```bash
python3 -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The sidebar includes sample briefs from `client_briefs.json`. Sample buttons start a fresh conversation automatically. After a recommendation is produced, use **Start new recommendation** or **Reset conversation** to begin another brief.

## Run Tests

```bash
python3 -m pytest -q
```

The tests do not call the OpenAI API. LLM parser and workflow tests use fake clients/parsers.

## Recommendation Logic

Required fields before recommendation:

- Vertical: `Retail`, `Finance`, `Travel`, `QSR`, or `Entertainment`
- Primary KPI: click-through rate or in-view rate
- Geo: `US`, `EMEA`, or `APAC`
- Budget in dollars

Optional fields:

- Advertiser name
- Impression goal
- Recommendation style, inferred from the brief:
  `single_product_preferred`, `bundle_allowed`, or `maximize_budget_delivery`

Metrics:

```text
CTR = clicks / impressions
In-view rate = viewable_impressions / impressions
Estimated impressions = budget / CPM * 1000
Usable inventory = available_imps * inventory_confidence
```

The engine defaults to bundle-capable planning, so it can recommend a small bundle when that improves KPI performance and budget delivery. It pivots to one product only when the brief asks for a simple, straightforward, single-product, or otherwise clearly one-product recommendation. If that selected product cannot absorb the full budget, the app still recommends it and explains the budget-fit caveat. `no_viable_option` is used only when the available data cannot support a recommendation.
