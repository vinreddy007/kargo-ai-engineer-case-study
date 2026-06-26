from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

from kargo_media_recommender import RecommendationWorkflow
from kargo_media_recommender.ui_helpers import (
    assistant_chat_summary,
    campaign_summary_items,
    decision_summary_rows,
    escape_markdown_dollars,
    load_sample_briefs,
    recommendation_title,
    recommended_product_rows,
    rejected_product_rows,
    requirements_rows,
    sample_brief_label,
)


ROOT = Path(__file__).resolve().parent
FINAL_RECOMMENDATION_STATUSES = {
    "single_product",
    "single_product_budget_caveat",
    "bundle",
    "no_viable_option",
}


def main() -> None:
    st.set_page_config(
        page_title="Kargo Media Recommender",
        page_icon="K",
        layout="centered",
    )

    _init_session_state()
    _configure_api_key_from_secrets()
    _inject_chat_styles()

    with st.sidebar:
        st.header("Samples")
        _render_sample_briefs()
        st.divider()
        _render_current_state()
        st.divider()
        if st.button("Reset conversation", width="stretch"):
            _reset_conversation()
            st.rerun()

    st.title("Kargo Media Recommender")
    st.caption("Paste a client brief, answer any clarification questions, and review the recommendation.")

    if not _has_openai_key():
        st.warning(
            "Set `OPENAI_API_KEY` in your shell or Streamlit secrets before sending a brief."
        )

    for message in st.session_state.messages:
        if message["role"] == "user":
            _render_user_message(message["content"])
            continue

        with st.chat_message("assistant"):
            st.markdown(escape_markdown_dollars(message["content"]))
            if recommendation := message.get("recommendation"):
                _render_recommendation_details(recommendation)

    if st.session_state.pending_prompt:
        with st.chat_message("assistant"):
            with st.spinner("Working on the recommendation..."):
                _process_pending_prompt()
        st.rerun()

    if _conversation_complete():
        _render_completion_actions()
        return

    if prompt := st.chat_input(
        _chat_input_placeholder(),
        disabled=bool(st.session_state.pending_prompt),
    ):
        _enqueue_user_message(prompt)
        st.rerun()


def _init_session_state() -> None:
    defaults = {
        "messages": [],
        "current_requirements": None,
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _configure_api_key_from_secrets() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_resource(show_spinner=False)
def _get_workflow(data_dir: str) -> RecommendationWorkflow:
    return RecommendationWorkflow.from_data_dir(Path(data_dir))


def _enqueue_user_message(prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_prompt = prompt


def _process_pending_prompt() -> None:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if not prompt:
        return

    if not _has_openai_key():
        response_text = (
            "I need `OPENAI_API_KEY` configured before I can parse briefs with the LLM."
        )
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        return

    try:
        workflow = _get_workflow(str(ROOT))
        output = workflow.run(
            prompt,
            current_requirements=st.session_state.current_requirements,
        )
    except Exception as exc:
        response_text = f"I could not process that brief: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        return

    st.session_state.current_requirements = output.requirements
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_chat_summary(output.recommendation, output.response_text),
            "recommendation": output.recommendation,
        }
    )


def _conversation_complete() -> bool:
    for message in reversed(st.session_state.messages):
        recommendation = message.get("recommendation")
        if recommendation is None:
            continue
        return recommendation.status in FINAL_RECOMMENDATION_STATUSES
    return False


def _needs_clarification_answer() -> bool:
    requirements = st.session_state.current_requirements
    return requirements is not None and bool(requirements.missing_required_fields)


def _chat_input_placeholder() -> str:
    if st.session_state.pending_prompt:
        return "Working on the recommendation..."
    if _needs_clarification_answer():
        return "Answer the clarification question"
    return "Paste a client brief"


def _render_sample_briefs() -> None:
    try:
        briefs = load_sample_briefs(ROOT / "client_briefs.json")
    except Exception as exc:
        st.error(f"Could not load sample briefs: {exc}")
        return

    for index, brief in enumerate(briefs, start=1):
        label = sample_brief_label(index, brief)
        if st.button(label, key=f"sample_{index}", width="stretch"):
            _reset_conversation()
            _enqueue_user_message(brief)
            st.rerun()


def _render_current_state() -> None:
    st.header("Current Brief")
    rows = requirements_rows(st.session_state.current_requirements)
    if not rows:
        st.caption("No parsed requirements yet.")
        return
    _render_table(rows)


def _render_recommendation_details(result) -> None:
    if result is None:
        return

    st.markdown("**Campaign setup**")
    _render_campaign_summary(campaign_summary_items(result.requirements))

    title = recommendation_title(result)
    if result.recommended_products:
        st.markdown(f"**{title}**")
        _render_recommended_products(
            recommended_product_rows(
                result.recommended_products,
                result.requirements.primary_kpi,
            )
        )
    elif title:
        st.markdown(f"**{title}**")

    st.markdown("**Why this recommendation**")
    st.markdown(escape_markdown_dollars(result.rationale))

    if result.tradeoffs:
        st.markdown("**Planning notes**")
        _render_planning_notes(result.tradeoffs)

    if result.rejected_alternatives:
        with st.expander("Rejected alternatives", expanded=True):
            _render_rejected_alternatives(
                rejected_product_rows(
                    result.rejected_alternatives,
                    result.requirements.primary_kpi,
                    result.recommended_products,
                )
            )

    decision_rows = decision_summary_rows(result)
    if decision_rows:
        with st.expander("Decision summary"):
            _render_decision_summary_table(decision_rows)


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.current_requirements = None
    st.session_state.pending_prompt = None


def _render_completion_actions() -> None:
    st.markdown(
        """
        <div class="completion-panel">
            <strong>Recommendation complete</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start new recommendation", type="primary", width="stretch"):
        _reset_conversation()
        st.rerun()


def _render_campaign_summary(items: list[dict[str, str]]) -> None:
    if not items:
        return

    summary_items = "".join(
        '<div class="campaign-summary__item">'
        f'<span>{_escape_html(item["Label"])}</span>'
        f'<strong>{_escape_html(item["Value"])}</strong>'
        "</div>"
        for item in items
    )
    st.markdown(
        f'<div class="campaign-summary">{summary_items}</div>',
        unsafe_allow_html=True,
    )


def _render_planning_notes(notes: tuple[str, ...]) -> None:
    if not notes:
        return

    note_items = "".join(f"<li>{_escape_html(note)}</li>" for note in notes)
    st.markdown(
        f'<ul class="planning-notes">{note_items}</ul>',
        unsafe_allow_html=True,
    )


def _render_rejected_alternatives(rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        tone = _safe_tone(row.get("Tone"))
        cards.append(
            f"""
            <div class="choice-card choice-card--{tone}">
                <div class="choice-card__header">
                    <strong>{_escape_html(row["Product"])}</strong>
                    {_status_pill(row.get("Badge", "Not selected"), tone)}
                </div>
                <div class="choice-metrics">
                    Forecasted imps: {_escape_html(row["Forecasted Imps"])}
                    <span>|</span>
                    {_escape_html(row["KPI Label"])}: {_escape_html(row["KPI Value"])}
                </div>
                <div class="choice-reason">{_escape_html(row["Reason"])}</div>
            </div>
            """
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_recommended_products(rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        tone = _safe_tone(row.get("Tone"))
        metrics = [
            f"Budget: {row['Budget']}",
            f"CPM: {row['CPM']}",
            f"Forecasted imps: {row['Forecasted Imps']}",
            f"{row['KPI Label']}: {row['KPI Value']}",
            f"Inventory confidence: {row['Inventory Confidence']}",
            f"Usable inventory: {row['Usable Inventory']}",
        ]
        metrics_html = " <span>|</span> ".join(_escape_html(metric) for metric in metrics)
        cards.append(
            f"""
            <div class="choice-card choice-card--{tone}">
                <div class="choice-card__header">
                    <strong>{_escape_html(row["Product"])}</strong>
                    {_status_pill(row.get("Badge", "Selected"), tone)}
                </div>
                <div class="choice-metrics">{metrics_html}</div>
            </div>
            """
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_decision_summary_table(rows: list[dict[str, str]]) -> None:
    visible_columns = [column for column in rows[0] if column not in {"Badge", "Tone"}]
    header_html = "".join(f"<th>{_escape_html(column)}</th>" for column in visible_columns)
    body_rows = []
    for row in rows:
        tone = _safe_tone(row.get("Tone"))
        cells = []
        for column in visible_columns:
            value = _escape_html(row[column])
            if column == "Decision":
                value = _status_pill(row[column], tone)
            cells.append(f"<td>{value}</td>")
        body_rows.append(f'<tr class="decision-row decision-row--{tone}">{"".join(cells)}</tr>')

    st.markdown(
        f"""
        <div class="decision-table-wrap">
            <table class="decision-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{"".join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_table(rows: list[dict[str, str]]) -> None:
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_bottom_spacer() -> None:
    st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)


def _render_user_message(content: str) -> None:
    st.markdown(
        f"""
        <div class="user-message-row">
            <div class="user-message-bubble">{html.escape(content)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escape_html(value: object) -> str:
    return html.escape(str(value))


def _safe_tone(tone: object) -> str:
    if tone in {"selected", "alternative", "blocked"}:
        return str(tone)
    return "blocked"


def _status_pill(label: object, tone: str) -> str:
    return (
        f'<span class="status-pill status-pill--{tone}">'
        f"{_escape_html(label)}"
        "</span>"
    )


def _inject_chat_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            margin: 1rem 0 1.25rem;
            background: #ffffff;
        }

        [data-testid="stChatMessage"] p {
            line-height: 1.55;
            margin: 0.1rem 0 0.45rem;
        }

        [data-testid="stChatMessage"] p:last-child {
            margin-bottom: 0;
        }

        [data-testid="stChatMessage"] .stMarkdown {
            margin-bottom: 0.4rem;
        }

        .user-message-row {
            display: flex;
            justify-content: flex-end;
            margin: 0.9rem 0 1.25rem;
        }

        .user-message-bubble {
            max-width: 82%;
            padding: 0.85rem 1.05rem;
            border-radius: 16px 16px 4px 16px;
            background: #eef4ff;
            border: 1px solid rgba(37, 99, 235, 0.16);
            color: #1f2937;
            line-height: 1.55;
        }

        [data-testid="stExpander"] {
            border-radius: 10px;
            margin-top: 0.75rem;
            margin-bottom: 0.9rem;
        }

        .completion-panel {
            border: 1px solid rgba(21, 128, 61, 0.22);
            border-left: 4px solid #15803d;
            border-radius: 8px;
            background: #f0fdf4;
            color: #166534;
            margin: 1.1rem 0 0.75rem;
            padding: 0.8rem 0.9rem;
        }

        .campaign-summary {
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.55rem 0 1rem;
        }

        .campaign-summary__item {
            border-left: 3px solid #94a3b8;
            background: #f8fafc;
            padding: 0.55rem 0.65rem;
        }

        .campaign-summary__item span {
            color: #64748b;
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.22rem;
            text-transform: uppercase;
        }

        .campaign-summary__item strong {
            color: #111827;
            display: block;
            font-size: 0.9rem;
            line-height: 1.3;
        }

        .planning-notes {
            margin: 0.15rem 0 0.9rem 1.1rem;
            padding-left: 0.25rem;
        }

        .planning-notes li {
            color: #1f2937;
            line-height: 1.55;
            margin: 0.25rem 0;
        }

        .choice-card {
            border: 1px solid rgba(17, 24, 39, 0.10);
            border-left-width: 4px;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin: 0.7rem 0;
            background: #ffffff;
        }

        .choice-card--selected {
            border-left-color: #15803d;
            background: #f0fdf4;
        }

        .choice-card--alternative {
            border-left-color: #b45309;
            background: #fffbeb;
        }

        .choice-card--blocked {
            border-left-color: #be123c;
            background: #fff1f2;
        }

        .choice-card__header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            color: #111827;
        }

        .choice-metrics {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.65;
            margin-top: 0.45rem;
        }

        .choice-metrics span {
            color: #d1d5db;
        }

        .choice-reason {
            color: #1f2937;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 0.55rem;
        }

        .status-pill {
            border-radius: 999px;
            border: 1px solid transparent;
            display: inline-flex;
            align-items: center;
            flex: 0 0 auto;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1.15;
            padding: 0.26rem 0.48rem;
            white-space: normal;
        }

        .status-pill--selected {
            background: #dcfce7;
            border-color: #86efac;
            color: #166534;
        }

        .status-pill--alternative {
            background: #fef3c7;
            border-color: #fcd34d;
            color: #92400e;
        }

        .status-pill--blocked {
            background: #ffe4e6;
            border-color: #fda4af;
            color: #9f1239;
        }

        .decision-table-wrap {
            border: 1px solid rgba(17, 24, 39, 0.10);
            border-radius: 8px;
            margin-top: 0.5rem;
            overflow: hidden;
            width: 100%;
        }

        .decision-table {
            border-collapse: collapse;
            font-size: 0.74rem;
            table-layout: fixed;
            width: 100%;
        }

        .decision-table th:nth-child(1),
        .decision-table td:nth-child(1) {
            width: 18%;
        }

        .decision-table th:nth-child(2),
        .decision-table td:nth-child(2) {
            width: 19%;
        }

        .decision-table th:nth-child(3),
        .decision-table td:nth-child(3) {
            width: 15%;
        }

        .decision-table th:nth-child(4),
        .decision-table td:nth-child(4) {
            width: 8%;
        }

        .decision-table th:nth-child(5),
        .decision-table td:nth-child(5) {
            width: 40%;
        }

        .decision-table th {
            background: #f9fafb;
            color: #6b7280;
            font-weight: 700;
            padding: 0.5rem 0.5rem;
            text-align: left;
            overflow-wrap: anywhere;
        }

        .decision-table td {
            border-top: 1px solid rgba(17, 24, 39, 0.08);
            color: #1f2937;
            line-height: 1.35;
            overflow-wrap: anywhere;
            padding: 0.52rem 0.5rem;
            vertical-align: top;
            white-space: normal;
            word-break: normal;
        }

        .decision-table .status-pill {
            max-width: 100%;
        }

        .decision-row--selected td:first-child {
            border-left: 4px solid #15803d;
        }

        .decision-row--alternative td:first-child {
            border-left: 4px solid #b45309;
        }

        .decision-row--blocked td:first-child {
            border-left: 4px solid #be123c;
        }

        .decision-row--selected {
            background: #f7fef9;
        }

        .decision-row--alternative {
            background: #fffdf4;
        }

        .decision-row--blocked {
            background: #fff7f8;
        }

        @media (max-width: 760px) {
            .campaign-summary {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 520px) {
            .campaign-summary {
                grid-template-columns: 1fr;
            }
        }

        h1 {
            font-size: 2rem !important;
            letter-spacing: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
