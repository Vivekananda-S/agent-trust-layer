"""Streamlit app for hand-labelling the seed queue. Launch with `atl-label app --config ...`.

Only train traces can appear here: the queue is built from the train split and re-checked
against it on load (atl.labels.sample.queued_traces). Environment outcome, injected fault and
prompt variant stay hidden until a trace is labelled, so they cannot anchor the judgement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from atl.labels.render import policy_text, trace_views
from atl.labels.sample import SeedConfig, queued_traces
from atl.labels.schema import Label, append_label, load_labels
from atl.labels.taxonomy import CLASSES, TAXONOMY_VERSION

ANNOTATOR = "human"
VERDICTS = {"No failure": False, "Failure": True, "Can't judge": None}
CONFIDENCES = ["low", "medium", "high"]
LONG_RESULT = 1500  # characters shown before a tool result is folded


def _config_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/labels/seed_v1.yaml")
    args, _ = parser.parse_known_args()
    return args.config


@st.cache_resource
def _load(config_path: str):  # noqa: ANN202 - cached by Streamlit, private
    cfg = SeedConfig.from_yaml(config_path)
    return queued_traces(cfg)


@st.cache_resource
def _policy(domain: str) -> str:
    return policy_text(domain)


def _go(pos: int) -> None:
    st.session_state.pos = pos


def _sidebar(items: list, labels: dict[str, Label]) -> int:
    done = sum(1 for item, _ in items if item.trace_id in labels)
    st.sidebar.progress(done / len(items), text=f"{done} of {len(items)} labelled")
    if "pos" not in st.session_state:
        first_open = next((i for i, (it, _) in enumerate(items) if it.trace_id not in labels), 0)
        st.session_state.pos = first_open
    pos = st.session_state.pos
    left, right = st.sidebar.columns(2)
    left.button("Previous", on_click=_go, args=(max(pos - 1, 0),), use_container_width=True)
    right.button(
        "Next", on_click=_go, args=(min(pos + 1, len(items) - 1),), use_container_width=True
    )
    next_open = next(
        (i for i, (it, _) in enumerate(items) if i > pos and it.trace_id not in labels), None
    )
    if next_open is not None:
        st.sidebar.button("Next unlabelled", on_click=_go, args=(next_open,), type="primary")
    jump = st.sidebar.number_input("Go to position", 0, len(items) - 1, pos, step=1)
    if jump != pos:
        _go(int(jump))
        st.rerun()
    st.sidebar.markdown(f"**Taxonomy v{TAXONOMY_VERSION}** (docs/labelling_guide.md)")
    for code, (name, definition) in CLASSES.items():
        st.sidebar.markdown(f"**{code} {name}**: {definition}")
    return st.session_state.pos


def _show_trace(trace) -> None:  # noqa: ANN001
    st.markdown(f"**Opening request:** {trace.task}")
    for v in trace_views(trace):
        head = f"`#{v.index}` **{v.title}**"
        if v.kind in ("user", "assistant"):
            with st.chat_message(v.kind):
                st.markdown(head)
                st.text(v.body)
        elif v.kind == "thought":
            with st.expander(f"#{v.index} thought"):
                st.text(v.body)
        else:
            st.markdown(head + (" :red[(tool error)]" if v.is_error else ""))
            lang = "json" if v.kind == "tool_call" else None
            st.code(v.body[:LONG_RESULT], language=lang, wrap_lines=True)
            if len(v.body) > LONG_RESULT:
                with st.expander("full result"):
                    st.code(v.body, wrap_lines=True)


def _label_form(trace, existing: Label | None, labels_path: Path) -> None:  # noqa: ANN001
    verdict_names = list(VERDICTS)
    default_verdict = (
        next(k for k, v in VERDICTS.items() if v == existing.failed) if existing else "No failure"
    )
    with st.form(f"label-{trace.trace_id}"):
        verdict = st.radio(
            "Verdict",
            verdict_names,
            index=verdict_names.index(default_verdict),
            horizontal=True,
        )
        classes = st.multiselect(
            "Failure classes",
            list(CLASSES),
            default=existing.classes if existing else [],
            format_func=lambda c: f"{c} {CLASSES[c][0]}",
        )
        step = st.number_input(
            "First failure step (#)",
            0,
            len(trace.steps) - 1,
            existing.first_failure_step or 0 if existing else 0,
        )
        other = st.text_input(
            "Other failure (fits no class)", existing.other_failure or "" if existing else ""
        )
        confidence = st.radio(
            "Confidence",
            CONFIDENCES,
            index=CONFIDENCES.index(existing.confidence) if existing else 2,
            horizontal=True,
        )
        rationale = st.text_area(
            "Rationale (one or two sentences)", existing.rationale if existing else ""
        )
        if not st.form_submit_button("Save label", type="primary"):
            return
    failed = VERDICTS[verdict]
    try:
        label = Label(
            trace_id=trace.trace_id,
            failed=failed,
            classes=classes if failed else [],
            first_failure_step=int(step) if failed else None,
            other_failure=(other or None) if failed else None,
            confidence=confidence,
            rationale=rationale,
            annotator=ANNOTATOR,
            taxonomy_version=TAXONOMY_VERSION,
        )
    except ValidationError as e:
        for err in e.errors():
            st.error(err["msg"])
        return
    append_label(label, labels_path)
    st.rerun()


def _reveal(trace) -> None:  # noqa: ANN001
    outcome = trace.env_outcome.task_success if trace.env_outcome else None
    st.info(
        f"**Revealed after labelling.** Env task success: `{outcome}` · injected fault: "
        f"`{trace.meta.fault}` · prompt variant: `{trace.prompt_variant}` · user variant: "
        f"`{trace.meta.user_variant}` · agent: `{trace.agent_model}`"
    )


def main() -> None:
    """Render the app for the queue position held in session state."""
    st.set_page_config(page_title="ATL labelling", layout="wide")
    config_path = _config_path()
    cfg = SeedConfig.from_yaml(config_path)
    items = _load(config_path)
    labels = load_labels(cfg.labels_path)
    pos = _sidebar(items, labels)
    item, trace = items[pos]
    existing = labels.get(trace.trace_id)

    st.subheader(f"Position {pos} · {trace.meta.domain} · {len(trace.steps)} steps")
    st.caption(f"{trace.trace_id} · task {trace.task_id}")
    left, right = st.columns([3, 2], gap="large")
    with left:
        _show_trace(trace)
    with right:
        with st.expander(f"{trace.meta.domain} policy"):
            st.markdown(_policy(trace.meta.domain))
        if existing:
            st.success(
                f"Labelled ({existing.created_at:%Y-%m-%d %H:%M} UTC). Saving again replaces it."
            )
            _reveal(trace)
        _label_form(trace, existing, cfg.labels_path)


main()
