"""Small demo UI: answer, status badge, clickable citations and the tool-call trace.

Run: ``make ui`` (uv run --extra ui streamlit run ui/app.py). Uses the configured LLM provider;
FakeLLM by default, so it works with no API key.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from policy_agent.agent.loop import PolicyAgent, build_agent
from policy_agent.config import get_settings
from policy_agent.models import ROLES, AskRequest

EXAMPLES = [
    "What proof of address is required for a customer sending $6,000 a month?",
    "What is the status of case REQ-1001?",
    "How is a suspicious activity report submitted to the Financial Intelligence Unit?",
    "What do the imported template notes in the communications style guide say?",
    "What is our parental leave policy?",
]
BADGE = {"answered": ":green-background[ANSWERED]", "escalated": ":orange-background[ESCALATED]"}


@st.cache_resource
def agent() -> PolicyAgent:
    return build_agent(get_settings())


st.set_page_config(page_title="Policy Agent", layout="wide")
st.title("Policy Agent")
st.caption("Fernleaf Payments is fictional. Answers are grounded in the synthetic policy library.")

with st.sidebar:
    role = st.selectbox("Role (X-User-Role)", ROLES, index=0)
    as_of = st.date_input("Policies in force on", value=date.today())
    a = agent()
    st.markdown(f"**LLM:** `{a.llm.provider}` / `{a.llm.model}`")

example = st.selectbox("Example questions", ["", *EXAMPLES])
question = st.text_area("Question", value=example, height=90)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Thinking..."):
        resp = a.ask(AskRequest(question=question, role=role, as_of=as_of))
    st.markdown(f"### {BADGE[resp.status]}  confidence: `{resp.confidence}`")
    if resp.reason:
        st.caption(f"Reason: {resp.reason}")
    st.write(resp.answer)

    chunks = {c.chunk_id: c for c in a.retriever.chunks}
    if resp.citations:
        st.subheader("Citations")
        for cid in resp.citations:
            chunk = chunks.get(cid)
            with st.expander(cid if chunk is None else f"{cid} - {chunk.title} / {chunk.section}"):
                st.write(chunk.text if chunk else "Case record from the case-management system.")

    st.subheader("Tool-call trace")
    st.dataframe(
        [
            {
                "tool": t.name,
                "arguments": t.arguments,
                "error": t.is_error,
                "ms": round(t.duration_ms, 1),
                "evidence": ", ".join(t.evidence_ids),
            }
            for t in resp.tool_calls
        ],
        use_container_width=True,
    )
    cols = st.columns(5)
    cols[0].metric("Steps", resp.steps)
    cols[1].metric("Latency", f"{resp.latency_ms:.0f} ms")
    cols[2].metric("Tokens in/out", f"{resp.usage.input_tokens}/{resp.usage.output_tokens}")
    cols[3].metric("Cost", f"${resp.cost_usd:.4f}")
    cols[4].metric("Flags", len(resp.flags))
    if resp.flags:
        st.caption("Flags: " + ", ".join(resp.flags))
    st.caption(f"request_id {resp.request_id} - prompt {resp.prompt_version}")
