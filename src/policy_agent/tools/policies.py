"""search_policies and get_policy, backed by the ingested corpus and its front-matter."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from policy_agent.models import Chunk, PolicyDoc
from policy_agent.tools.registry import Tool, ToolContext, ToolOutput, error_output


def render_chunk_text(chunk: Chunk, ctx: ToolContext) -> str:
    """Hook for the injection guardrail, which wraps and sanitises retrieved text (see guardrails)."""
    return chunk.text


class SearchPoliciesArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Search query in plain language.")
    as_of: date | None = Field(
        default=None,
        description="Date the policies must be in force on (YYYY-MM-DD). Defaults to the request date.",
    )


def search_policies(args: SearchPoliciesArgs, ctx: ToolContext) -> ToolOutput:
    as_of = args.as_of or ctx.as_of
    res = ctx.retriever.search(args.query, ctx.role, as_of)
    results: list[dict[str, Any]] = [
        {
            "chunk_id": r.chunk.chunk_id,
            "policy_id": r.chunk.policy_id,
            "title": r.chunk.title,
            "version": r.chunk.version,
            "effective_date": r.chunk.effective_date.isoformat(),
            "section": r.chunk.section,
            "score": round(r.score, 4),
            "text": render_chunk_text(r.chunk, ctx),
        }
        for r in res.results
    ]
    return ToolOutput(
        content={
            "as_of": as_of.isoformat(),
            "evidence_sufficient": res.evidence_sufficient,
            "access_restricted": res.access_restricted,
            "results": results,
        },
        evidence_ids=[r.chunk.chunk_id for r in res.results],
    )


class GetPolicyArgs(BaseModel):
    policy_id: str = Field(min_length=3, max_length=40, description="Policy id, e.g. POL-ONB-021.")
    as_of: date | None = Field(
        default=None, description="Return the version in force on this date (YYYY-MM-DD)."
    )


def _version_in_force(versions: list[PolicyDoc], as_of: date) -> PolicyDoc | None:
    return next((v for v in versions if v.in_force(as_of)), None)


def get_policy(args: GetPolicyArgs, ctx: ToolContext) -> ToolOutput:
    as_of = args.as_of or ctx.as_of
    versions = ctx.retriever.policy_versions(args.policy_id.upper())
    if not versions:
        return error_output(f"policy {args.policy_id} not found")
    if not versions[0].visible_to(ctx.role):
        return error_output(
            f"policy {args.policy_id} is restricted for role {ctx.role}; escalate to Compliance"
        )
    doc = _version_in_force(versions, as_of)
    if doc is None:
        return error_output(f"no version of {args.policy_id} was in force on {as_of.isoformat()}")
    sections = [c for c in ctx.retriever.chunks if c.policy_id == doc.policy_id and c.version == doc.version]
    return ToolOutput(
        content={
            "policy_id": doc.policy_id,
            "title": doc.title,
            "version": doc.version,
            "effective_date": doc.effective_date.isoformat(),
            "superseded_on": doc.superseded_on.isoformat() if doc.superseded_on else None,
            "owner": doc.owner,
            "summary": doc.summary,
            "versions": [
                {"version": v.version, "effective_date": v.effective_date.isoformat()} for v in versions
            ],
            "sections": [
                {"chunk_id": c.chunk_id, "section": c.section, "text": render_chunk_text(c, ctx)}
                for c in sections
            ],
        },
        evidence_ids=[c.chunk_id for c in sections],
    )


SEARCH_POLICIES = Tool(
    name="search_policies",
    description=(
        "Search the internal policy library (hybrid keyword + semantic search, reranked). Only policies the "
        "user may see and that are in force on as_of are searched. Returns the top passages with chunk_ids "
        "to cite, evidence_sufficient (false means nothing relevant was found) and access_restricted (true "
        "means the best match is in a policy this user cannot see)."
    ),
    args_model=SearchPoliciesArgs,
    fn=search_policies,
)

GET_POLICY = Tool(
    name="get_policy",
    description=(
        "Fetch one policy by id: the version in force on as_of, its metadata (owner, effective date, "
        "version history) and every section with its chunk_id."
    ),
    args_model=GetPolicyArgs,
    fn=get_policy,
)
