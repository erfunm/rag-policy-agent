"""get_case_status: backed by a JSON file standing in for the case-management system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from policy_agent.tools.registry import Tool, ToolContext, ToolOutput, error_output


class CaseStore:
    def __init__(self, cases: dict[str, dict[str, Any]]) -> None:
        self._cases = {k.upper(): v for k, v in cases.items()}

    @classmethod
    def from_file(cls, path: Path) -> CaseStore:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def get(self, case_id: str) -> dict[str, Any] | None:
        return self._cases.get(case_id.upper())


class GetCaseStatusArgs(BaseModel):
    case_id: str = Field(description="Case reference, e.g. REQ-1001.", pattern=r"^[A-Za-z]{3}-\d{3,6}$")


def case_evidence_id(case_id: str) -> str:
    return f"CASE:{case_id.upper()}"


def get_case_status(args: GetCaseStatusArgs, ctx: ToolContext) -> ToolOutput:
    record = ctx.cases.get(args.case_id)
    if record is None:
        return error_output(f"case {args.case_id.upper()} not found")
    eid = case_evidence_id(args.case_id)
    return ToolOutput(
        content={"evidence_id": eid, "case_id": args.case_id.upper(), **record}, evidence_ids=[eid]
    )


GET_CASE_STATUS = Tool(
    name="get_case_status",
    description=(
        "Look up a case in the case-management system: status, type, risk flag, assigned team and the "
        "related policy. Returns an error if the case does not exist. Cite the returned evidence_id."
    ),
    args_model=GetCaseStatusArgs,
    fn=get_case_status,
)
