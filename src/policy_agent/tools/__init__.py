"""Tools the agent can call. Specs are generated from pydantic argument models."""

from __future__ import annotations

from policy_agent.tools.cases import GET_CASE_STATUS
from policy_agent.tools.policies import GET_POLICY, SEARCH_POLICIES
from policy_agent.tools.registry import Tool, ToolContext, ToolOutput, ToolRegistry

__all__ = ["Tool", "ToolContext", "ToolOutput", "ToolRegistry", "default_registry"]


def default_registry() -> ToolRegistry:
    return ToolRegistry([SEARCH_POLICIES, GET_POLICY, GET_CASE_STATUS])
