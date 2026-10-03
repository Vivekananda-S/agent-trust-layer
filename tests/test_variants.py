"""Tests for agent prompt variants, user variants and per-run condition sampling."""

import importlib.util
from collections import Counter

import pytest
from pydantic import BaseModel

from atl.agent.run_tau2 import DEFAULT_TAU2_CHECKOUT, AgentRunConfig, choose_conditions
from atl.agent.variants import (
    AGENT_NAMES,
    NO_POLICY_PROMPT,
    PUSHY_USER,
    SLOPPY_INSTRUCTION,
    with_user_variant,
)


class Instr(BaseModel):
    task_instructions: str
    reason_for_call: str


class Scenario(BaseModel):
    instructions: Instr | str


class Task(BaseModel):
    id: str
    user_scenario: Scenario


def test_default_user_variant_is_unchanged() -> None:
    task = Task(id="1", user_scenario=Scenario(instructions="be nice"))
    assert with_user_variant(task, "tau2_default") is task


def test_pushy_appends_to_structured_instructions() -> None:
    task = Task(
        id="1",
        user_scenario=Scenario(
            instructions=Instr(task_instructions="Return it.", reason_for_call="r")
        ),
    )
    out = with_user_variant(task, "pushy")
    assert out.user_scenario.instructions.task_instructions == f"Return it.\n\n{PUSHY_USER}"
    assert out.user_scenario.instructions.reason_for_call == "r"
    assert task.user_scenario.instructions.task_instructions == "Return it."  # original untouched


def test_pushy_appends_to_string_instructions() -> None:
    out = with_user_variant(Task(id="1", user_scenario=Scenario(instructions="Go.")), "pushy")
    assert out.user_scenario.instructions == f"Go.\n\n{PUSHY_USER}"


def _cfg(**kw: object) -> AgentRunConfig:
    base = {"run_name": "r", "domain": "retail", "agent_model": "a", "user_model": "u"}
    return AgentRunConfig.model_validate({**base, "budget_usd": 1.0, **kw})


def test_conditions_seeded_and_weighted() -> None:
    cfg = _cfg(
        prompt_variants={"tau2_default": 0.5, "sloppy": 0.25, "no_policy": 0.25},
        user_variants={"tau2_default": 0.7, "pushy": 0.3},
        faults={"run_rate": 0.4, "call_rate": 0.3, "types": ["timeout"]},
    )
    assert choose_conditions(cfg, "7", 0) == choose_conditions(cfg, "7", 0)
    draws = [choose_conditions(cfg, str(i), 0) for i in range(4000)]
    prompts = Counter(d[0] for d in draws)
    assert abs(prompts["tau2_default"] / 4000 - 0.5) < 0.03
    assert abs(prompts["no_policy"] / 4000 - 0.25) < 0.03
    assert abs(sum(d[1] == "pushy" for d in draws) / 4000 - 0.3) < 0.03
    assert abs(sum(d[2] for d in draws) / 4000 - 0.4) < 0.03


def test_defaults_are_the_plain_tau2_setup() -> None:
    assert {choose_conditions(_cfg(), str(i), 0) for i in range(50)} == {
        ("tau2_default", "tau2_default", False)
    }


def test_bad_weights_rejected() -> None:
    with pytest.raises(ValueError):
        _cfg(prompt_variants={"sloppy": 0.0})
    with pytest.raises(ValueError):
        _cfg(user_variants={"angry": 1.0})


needs_tau2 = pytest.mark.skipif(
    importlib.util.find_spec("tau2") is None
    or not (DEFAULT_TAU2_CHECKOUT / "data" / "tau2" / "domains").is_dir(),
    reason="needs the agent extra and `atl-agent fetch-data`",
)


@needs_tau2
def test_registered_prompt_variants() -> None:
    from atl.agent.run_tau2 import _point_tau2_at_data

    _point_tau2_at_data()
    from tau2.registry import registry

    from atl.agent.variants import register_prompt_variants

    register_prompt_variants()
    register_prompt_variants()  # idempotent
    make = {v: registry.get_agent_factory(AGENT_NAMES[v]) for v in ("sloppy", "no_policy")}
    sloppy = make["sloppy"](tools=[], domain_policy="POLICY TEXT", llm="m", llm_args={})
    no_policy = make["no_policy"](tools=[], domain_policy="POLICY TEXT", llm="m", llm_args={})
    assert "POLICY TEXT" in sloppy.system_prompt and SLOPPY_INSTRUCTION in sloppy.system_prompt
    assert (
        no_policy.system_prompt == NO_POLICY_PROMPT and "POLICY TEXT" not in no_policy.system_prompt
    )
