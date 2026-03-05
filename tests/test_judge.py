"""
Unit tests for the Judge Agent
Tests metric evaluation, winner selection, LLM verdict generation,
and edge cases like no successful runs.

Strategy:
- 'modal' and 'langchain_groq.ChatGroq' are stubbed in sys.modules BEFORE
  any project imports so no real API keys or cloud SDKs are needed.
- We import agents.judge once, then use patch.object() inside each test to
  control execute_in_parallel and the module-level llm.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─────────────────────────────────────────────
# 1. Stub 'modal' before anything imports it
# ─────────────────────────────────────────────

_modal_stub = types.ModuleType("modal")
_modal_image = MagicMock()
_modal_image.debian_slim.return_value = _modal_image
_modal_image.pip_install.return_value  = _modal_image
_modal_stub.Image = _modal_image
_modal_stub.App   = MagicMock(return_value=MagicMock())

def _modal_function_decorator(*args, **kwargs):
    """Stand-in for @modal.App.function(...)"""
    def decorator(fn):
        fn.remote = MagicMock()
        fn.remote.aio = AsyncMock(return_value={"status": "success", "results": {}})
        return fn
    return decorator

_modal_stub.App.return_value.function = _modal_function_decorator
sys.modules["modal"] = _modal_stub

# ─────────────────────────────────────────────
# 2. Stub 'langchain_groq' so ChatGroq() doesn't call the real API
# ─────────────────────────────────────────────

_mock_llm_instance = MagicMock()
_mock_llm_instance.invoke.return_value = MagicMock(content="Default mock verdict.")

_langchain_groq_stub = types.ModuleType("langchain_groq")
_langchain_groq_stub.ChatGroq = MagicMock(return_value=_mock_llm_instance)
sys.modules["langchain_groq"] = _langchain_groq_stub

# ─────────────────────────────────────────────
# 3. Now it's safe to import project modules
# ─────────────────────────────────────────────

# Remove any stale cached version
for _mod in list(sys.modules):
    if _mod in ("agents.judge", "tools.executor"):
        del sys.modules[_mod]

import agents.judge as _judge_module  # noqa: E402
from graph.state import AgentState     # noqa: E402

# Ensure the module-level llm points to our mock instance
_judge_module.llm = _mock_llm_instance


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_state(candidate_scripts: list) -> AgentState:
    return AgentState(candidate_scripts=candidate_scripts)


def make_script(approach_name: str, code: str = "print('hello')") -> dict:
    return {"approach_name": approach_name, "code": code}


def make_execution_result(approach_name: str, status: str, accuracy: float = 0.0) -> dict:
    """
    Matches the dict returned by execute_in_parallel after our fix:
    { approach_name, status, execution: { results: { accuracy, f1_score } } }
    """
    return {
        "approach_name": approach_name,
        "status": status,
        "execution": {
            "results": {
                "accuracy": accuracy,
                "f1_score": round(accuracy * 0.95, 4),
            }
        },
    }


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_llm_mock():
    """Reset the shared LLM mock before every test to avoid bleed-through."""
    _mock_llm_instance.reset_mock()
    _mock_llm_instance.invoke.return_value = MagicMock(content="Default verdict.")
    yield _mock_llm_instance


@pytest.fixture
def two_successful_scripts():
    return make_state([make_script("random_forest"), make_script("xgboost")])


@pytest.fixture
def one_failed_script():
    return make_state([make_script("svm"), make_script("logistic_regression")])


@pytest.fixture
def all_failed_scripts():
    return make_state([make_script("bad_model_1"), make_script("bad_model_2")])


# ─────────────────────────────────────────────
# Tests: No successful executions
# ─────────────────────────────────────────────

class TestJudgeNoSuccessfulRuns:

    @pytest.mark.asyncio
    async def test_returns_end_when_all_fail(self, all_failed_scripts):
        """Judge should return next_step='end' when nothing succeeds."""
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("bad_model_1", "error"),
                make_execution_result("bad_model_2", "error"),
            ]
            result = await _judge_module.judge_agent(all_failed_scripts)

        assert result["next_step"] == "end"

    @pytest.mark.asyncio
    async def test_message_contains_no_winner_text(self, all_failed_scripts):
        """A helpful error message should appear when no runs succeed."""
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [make_execution_result("bad_model_1", "error")]
            result = await _judge_module.judge_agent(all_failed_scripts)

        text = result["messages"][0].content.lower()
        assert any(p in text for p in ["no candidate", "cannot determine", "no successful"]), (
            f"Unexpected message: {text}"
        )

    @pytest.mark.asyncio
    async def test_llm_not_called_when_all_fail(self, reset_llm_mock, all_failed_scripts):
        """LLM should not be consulted when there's no winner to describe."""
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [make_execution_result("bad_model_1", "error")]
            await _judge_module.judge_agent(all_failed_scripts)

        reset_llm_mock.invoke.assert_not_called()


# ─────────────────────────────────────────────
# Tests: Winner selection
# ─────────────────────────────────────────────

class TestJudgeWinnerSelection:

    @pytest.mark.asyncio
    async def test_highest_accuracy_wins(self, reset_llm_mock, two_successful_scripts):
        """The approach with the highest accuracy should become the champion."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost is clearly superior.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.82),
                make_execution_result("xgboost",       "success", accuracy=0.91),
            ]
            result = await _judge_module.judge_agent(two_successful_scripts)

        assert result["final_output"]["approach_name"] == "xgboost"
        assert result["final_output"]["accuracy"] == 0.91

    @pytest.mark.asyncio
    async def test_winner_chosen_among_mixed_results(self, reset_llm_mock, one_failed_script):
        """Only the successful run should be eligible; the failed one is ignored."""
        reset_llm_mock.invoke.return_value = MagicMock(content="Logistic Regression wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("svm",                 "error",   accuracy=0.0),
                make_execution_result("logistic_regression", "success", accuracy=0.78),
            ]
            result = await _judge_module.judge_agent(one_failed_script)

        assert result["final_output"]["approach_name"] == "logistic_regression"

    @pytest.mark.asyncio
    async def test_champion_bundle_has_code(self, reset_llm_mock, two_successful_scripts):
        """final_output should include the source code of the winning script."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.75),
                make_execution_result("xgboost",       "success", accuracy=0.88),
            ]
            result = await _judge_module.judge_agent(two_successful_scripts)

        assert "code" in result["final_output"]
        assert result["final_output"]["code"] != "Code not found"

    @pytest.mark.asyncio
    async def test_champion_bundle_has_metrics(self, reset_llm_mock, two_successful_scripts):
        """final_output['metrics'] should contain the full results dict."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.80),
                make_execution_result("xgboost",       "success", accuracy=0.93),
            ]
            result = await _judge_module.judge_agent(two_successful_scripts)

        metrics = result["final_output"]["metrics"]
        assert "accuracy" in metrics
        assert "f1_score" in metrics


# ─────────────────────────────────────────────
# Tests: LLM verdict
# ─────────────────────────────────────────────

class TestJudgeVerdict:

    @pytest.mark.asyncio
    async def test_llm_called_once_on_success(self, reset_llm_mock, two_successful_scripts):
        """LLM must be invoked exactly once to produce the verdict."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost is superior.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.82),
                make_execution_result("xgboost",       "success", accuracy=0.91),
            ]
            await _judge_module.judge_agent(two_successful_scripts)

        reset_llm_mock.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_prompt_mentions_winner(self, reset_llm_mock, two_successful_scripts):
        """The prompt sent to the LLM should reference the winning approach name."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost is the best.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.82),
                make_execution_result("xgboost",       "success", accuracy=0.91),
            ]
            await _judge_module.judge_agent(two_successful_scripts)

        prompt_sent = reset_llm_mock.invoke.call_args[0][0]
        assert "xgboost" in prompt_sent.lower()

    @pytest.mark.asyncio
    async def test_verdict_appears_in_messages(self, reset_llm_mock, two_successful_scripts):
        """The LLM verdict string should appear inside the returned messages list."""
        verdict_text = "XGBoost dominated with superior generalisation."
        reset_llm_mock.invoke.return_value = MagicMock(content=verdict_text)

        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.82),
                make_execution_result("xgboost",       "success", accuracy=0.91),
            ]
            result = await _judge_module.judge_agent(two_successful_scripts)

        combined = " ".join(m.content for m in result["messages"])
        assert verdict_text in combined


# ─────────────────────────────────────────────
# Tests: Return shape
# ─────────────────────────────────────────────

class TestJudgeReturnShape:

    @pytest.mark.asyncio
    async def test_success_return_has_required_keys(self, reset_llm_mock, two_successful_scripts):
        """On success the returned dict must have final_output, messages, next_step."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.85),
                make_execution_result("xgboost",       "success", accuracy=0.90),
            ]
            result = await _judge_module.judge_agent(two_successful_scripts)

        assert "final_output" in result
        assert "messages"     in result
        assert "next_step"    in result

    @pytest.mark.asyncio
    async def test_success_next_step_is_completed(self, reset_llm_mock, two_successful_scripts):
        """next_step must be 'completed' when a winner is found."""
        reset_llm_mock.invoke.return_value = MagicMock(content="Random Forest wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.85),
            ]
            result = await _judge_module.judge_agent(two_successful_scripts)

        assert result["next_step"] == "completed"

    @pytest.mark.asyncio
    async def test_failure_return_has_required_keys(self, all_failed_scripts):
        """Even when all scripts fail the returned dict must be well-formed."""
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [make_execution_result("bad_model_1", "error")]
            result = await _judge_module.judge_agent(all_failed_scripts)

        assert "messages"  in result
        assert "next_step" in result


# ─────────────────────────────────────────────
# Tests: Executor integration
# ─────────────────────────────────────────────

class TestJudgeExecutorIntegration:

    @pytest.mark.asyncio
    async def test_executor_called_with_candidate_scripts(self, reset_llm_mock, two_successful_scripts):
        """execute_in_parallel must receive the exact candidate_scripts from state."""
        reset_llm_mock.invoke.return_value = MagicMock(content="XGBoost wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.80),
                make_execution_result("xgboost",       "success", accuracy=0.88),
            ]
            await _judge_module.judge_agent(two_successful_scripts)

        mock_exec.assert_called_once_with(two_successful_scripts.candidate_scripts)

    @pytest.mark.asyncio
    async def test_executor_called_once(self, reset_llm_mock, two_successful_scripts):
        """execute_in_parallel should only be called once per judge invocation."""
        reset_llm_mock.invoke.return_value = MagicMock(content="Random Forest wins.")
        with patch.object(_judge_module, "execute_in_parallel", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = [
                make_execution_result("random_forest", "success", accuracy=0.80),
            ]
            await _judge_module.judge_agent(two_successful_scripts)

        assert mock_exec.call_count == 1


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
