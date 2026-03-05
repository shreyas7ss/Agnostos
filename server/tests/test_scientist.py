"""
Unit tests for the Scientist Agent
Tests proposal generation for tabular and image data experiments
"""
import pytest
from unittest.mock import patch, MagicMock

from agents.scientist import MLScript, ScienctistProposals


# ============ Pydantic Model Tests ============

class TestMLScript:
    """Tests for the MLScript Pydantic model."""
    
    def test_valid_ml_script(self):
        """Test that MLScript accepts valid input."""
        script = MLScript(
            approach_name="RandomForest",
            explanation="Good for tabular data with mixed features",
            code="import sklearn\nmodel = RandomForestClassifier()"
        )
        
        assert script.approach_name == "RandomForest"
        assert "sklearn" in script.code
    
    def test_ml_script_required_fields(self):
        """Test that MLScript requires all fields."""
        with pytest.raises(Exception):
            MLScript(approach_name="Test")
    
    def test_ml_script_model_dump(self):
        """Test that MLScript can be serialized to dict."""
        script = MLScript(
            approach_name="XGBoost",
            explanation="Handles missing values well",
            code="import xgboost as xgb"
        )
        
        result = script.model_dump()
        
        assert isinstance(result, dict)
        assert result["approach_name"] == "XGBoost"
        assert "explanation" in result
        assert "code" in result


class TestScientistProposals:
    """Tests for the ScientistProposals Pydantic model."""
    
    def test_valid_proposals(self):
        """Test that ScientistProposals accepts a list of experiments."""
        proposals = ScienctistProposals(
            experiments=[
                MLScript(
                    approach_name="RandomForest",
                    explanation="Ensemble method",
                    code="# RF code"
                ),
                MLScript(
                    approach_name="XGBoost",
                    explanation="Gradient boosting",
                    code="# XGB code"
                )
            ]
        )
        
        assert len(proposals.experiments) == 2
        assert proposals.experiments[0].approach_name == "RandomForest"
    
    def test_empty_experiments_list(self):
        """Test handling of empty experiments list."""
        proposals = ScienctistProposals(experiments=[])
        
        assert len(proposals.experiments) == 0


# ============ Fixtures ============

@pytest.fixture
def tabular_manifesto():
    """Sample tabular data manifesto from profiler."""
    return {
        "data_type": "tabular",
        "num_rows": 1000,
        "num_cols": 10,
        "column_names": ["feature1", "feature2", "target"],
        "missing_values": {"feature1": 5, "feature2": 0, "target": 0},
        "data_types": {"feature1": "float64", "feature2": "int64", "target": "int64"},
        "categorical_cardinality": {"category": 5},
        "numeric_summary": {
            "feature1": {"mean": 50.0, "std": 10.0},
            "feature2": {"mean": 100.0, "std": 20.0}
        }
    }


@pytest.fixture
def image_manifesto():
    """Sample image data manifesto from profiler."""
    return {
        "data_type": "image",
        "total_images": 500,
        "formats": ["PNG", "JPEG"],
        "color_modes": ["RGB"],
        "sample_resolutions": [(224, 224), (224, 224)],
        "average_res": [224.0, 224.0],
        "is_uniform_size": True
    }


@pytest.fixture
def mock_scientist_response():
    """Mock response from structured LLM for scientist agent."""
    return ScienctistProposals(
        experiments=[
            MLScript(
                approach_name="RandomForest",
                explanation="Ensemble method good for tabular data",
                code="""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
with open('metrics.json', 'w') as f:
    json.dump({"accuracy": accuracy, "loss": 1-accuracy}, f)
"""
            ),
            MLScript(
                approach_name="XGBoost",
                explanation="Gradient boosting for high accuracy",
                code="""
import pandas as pd
import xgboost as xgb
import json

# XGBoost training code
"""
            )
        ]
    )


# ============ Agent Tests ============

class TestScientistAgent:
    """Tests for the scientist_agent function."""
    
    @patch('agents.scientist.ChatGroq')
    @patch('agents.scientist.settings')
    def test_returns_candidate_scripts(self, mock_settings, mock_chat_groq, tabular_manifesto, mock_scientist_response):
        """Test that scientist agent returns candidate scripts."""
        from agents.scientist import scientist_agent
        from graph.state import AgentState
        
        # Setup mocks
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        mock_settings.max_parallel_attempts = 2
        
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_scientist_response
        mock_chat_groq.return_value = mock_llm
        
        state = AgentState(
            dataset_path="test_data.csv",
            data_manifesto=tabular_manifesto
        )
        
        result = scientist_agent(state)
        
        assert "candidate_scripts" in result
        assert "messages" in result
        assert result["next_step"] == "judge"
    
    @patch('agents.scientist.ChatGroq')
    @patch('agents.scientist.settings')
    def test_tabular_uses_sklearn_prompt(self, mock_settings, mock_chat_groq, tabular_manifesto, mock_scientist_response):
        """Test that tabular data triggers sklearn-focused prompt."""
        from agents.scientist import scientist_agent
        from graph.state import AgentState
        
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        mock_settings.max_parallel_attempts = 2
        
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = mock_scientist_response
        mock_llm.with_structured_output.return_value = structured_llm
        mock_chat_groq.return_value = mock_llm
        
        state = AgentState(
            dataset_path="test_data.csv",
            data_manifesto=tabular_manifesto
        )
        
        scientist_agent(state)
        
        # Verify LLM was called
        structured_llm.invoke.assert_called_once()
        call_args = structured_llm.invoke.call_args[0][0]
        assert "Scikit-Learn" in call_args or "XGBoost" in call_args
    
    @patch('agents.scientist.ChatGroq')
    @patch('agents.scientist.settings')
    def test_image_uses_pytorch_prompt(self, mock_settings, mock_chat_groq, image_manifesto, mock_scientist_response):
        """Test that image data triggers PyTorch-focused prompt."""
        from agents.scientist import scientist_agent
        from graph.state import AgentState
        
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        mock_settings.max_parallel_attempts = 2
        
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = mock_scientist_response
        mock_llm.with_structured_output.return_value = structured_llm
        mock_chat_groq.return_value = mock_llm
        
        state = AgentState(
            dataset_path="/path/to/images",
            data_manifesto=image_manifesto
        )
        
        scientist_agent(state)
        
        # Verify LLM was called
        structured_llm.invoke.assert_called_once()
        call_args = structured_llm.invoke.call_args[0][0]
        assert "PyTorch" in call_args or "ResNet" in call_args
    
    @patch('agents.scientist.ChatGroq')
    @patch('agents.scientist.settings')
    def test_handles_llm_failure(self, mock_settings, mock_chat_groq, tabular_manifesto):
        """Test that agent handles LLM failures gracefully."""
        from agents.scientist import scientist_agent
        from graph.state import AgentState
        
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        mock_settings.max_parallel_attempts = 2
        
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM Error")
        mock_chat_groq.return_value = mock_llm
        
        state = AgentState(
            dataset_path="test_data.csv",
            data_manifesto=tabular_manifesto
        )
        
        result = scientist_agent(state)
        
        assert result["candidate_scripts"] == []
        assert "failed" in result["messages"][0].content.lower()
        assert result["next_step"] == "judge"
    
    @patch('agents.scientist.ChatGroq')
    @patch('agents.scientist.settings')
    def test_respects_max_parallel_attempts(self, mock_settings, mock_chat_groq, tabular_manifesto):
        """Test that agent respects max_parallel_attempts setting."""
        from agents.scientist import scientist_agent
        from graph.state import AgentState
        
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        mock_settings.max_parallel_attempts = 3
        
        # Create response with more experiments than allowed
        too_many_experiments = ScienctistProposals(
            experiments=[
                MLScript(
                    approach_name=f"Model_{i}",
                    explanation=f"Description {i}",
                    code=f"# Code {i}"
                ) for i in range(5)
            ]
        )
        
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = too_many_experiments
        mock_chat_groq.return_value = mock_llm
        
        state = AgentState(
            dataset_path="test_data.csv",
            data_manifesto=tabular_manifesto
        )
        
        result = scientist_agent(state)
        
        # Should be limited to max_parallel_attempts
        assert len(result["candidate_scripts"]) <= mock_settings.max_parallel_attempts
    
    @patch('agents.scientist.ChatGroq')
    @patch('agents.scientist.settings')
    def test_messages_contain_summary(self, mock_settings, mock_chat_groq, tabular_manifesto, mock_scientist_response):
        """Test that returned messages contain strategy summary."""
        from agents.scientist import scientist_agent
        from graph.state import AgentState
        from langchain_core.messages import AIMessage
        
        mock_settings.llm_model = "llama-3.3-70b-versatile"
        mock_settings.max_parallel_attempts = 2
        
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_scientist_response
        mock_chat_groq.return_value = mock_llm
        
        state = AgentState(
            dataset_path="test_data.csv",
            data_manifesto=tabular_manifesto
        )
        
        result = scientist_agent(state)
        
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][0], AIMessage)
        # Summary should mention strategies
        assert "RandomForest" in result["messages"][0].content or "strategies" in result["messages"][0].content.lower()


class TestScientistScriptGeneration:
    """Tests for the script generation quality."""
    
    def test_script_contains_required_elements(self, mock_scientist_response):
        """Test that generated scripts have required components."""
        for exp in mock_scientist_response.experiments:
            script_dict = exp.model_dump()
            
            assert "approach_name" in script_dict
            assert "explanation" in script_dict
            assert "code" in script_dict
            assert len(script_dict["approach_name"]) > 0
    
    def test_script_code_is_executable_format(self, mock_scientist_response):
        """Test that code field contains valid Python-like content."""
        script = mock_scientist_response.experiments[0]
        
        # Should contain Python import statements or code structure
        assert "import" in script.code or "#" in script.code


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
