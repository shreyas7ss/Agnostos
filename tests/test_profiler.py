"""
Unit tests for the Profiler Agent
Tests both tabular and image profiling capabilities
"""
import os
import pytest
import tempfile
import pandas as pd
from PIL import Image
from unittest.mock import patch, MagicMock

from tools.stats import tabular_profiler
from tools.vision import image_profiler


# ============ Fixtures ============

@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    df = pd.DataFrame({
        "age": [25, 30, 35, None, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "department": ["HR", "IT", "IT", "Sales", "HR"],
        "rating": [4.5, 3.8, 4.2, 4.0, 4.8]
    })
    
    # Create temp file, write, close, then yield path
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(path, index=False)
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_image_folder():
    """Create a temporary folder with sample images."""
    tmpdir = tempfile.mkdtemp()
    
    # Create sample images
    for i in range(3):
        img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
        img.save(os.path.join(tmpdir, f'image_{i}.png'))
    
    yield tmpdir
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============ Tool Tests ============

class TestTabularProfiler:
    """Tests for the tabular_profiler tool."""
    
    def test_basic_stats(self, sample_csv_file):
        """Test that basic statistics are calculated correctly."""
        result = tabular_profiler(sample_csv_file)
        
        assert result["num_rows"] == 5
        assert result["num_cols"] == 4
        assert "age" in result["column_names"]
        assert "salary" in result["column_names"]
    
    def test_missing_values_detected(self, sample_csv_file):
        """Test that missing values are detected."""
        result = tabular_profiler(sample_csv_file)
        
        assert result["missing_values"]["age"] == 1
        assert result["missing_values"]["salary"] == 0
    
    def test_data_types(self, sample_csv_file):
        """Test that data types are correctly identified."""
        result = tabular_profiler(sample_csv_file)
        
        assert "float" in result["data_types"]["age"] or "int" in result["data_types"]["age"]
        assert result["data_types"]["department"] == "object"
    
    def test_correlation_matrix(self, sample_csv_file):
        """Test that correlation matrix is computed for numeric columns."""
        result = tabular_profiler(sample_csv_file)
        
        assert "correlation_matrix" in result
        assert "salary" in result["correlation_matrix"]
    
    def test_categorical_cardinality(self, sample_csv_file):
        """Test that categorical cardinality is computed."""
        result = tabular_profiler(sample_csv_file)
        
        assert result["categorical_cardinality"]["department"] == 3


class TestImageProfiler:
    """Tests for the image_profiler tool."""
    
    def test_image_count(self, sample_image_folder):
        """Test that image count is correct."""
        result = image_profiler(sample_image_folder)
        
        assert result["total_images"] == 3
    
    def test_image_formats(self, sample_image_folder):
        """Test that image formats are detected."""
        result = image_profiler(sample_image_folder)
        
        assert "PNG" in result["formats"]
    
    def test_color_modes(self, sample_image_folder):
        """Test that color modes are detected."""
        result = image_profiler(sample_image_folder)
        
        assert "RGB" in result["color_modes"]
    
    def test_uniform_size_detection(self, sample_image_folder):
        """Test that uniform size is correctly detected."""
        result = image_profiler(sample_image_folder)
        
        assert result["is_uniform_size"] == True
    
    def test_empty_folder(self):
        """Test handling of empty folder."""
        tmpdir = tempfile.mkdtemp()
        try:
            result = image_profiler(tmpdir)
            assert "error" in result
        finally:
            os.rmdir(tmpdir)


# ============ Agent Tests ============

class TestProfilerAgent:
    """Tests for the profiler_agent function."""
    
    @patch('agents.profiler.llm')
    def test_tabular_routing(self, mock_llm, sample_csv_file):
        """Test that CSV files are routed to tabular profiler."""
        from agents.profiler import profiler_agent
        from graph.state import AgentState
        
        mock_llm.invoke.return_value = MagicMock(content="Test summary")
        
        state = AgentState(dataset_path=sample_csv_file)
        
        result = profiler_agent(state)
        
        assert result["data_manifesto"]["data_type"] == "tabular"
        assert result["next_step"] == "scientist"
    
    @patch('agents.profiler.llm')
    def test_image_routing(self, mock_llm, sample_image_folder):
        """Test that image folders are routed to image profiler."""
        from agents.profiler import profiler_agent
        from graph.state import AgentState
        
        mock_llm.invoke.return_value = MagicMock(content="Test summary")
        
        state = AgentState(dataset_path=sample_image_folder)
        
        result = profiler_agent(state)
        
        assert result["data_manifesto"]["data_type"] == "image"
        assert result["next_step"] == "scientist"
    
    @patch('agents.profiler.llm')
    def test_llm_called(self, mock_llm, sample_csv_file):
        """Test that LLM is called with correct context."""
        from agents.profiler import profiler_agent
        from graph.state import AgentState
        
        mock_llm.invoke.return_value = MagicMock(content="Test summary")
        
        state = AgentState(dataset_path=sample_csv_file)
        
        profiler_agent(state)
        
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "tabular" in call_args.lower()


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
