# Agnostos: The Autonomous Multimodal ML Lab 🔬

Agnostos is an end-to-end agentic system designed to automate the machine learning lifecycle. Unlike standard AutoML, Agnostos utilizes a hierarchical swarm of specialized AI agents that perform autonomous EDA, generate training code, and execute simulations in isolated sandboxes to find the optimal model for any Tabular or Image dataset.

## 🚀 Key Features

- **Multimodal Intelligence**: Automatically detects and routes data to specialized branches for Tabular (Scikit-learn/XGBoost) or Image (PyTorch/YOLO) tasks.

- **Agentic Orchestration**: Built with LangGraph to manage complex, stateful transitions and parallel agent execution.

- **Self-Healing Simulations**: Uses E2B/Modal MicroVMs to execute agent-generated code. If training fails, agents analyze the traceback and autonomously fix the code.

- **VLM-Powered EDA**: Employs Vision-Language Models (VLMs) to "look" at data distributions and sample images, providing semantic insights beyond simple statistics.

## 🏗️ Project Structure

The repository follows a decoupled architecture separating reasoning, skeleton, and capabilities:

```
agnostos-lab/
├── agents/             # The "Brains": Personas & System Prompts
│   ├── profiler.py     # Performs deep EDA on data (Tabular stats / Image VLM)
│   ├── scientist.py    # Generates custom training logic (YOLO, Transformers, Trees)
│   └── judge.py        # Critiques metrics & selects the production-ready winner
├── graph/              # The "Skeleton": State Machine Orchestration
│   ├── state.py        # Shared memory State (TypedDict) for inter-agent context
│   └── workflow.py     # LangGraph StateGraph (Nodes, Edges, Conditional Logic)
├── tools/              # The "Hands": Execution & Capabilities
│   ├── executor.py     # Interface for E2B/Modal sandboxed code execution
│   ├── vision.py       # Tools for CLIP embeddings and VLM analysis
│   └── stats.py        # Automated plotting and statistical profiling tools
├── app.py              # Streamlit-based Command Center
└── requirements.txt    # Production dependencies
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frameworks** | LangGraph, LangChain, PyTorch |
| **Models** | GPT-4o / Claude 3.5 (Orchestration), YOLOv11 (Vision), Llava (Visual EDA) |
| **Infrastructure** | E2B (Sandboxing), Modal (GPU Compute), Docker |
| **Frontend** | Streamlit |

## 🧠 System Workflow

1. **Ingestion & Routing**: Data is classified by the Router tool.

2. **Autonomous Profiling**: The Profiler Agent runs statistical tests and VLM sampling to create a "Data Manifesto."

3. **Strategy Planning**: The Planner Agent spawns multiple Scientist Agents in parallel to explore different architectures (e.g., ResNet vs. YOLO).

4. **Sandboxed Simulation**: Scientists write and run training scripts in MicroVMs. Metrics are streamed back to the state.

5. **The Verdict**: The Judge Agent performs a weighted evaluation of Accuracy vs. Latency and prepares a deployment-ready FastAPI wrapper.

Since you are using Modal for compute and Neon for your database, you’ve created a fully serverless, high-performance stack.
## 🏁 Quick Start

### Prerequisites

- Python 3.10+
- E2B API Key (for sandboxing)
- OpenAI/Anthropic API Key

### Installation

1. Clone the repo:
```bash
git clone https://github.com/your-username/agnostos-lab.git
cd agnostos-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Lab:
```bash
streamlit run app.py
```


