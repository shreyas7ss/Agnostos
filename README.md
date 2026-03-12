# Agnostos: The Autonomous Multimodal ML Lab 🔬

Agnostos is an end-to-end agentic system designed to automate the machine learning lifecycle. Unlike standard AutoML, Agnostos utilizes a hierarchical swarm of specialized AI agents that perform autonomous EDA, generate training code, and execute simulations in isolated, 100% cloud-based sandboxes to find the optimal model for Tabular or Image datasets.

## 🚀 Key Features

- **100% Cloud-Native Execution & Storage**: No local compute required! Datasets are uploaded directly to Supabase Storage, and all agent-generated ML training scripts execute securely in serverless cloud containers using Modal. 
- **Self-Healing Execution Loop**: The Executor Agent doesn't just run code—it fixes it. If a cloud training run fails due to a bug or missing dependency, the executor catches the traceback, feeds it back to a specialized code-fixing LLM, and autonomously repairs and re-runs the script!
- **Holistic Model Evaluation**: The Judge Agent decides the winner based on a comprehensive suite of metrics—including **Accuracy, F1 Score, Precision, and Recall**. This guarantees that the winning model is truly robust, avoiding traps like high accuracy on heavily imbalanced datasets.
- **Agentic Orchestration**: Built with LangGraph to manage complex, stateful transitions and run multiple ML scientists in parallel.
- **VLM-Powered EDA**: Employs Vision-Language Models (VLMs) to "look" at data distributions, providing semantic insights beyond simple statistics.

## 🏗️ Project Structure

The repository follows a decoupled architecture separating reasoning, skeleton, and capabilities:

```
agnostos-lab/
├── agents/             # The "Brains": Personas & System Prompts
│   ├── profiler.py     # Performs deep EDA on data (Tabular stats / Image VLM)
│   ├── scientist.py    # Generates custom training logic in parallel
│   └── judge.py        # Critiques diverse metrics (F1, Accuracy, Recall) & selects the ultimate winner
├── graph/              # The "Skeleton": State Machine Orchestration
│   ├── state.py        # Shared memory State (TypedDict) for inter-agent context
│   └── workflow.py     # LangGraph StateGraph (Nodes, Edges, Conditional Logic)
├── tools/              # The "Hands": Execution & Capabilities
│   ├── executor.py     # Autonomous self-healing interface for Modal cloud execution
│   └── stats.py        # Automated plotting and statistical profiling tools
├── app.py              # FastAPI server orchestrating the backend
├── client/             # The modern, futuristic React frontend
└── requirements.txt    # Production dependencies
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frameworks** | LangGraph, LangChain, FastAPI, React (Vite) |
| **Models** | Llama 3 70B / Qwen (via Groq), YOLOv11 (Vision), Llava (Visual EDA) |
| **Infrastructure** | Modal (Serverless GPU Compute), Supabase (Cloud Storage) |
| **Data Stack** | Pandas, Scikit-Learn, LightGBM, XGBoost, PyTorch |

## 🧠 System Workflow

1. **Ingestion & Storage**: You upload a dataset via the React UI. It gets securely stored in Supabase Storage and returns a public URL for cloud-native access.
2. **Autonomous Profiling**: The Profiler Agent reads the data remotely, running statistical tests to create a contextual "Data Manifesto."
3. **Strategy Planning**: The Scientist Agent spawns 3 identical model-building routines in parallel, each taking a unique approach (e.g., Random Forest vs LightGBM vs Neural Net).
4. **Self-Healing Simulation**: Generated scripts are launched simultaneously into Modal cloud containers. If an import fails or a tensor shape mismatches, the Executor uses an LLM debugger to patch the code and retry automatically!
5. **The Verdict**: The Judge Agent inspects each parallel model's `metrics.json`. By weighing **Accuracy, F1 Score, Precision, and Recall**, it selects the absolute best architecture for your specific dataset topology.

## 🏁 Quick Start

### Prerequisites

- Python 3.10+ and Node.js
- `GROQ_API_KEY` (Free LLM inference)
- Modal Account (for serverless execution)
- Supabase Project (for dataset storage)

### Installation

1. Clone the repo:
```bash
git clone https://github.com/shreyas7ss/Agnostos.git
cd Agnostos
```

2. Setup the Server:
```bash
cd server
pip install -r requirements.txt
# Set your keys in .env
python main.py
```

3. Setup the UI:
```bash
cd ../client
npm install
npm run dev
```

4. Go to `http://localhost:5173` and launch a parallel ML pipeline!
