# SCHEMA: A Structured Collaborative Hierarchical Expert Multi-Agent System for Research Proposal Generation

**SCHEMA** is a multi-agent framework designed to simulate a professional research laboratory environment. By utilizing iterative expert discussions, scientific critique, and real-world grounding, SCHEMA transforms limited research ideas into high-quality, structured scientific proposals.

## 🚀 Overview

Unlike traditional LLM generation, SCHEMA employs a **Hierarchical Expert Multi-Agent System**:

* **Expert Agents (EAs):** Specialized personas that brainstorm and refine specific domains.
* **Scientific Critic (SC):** Provides rigorous, objective feedback and evaluates "satisfaction" scores.
* **Principal Investigator (PI):** Manages the workflow, ensuring the discussion remains focused and productive.
* **Real-world Grounding:** Integrated with **Tavily** to fetch current trends and ensure empirical validity.

## 🛠️ Installation & Setup

1. **Download the code**


2. **Install dependencies:**
Navigate to the SCHEMA MAIN directory and install the required dependencies:
```bash
cd "SCHEMA MAIN"
pip install -r requirements.txt

```

4. **Environment Configuration:**
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

```



## 📊 Running Evaluations

To replicate the evaluations presented in the paper (across the 500-paper dataset spanning ECS, HMS, SS, BEM, and HLA), simply run the main execution script in the SCHEMA MAIN file:

```bash
python main.py

```

This script will initialize the agentic workflow, conduct the simulated meetings, and output the generated proposals along with their evaluation metrics.



