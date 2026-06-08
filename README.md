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
pip install -r requirements.txt
```

3. **Environment Configuration:**
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

```

4. **Automated Evaluation:**
From the Dataset folders, you can choose any either "Full" or "Ablation" and then copy and paste the "INPUT" and "Reference_Reports_json" in the SCHEMA folder


## 📊 Running Evaluations

To replicate the evaluations presented in the paper (with the dataset spanning ECS, HMS, SS, BEM, and HLA), simply run the main execution script in the SCHEMA MAIN file:

```bash
cd SCHEMA
python main.py

```

This script will initialize the agentic workflow, conduct the simulated meetings, and output the generated proposals along with their evaluation metrics.


**Note 1**: You can change the "model_selection_mode" (LLM Routing) by changing the variable name to the mode you wish in the main.py file 

**Note 2**: If you wish to only test the system once, you can copy paste the "main" and "environment" file from the "Individual_runs" folder placed inside "SCHEMA" and then run:

```bash
python main.py --crew_model* --topic** --sc-model*** --pi-model**** --iterations*****

```
In case of individual run you can also comment the eval() function from the main.py file, cause for evaluation you require a reference report as a ground-truth for comparison. 

**Note 3**: all parser options are optional, if you dont mention anything it will ask about topic and descrition along with number of agents and number of meeting in each simulation to you. In this case default model for all agents and mixed mode for EA is selected.
*= you can mention the name of the model<sup>#</sup> you want for crew (EAs), deafult is mixed.
**= you can mention the name of the research topic you have already created in the evironment.py file, it will only work if you have also added the topic in "mission_choices" list in the main.py file.
***= name of the model you want for SC (defult = gpt-5-mini).
****= name of the model you want for PI (defult = gpt-5-mini).
*****= how many number of complete simulations you want, default is 1 (it will create new proposals each time).

<sup>#</sup> = Model names should be the key value of the corresponding models mentioned in the "AVAILABLE_MODEL" list in the llm_api.py file.



5. **Results:**
You can find the results of the expriments conducted for the paper in the "Results" folder.
