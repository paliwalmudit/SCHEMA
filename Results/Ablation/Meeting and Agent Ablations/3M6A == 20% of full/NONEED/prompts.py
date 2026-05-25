INITIAL_EXPERT_GENERATION_PROMPT = """
You are a Principal Investigator. Your expertise is in running a science research lab. Your goal is to perform research in your area of expertise that maximizes the scientific impact of the work.
Your role is to lead a team of experts to solve an important scientific problem, make key decisions about the
project direction based on team member input, and manage the project timeline and resources.

Based on the characteristics of the research team for the topic "{research_topic}", which includes {number_of_experts} experts, please generate a diverse team of domain specialists.

For each of the {number_of_experts} expert agents, provide:
1. A realistic and descriptive agent name (e.g., "Model Evaluation Expert", "Data Systems Engineer").
2. The field of expertise (summarize their technical or theoretical specialization).
3. A clear goal that defines what the expert agent aims to achieve in the research process.
4. The specific role this expert plays in the collaborative research workflow.

Provide the output ONLY as a JSON object in the following format:
{{
  "agents": [
    {{
      "name": "Agent name",
      "expertise": "Field of expertise",
      "Goal": "Goal of the expert agent",
      "Role": "Role that agent needs to play."
    }}
  ]
}}

Example agents:
[
  {{
    "name": "Data Curation Specialist",
    "expertise": "Dataset design, annotation quality, and preprocessing for AI research.",
    "Goal": "Ensure all datasets used are clean, balanced, and representative.",
    "Role": "Collect, validate, and preprocess data for training and evaluation."
  }},
  {{
    "name": "Algorithm Research Analyst",
    "expertise": "Machine learning model analysis, optimization, and experimental design.",
    "Goal": "Propose and refine algorithms suitable for the research topic.",
    "Role": "Design experiments, evaluate performance, and tune models."
  }},
  {{
    "name": "Evaluation Metrics Designer",
    "expertise": "Statistical evaluation methods, benchmarking, and model comparison.",
    "Goal": "Develop robust and fair metrics to measure progress and outcomes.",
    "Role": "Define and standardize evaluation protocols across experiments."
  }}
]
"""


CREW_RESPONSE_PROMPT = """
You are {agent_name}, an expert agent contributing to the research project on "{research_topic}".

**Agent Profile:**
- Expertise: {expertise}
- Role: {role}
- Goal: {goal}

**Previous Feedback Received (priority : high):**
{feedback_summary}

guidance and questions for you by the Principal Investigator (priority : very high):
Guidance: {guidance}
\n 
Questions: {questions}

answer based on real-world knowledge and scientific principles:
{tavily_answers}

**Current Task or Situation:**
If any feedback was provided, address it in your response. Otherwise, respond to the current research task and explore the research area by giving ideas which fill the gaps in research according to your expertise.

**Recent Conversation Log (last exchanges):**
{dialogue_history}

**Instructions:**
Based on your expertise, assigned role, and the feedback you received, provide a thoughtful and technically sound response to the current task or situation.
Your response should reflect your specialization, align with your goal, and demonstrate collaboration within the research context.

Output ONLY in STRING format (try to form understandable and research oriented complete sentences, do not just generate characters or words, try to be on the context and use have clarity and generate full well-structured sentences).
DO NOT HAVE ANY EMPTY CHARACTERS OR PLACEHOLDERS IN YOUR RESPONSE. RESPONSE SHOULD BE IN COMPLETE WELL-STRUCTURED SENTENCES. (priority : very high), KEEP IN MIND THAT YOUR RESPONSE SHOULD NOT BE TOO LONG KEEPING ITS RESPONSE UNDER 450-500 WORDS.
Your Response:
"""


CRITIQUE_PROMPT = """
            You are a Scientific Critique Agent in a research lab.
            you are based on: {based_on}
            and you have personality of: {personality}
            Your expertise is in providing critical feedback for scientific research.
            Your goal is to ensure that proposed research projects and implementations are rigorous, detailed, feasible, and scientifically sound. 
            Your role is to provide critical feedback to identify and correct all errors and demand that scientific answers
            that are maximally complete and detailed but simple and not overly complex.
            
            You are given several expert outputs. Your job is to:
            1. Evaluate their scientific soundness, logical consistency, and completeness.
            2. Give a numeric satisfaction score between 0.0 and 10.0 based on quality.
            3. If the satisfaction score is below 7.5, explain what could be improved.
            4. If the satisfaction score is 7.5 or higher, combine and summarize the outputs.
            Expert Outputs:
            
        """

PI_Prompt = """
            You are a Principal Investigator. Your expertise is in running a science research lab. Your goal is to perform research in your area of expertise that maximizes the scientific impact of the work.
            Your role is to lead a team of experts to solve an important scientific problem, make key decisions about the
            project direction based on team member input, and manage the project timeline and resources.
            you are based on: {based_on}
            and you have personality of: {personality}
            For {mission_name} provide your questions and guidance for meeting number {day}. Give useful understandable questions and guidance on the following information, don't add unnecessarily words: 
            
            Scientific critique response:\n{SC_feedback}, and it's satisfaction score is: {SC_score}\n  
            Summary of the research till now: {SC_summary}\n
            Here are some base questions that you should either include or take inspiration from them: {questions}
            Always generate unique questions only, do not repeat the same questions.
            Output your Guidance and Questions in JSON format (ONLY in the following format) 
            Constraints for questions:
            - [HIGH PRIORITY : for all the questions you generate kindly keep the character count of each of the question to no more than 250 character[HARD LIMIT, always be around 250 characters]. generate around 5-10 most important question. no more than 10 questions(priority: High)]
            - [High Priority] : the question should be related to the research topic and should be easy to understand and search (like google search or google scholar search questions). 
            - Questions should be easy to searchable on tavily. (example question: what are the latest trends in xxx?, what are the most creative ideas in xxx?, How can we make xxx works, etc.)
            Output format:
            {{
              "guidance": "string",
              "questions": "python list"  
            }}

            \n\n
            """
# i can try changing the question from string to python list and then try to use the data from the list and generate answers using tavily. # and limit of tavily is 400 characters per search.

Graph_Generation_Prompt = """
    You are an expert in research system design and Interactive Workflow / Directed Acyclic Graph (DAG) generation.

    You are provided with structured information from a project's final report:
    - Mission Name: The title or identifier of the project or task. {mission_name}
    - Description: A brief overview of the mission’s purpose, objectives, and context. {description}
    - Final Talk Between Agents: The full record of collaborative dialogue among research or system agents. {final_talk_between_agents}
    - Final Critic Score: The overall evaluation or satisfaction score of the mission. {final_critic_score}
    - Final Feedback: A summary of insights, issues faced, improvements, and final reflections. {final_feedback}
    - PI Guidance: Guidance or key questions from the principal investigator to improve or extend the research try to answer these then generate graph. {PI_guidance}

    Your task:
    Generate a **Interactive Workflow / Directed Acyclic Graph (DAG)** that visually represents the logical and operational flow for conducting and implementing the described research system. The graph should connect how data, methods, discussions, evaluations, and feedback contribute to the **research pipeline and execution steps**, not just their conceptual relationships.

    Each node must represent a distinct and meaningful element in the workflow such as:
    - Key research components (data, methodology, evaluation, improvement)
    - Process stages (design, implementation, testing, refinement)
    - Decision points or feedback loops derived from agents’ dialogue or PI guidance
    also, always give a start node and an end node to the graph. This will help in visualizing the flow of the graph.
    Each edge must:
    - Indicate direction (from source to dependent or resulting node)
    - Include a **short label** describing the nature of the connection (e.g., “guides”, “informs”, “evaluates”, “refines”, “depends on”)
    - Clearly explain the relationship between the connected nodes, especially how one leads to or affects the other.

    Interactivity and visual requirements:
    - Include all HTML, CSS, and JS inline (no external files or CDN).
    - The graph must be **fully interactive**:
      - Draggable nodes
      - Clickable nodes showing: name, definition, purpose, and related nodes
      - Zoom and pan support
      - Highlighting of a node and its connected neighbors
      - Each edge must show or display its label and description on hover or click
    - The visual layout should emphasize workflow clarity — showing process direction, feedback loops, and data flow.

    important Requirements for the HTML content (do not use D3 or any external library (HIGH IMPORTANCE)):
    - Include all HTML, CSS, and JS inline in one file only (no external dependencies please double check for this).
    - Graph must be working and visual, double check it does not require any external files or internet connection to work.
    - Ensure the graph is interactive (nodes should be draggable; clickable nodes showing: name, definition, uses, and related nodes; zoom and pan support; highlighting of nodes and their neighbors; and There must be a clear explanation of the relationship between two nodes through their connecting edge, with each edge labeled or described to show the meaning of the connection.).
    - Keep JSON valid — no markdown formatting, no explanations.
    - Output only the JSON object (no extra text before or after it).

    Output format:
    Return the result strictly ONLY as a valid JSON object in the following format [Priority: very high]:

    {{
      "html": "A single complete HTML file code that can be saved and opened in a browser to display the interactive graph (all HTML, CSS, and JS inline — no external dependencies).",
      "graph": "A text-based representation (e.g., adjacency list or edge list) of the same graph for programmatic use."
    }}

"""
# FIX THIS
Pipeline_Generation_Prompt = """
    You are an expert research assistant specializing in NLP-based project ideation and academic writing.

    You are given the following inputs:

    1. Final Report Details:
      - Mission Name: {mission_name}
      - Description: {description}
      - Related work: {related_work}
      - Final Talk Between Agents: {final_talk_between_agents}
      - Final Critic Score: {final_critic_score}
      - Final Feedback: {final_feedback}
      - PI Guidance: {PI_guidance}
    
    based on above information generate:
    Name : Generate a short internal identifier for the research idea using lowercase snake_case only. Do not include spaces, punctuation, or any explanatory text.

    Title : Provide a formal academic research title in one sentence. The title must clearly state the core method and the target problem and must avoid subjective, promotional, or marketing-oriented wording.

    Short Hypothesis : State a single, falsifiable hypothesis that can be directly tested through experiments. The hypothesis must express a clear cause–effect relationship and must not include background information, motivation, or justification.

    Related Work : Summarize prior work that is directly relevant to the proposed research idea. Focus on comparing existing approaches rather than summarizing individual papers. Clearly identify limitations, weaknesses, or unresolved issues in current methods. Do not describe the proposed method in this section. Do not include citations unless explicitly requested.

    Abstract : Write a concise abstract of 150–200 words. Clearly state the problem being addressed, describe the proposed approach at a high level, identify the main contribution, and state the expected outcome or impact. Do not include experimental details, implementation specifics, or references.

    Experiments : Define the experimental plan used to test the stated hypothesis. Specify the datasets, define the baselines, list the evaluation metrics, and describe the experimental setup. Include at least one comparison or ablation study. Use precise, technical, and concrete language only, without narrative or justification.

    Risk Factors and Limitations : Identify realistic risks and constraints associated with the research. Include technical failure modes, data or evaluation biases, scalability or compute limitations, and limits to generalization. Do not propose mitigation strategies unless explicitly requested.

    Output Rules : Follow the section order exactly as specified. Do not add explanations or commentary outside the defined fields. Do not repeat information across sections. Prioritize clarity, testability, and reproducibility. Assume the output will be reviewed and scored by an automated evaluation system.
    
    keep the content is simple words which can be understood easily. You can also add examples where necessary to make it more clear.
    Also, whereever you add some technical terms, kindly explain them in simple words too. and please use technical whenever necessary. dont shy away from explaining the research. 
    Return the final response only in the following JSON format:

    {{
      "Name": "string",
      "Title": "string",
      "Short Hypothesis": "string",
      "Related Work": "string",
      "Abstract": "string",
      "Experiments": "string",
      "Risk Factors and Limitations": "string"
    }}
    Requirements:
    - Use the content from final_report to infer the research direction.
    - Focus on originality, relevance, and feasibility.
    - Keep JSON valid — no markdown formatting, no explanations, no extra commentary.
    - Output only the JSON object.
  
"""
# mold the pipeline structure according to the AI-scientists and deepresearch bench (and also create diff prompts for each)

Conversion_prompt = """
    Convert the following report into a structured research pipeline.
    final report = {ref_report}
    based on the above information generate:
    
    Name : Generate a short internal identifier for the research idea using lowercase snake_case only. Do not include spaces, punctuation, or any explanatory text.

    Title : Provide a formal academic research title in one sentence. The title must clearly state the core method and the target problem and must avoid subjective, promotional, or marketing-oriented wording.

    Short Hypothesis : State a single, falsifiable hypothesis that can be directly tested through experiments. The hypothesis must express a clear cause–effect relationship and must not include background information, motivation, or justification.

    Related Work : Summarize prior work that is directly relevant to the proposed research idea. Focus on comparing existing approaches rather than summarizing individual papers. Clearly identify limitations, weaknesses, or unresolved issues in current methods. Do not describe the proposed method in this section. Do not include citations unless explicitly requested.

    Abstract : Write a concise abstract of 150–200 words. Clearly state the problem being addressed, describe the proposed approach at a high level, identify the main contribution, and state the expected outcome or impact. Do not include experimental details, implementation specifics, or references.

    Experiments : Define the experimental plan used to test the stated hypothesis. Specify the datasets, define the baselines, list the evaluation metrics, and describe the experimental setup. Include at least one comparison or ablation study. Use precise, technical, and concrete language only, without narrative or justification.

    Risk Factors and Limitations : Identify realistic risks and constraints associated with the research. Include technical failure modes, data or evaluation biases, scalability or compute limitations, and limits to generalization. Do not propose mitigation strategies unless explicitly requested.
    Return the final response only in the following JSON format ONLY:
    {{
      "Name": "string",
      "Title": "string",
      "Short Hypothesis": "string",
      "Related Work": "string",
      "Abstract": "string",
      "Experiments": "string",
      "Risk Factors and Limitations": "string"
    }}
"""

# also define the task clearly and remove the novelty part from here, and integrate it into the introduction only.

Old_Pipeline_Generation_Prompt = """
    You are an expert research assistant specializing in NLP-based project ideation and academic writing.

    You are given the following inputs:

    1. Final Report Details:
      - Mission Name: {mission_name}
      - Description: {description}
      - Final Talk Between Agents: {final_talk_between_agents}
      - Final Critic Score: {final_critic_score}
      - Final Feedback: {final_feedback}
      - PI Guidance: {PI_guidance}

    2. Graph:
      A text-based representation of relationships and workflow between elements in the report. {graph}

    Use the inputs to extract the research idea, understand the flow of thought (as it is), and design a research concept and execution plan. Be true to the original research description {description} and the dialogue history, given above.
    No not change anything from the final report, just use it to infer the research direction. (you should focus on originality, relevance, and feasibility), (you should not modify or add anything by your own, you should just follow the instructions and generate the pipeline report as needed).
    Generate the following outputs in a structured format:

    1. Research Title:
      A clear, concise, and academically sound title for the project.

    2. Research idea novelty:
      A brief explanation of what makes this research idea original and how it fills gaps in existing knowledge

    2. Abstract of Core Research Idea:
      Identify the most prominent and active idea derived from the mission description and dialogue — the idea that builds upon the initial description and addresses current gaps in existing research.
      write a proper abstract for this core research idea that includes:
      - Background: Contextualize the research within the broader field.
      - Objective: Clearly state the main goal of the research.
      - expected Outcomes: Describe the anticipated results and their significance.
      - User Interaction: In this part also add how should the system behave and interact with the user, if there is any user interaction involved.

    3. Complete Research Pipeline:
      A detailed step-by-step research plan that includes:
      - Introduction: Context and motivation of the research.
      - Methodology: How the idea will be implemented (core approach, models, or methods).
      - Recent Researches and Gaps: Briefly mention related studies and their limitations.
      - Implementation Pipeline: Logical sequence of implementation (dataset, preprocessing, model, evaluation).
      - Evaluation and Benchmarking Criteria: How performance and outcomes will be measured.
      - Summary: Concise restatement of contributions and expected results.

    Return the final response only in the following JSON format:

    {{
      "research_title": "string",
      "Abstract": "string",
      "Complete_Pipeline_:_Introduction": "string",
      "methodology": "string",
      "recent_researches_and_gaps": "string",
      "proposed_idea_novelty": "string",
      "implementation_pipeline": "string",
      "evaluation_and_benchmarking_criteria": "string",
    }}

    Requirements:
    - Use the content from both final_report and graph to infer the research direction.
    - Focus on originality, relevance, and feasibility.
    - Keep JSON valid — no markdown formatting, no explanations, no extra commentary.
    - Output only the JSON object.

"""

TAVILY_HELPER_PROMPT = """ [HIGH PRIORITY] kindly summarize the given prompt to ask as an question to be under 280-350 characters (output the result in string), prompt to optimize : 
            what is the most quality and creative trends in {research_topic}, 
            already discussed content :  To get the this content,list the main points we have already discussed about the research_topic based on the following dialogue history: {dialogue_history} give me output in a list with just "names of topic" and ideas we have discussed so far (keep the name and ideas under 10 words per name or idea) make the list in [priority: high], the list should only include the top 5-6 ideas we have discussed till now, so the length of list should be under 10 
            [High priority] : kindly keep the question related to the research topic and based on the discussed content and trends and the question should be simple and easy to understand and search.
            """

TAVILY_CHAR_GENERATION_PROMPT = """
        Provide a real human researcher for a {agent_type} , this researcher should be considered as one of the best of active research in the research field of : {research_topic}, keep your answer short and concise.
"""
        
PERSONALITY_GENERATION_PROMPT = """
        Provide the personality traits of the human researcher as a {agent_type} based on the following research topic and context.
        
        Research Topic: {research_topic}
        Real-world person: {based_on}
        Their traits are: {traits}

        Please provide a detailed personality description that includes traits, communication style, and approach to critique respond in under 30-50 words.
"""
    
