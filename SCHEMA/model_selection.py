from prompts import PI_MODEL_SELECTION_PROMPT, PI_Model_Hire_Prompt, PI_Model_Hire_Prompt_Return, PI_Model_Hire_Prompt_CI
from llm_api import get_llm_response
import json
from model_selection_logger import log
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

Agent_Selection_Model = "gpt-5.4-mini" # This model will only be used for model selection and it will not interfare with any other process of proposal generation.

emb_model = SentenceTransformer('all-MiniLM-L6-v2')


MODEL_PROFILES = {
    "gpt-5-mini": "CAPABILITIES: High-density reasoning model optimized for structured synthesis, LaTeX technical drafting, and algorithmic logic. Designed for high-throughput inference in research pipelines. STRENGTHS: Superior instruction adherence, complex code debugging (e.g., PyTorch), literature synthesis, and structured technical writing. Efficient performance-to-latency ratio for large-scale ablation studies. WEAKNESSES: May experience reasoning drift in extremely long-context multi-pass tasks compared to frontier-scale flagship models; limited 'creative' scientific discovery. SPECIALTY: Technical analysis, research drafting, and structured reasoning.",

    "gpt-5-nano": "CAPABILITIES: Ultra-lightweight distilled model built for high-speed classification, metadata tagging, and token-efficient preprocessing. Optimized for minimal VRAM footprint and massive parallelization. STRENGTHS: Exceptional at rapid data filtering, single-pass parameter extraction from large datasets, and low-latency preprocessing pipelines. High efficiency in simple NLP workflows. WEAKNESSES: Significant drop-off in zero-shot reasoning for complex mathematics; prone to hallucinations in nuanced research critique; limited context window. SPECIALTY: Data preprocessing, tagging, and lightweight automation.",

    "gpt-4.1-mini": "CAPABILITIES: Stability-focused model specialized in multimodal document understanding (PDF/Chart analysis) and iterative refinement. Provides a reliable baseline for comparative research. STRENGTHS: Predictable behavior with rigid instruction following, strong visual interpreting for research figures/tables, and reliable software boilerplate generation. Highly consistent output for standardized tasks. WEAKNESSES: less efficient at handling modern sparse-attention long-context tasks; reduced creative logic by still creative and can bring new perspectives. SPECIALTY: Comparative baselining, document analysis, and educational support."
}

def round_robin(i:int, log_name, agent_information:dict, mixed_models:list): 
    selected_model = mixed_models[i%len(mixed_models)]
    log(prompt="Model Selection - Round Robin", log_name=log_name, iteration=i, agent_information=agent_information, selected_model=selected_model, reasoning="Selected model based on round robin selection. agent index: {i}, total models: {len(mixed_models)}")
    return selected_model

def PI_choice(iter:int, log_name, agent_information:dict, mixed_models:list, PI:any, logger = None):
    avail_model = ""
    for i,model in enumerate(mixed_models):
        avail_model += f"Model {i+1}: {model}\nCapabilities: {MODEL_PROFILES[model]}\n\n"
    prompt = PI_MODEL_SELECTION_PROMPT.format(research_topic = PI.title, description = PI.desc, Available_Models = avail_model, Agent_information = agent_information)
    retry_count = 0
    
    while retry_count < 3:
        try:
            response = get_llm_response(model_name=Agent_Selection_Model, prompt=prompt, json_mode=True, logger = logger)
            response_json = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from PI. retrying...")
            retry_count += 1
            continue

        selected_model = mixed_models[int(response_json["Selected_Model"] - 1)] # Convert back to 0-based index
        reasoning = response_json["Reasoning"]

        if selected_model not in mixed_models:
            print(f"PI selected an invalid model: {selected_model}. retrying...")
            retry_count += 1
            continue
        log(prompt, log_name, iter, agent_information, selected_model, reasoning)
        return selected_model
    print("PI failed to select a valid model after 3 attempts. Defaulting to round robin selection.")


def model_selection_option(iter:int, mode:str="round_robin", agent_information:dict = None, PI:any = None, logger = None, mixed_models:list = None):

    log_name = PI.name.replace("_Principal_Investigator_Agent","")

    if mixed_models is None:
        print("No mixed models provided, defaulting to global mixed_models list.")
    
    if mode == "round_robin":
        return round_robin(iter, log_name, agent_information, mixed_models)
    
    # elif mode == "random":
    #     import random
    #     model_choice = random.choice(mixed_models)
    #     log("Model Selection - Random", agent_information, model_choice, f"Selected model based on random selection from available models: {mixed_models}")
    #     return model_choice
    
    elif mode == "PI_Choice":
        if PI is None:
            raise ValueError("PI agent must be provided for 'PI choice' mode.")
        
        # Here we simply let PI choose model for each agent.

        return PI_choice(iter, log_name, agent_information, mixed_models, PI, logger)

    elif mode == "model_profile":
        # here we will match each agent with its appropriate model based on the agent's profile and the model's capabilities.
        # we will use sentence enbedding and semantic similarity for this purpose.
        # Fix the logic

        # Precompute model embeddings
        model_embeddings = {}
        for model_name in mixed_models:
            if model_name in MODEL_PROFILES:
                text = MODEL_PROFILES[model_name]
            else:
                print("No profile found for model:", model_name, "fetching capabilities from llm.")
                text = get_llm_response(
                    model_name="gpt-5-mini",
                    prompt = f"Describe the capabilities of research of the {model_name} model. return the answer in detail in string format",
                    logger=logger
                )
            model_embeddings[model_name] = emb_model.encode(text)

        # Compute agent embedding
        agent_embedding = emb_model.encode(agent_information["required_capabilities"])

        # Find the most similar model
        best_model = None
        best_similarity = -1

        for model_name, model_emb in model_embeddings.items():
            similarity = cosine_similarity([agent_embedding], [model_emb])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_model = model_name

        log("Model Profile Matching", log_name, iter, agent_information, best_model, f"Selected model based on profile similarity with a score of {best_similarity:.4f}")

        return best_model

    elif mode == "Voting":
        # Simulate a voting process where each model is evaluated by the PI.
        vote_list = []
        for model in mixed_models:
            vote_prompt = f"Given the agent information: {agent_information}, which model is most suitable for this agent? vote for the model number between 1 and last model number {len(mixed_models)}.\nModel information\n"
            for m in mixed_models:
                vote_prompt += f"model number {mixed_models.index(m) + 1}: {MODEL_PROFILES.get(m, 'No profile available')}\n"
            vote_prompt += "Please provide only a single integer model number as your vote. [NOTHING ELSE BUT JUST MODEL NUMBER]"
            try:
                vote = get_llm_response(model_name=model, prompt = vote_prompt, logger=logger)
                vote_list.append(int(vote))
            except ValueError:
                raise ValueError(f"Invalid vote from model {model}: {vote}.")
            
        # Count votes
        best_model_index = int(Counter(vote_list).most_common(1)[0][0]) - 1  # Convert back to 0-based index
        best_model = mixed_models[best_model_index]
        log("MODEL SELECTION - VOTING\n", log_name, iter, agent_information, best_model, f"Selected model based on voting with a score of {Counter(vote_list).most_common(1)[0][0]} \n, Total Voting: {vote_list}")
        return best_model

    elif mode == "Interview":
        # Here PI will interview each model for each agent task and then select which model it thinks is best for this agent based on the interview.
        candidate_list = []
        for model in mixed_models:

            interview_prompt = f"Given the agent information: {agent_information}, How do you think will you handle this task and how confident you are in your capabilities to truthfully carry out the responsibilites and expert discussion as the expert in this task as the role of {agent_information["name"]}\n"
            interview_prompt+= f"Describe your PLAN for completing this task, including your strategy, methodology, and execution steps you would follow for as the agent: {agent_information["name"]} for the research topic: { PI.title}. Also explain your confidence level in delivering a high-quality result and briefly justify that confidence. Return the response in STRING format."
            interview = get_llm_response(model_name=model, prompt=interview_prompt, logger=logger)
            candidate_list.append(interview)

        try:
            try:
                hire_prompt = PI_Model_Hire_Prompt.format(research_topic = PI.title, research_desc = PI.desc, agent_name = agent_information["name"], agent_expertise = agent_information['expertise'], agent_goal = agent_information['Goal'], agent_role = agent_information['Role'], agent_req_cap = agent_information["required_capabilities"])
            except KeyError as e:
                print(f"Missing key in agent information: {e}. Sending complete model information.")
                hire_prompt = PI_Model_Hire_Prompt_CI.format(research_topic = PI.title, research_desc = PI.desc, agent_information = agent_information)
                
            for i, reason in enumerate(candidate_list):
                hire_prompt += f"Model {i+1} :\tModel Reasons: {reason}\n Model Capabilities: {agent_information["required_capabilities"]}\n-------------------------------------\n"
            hire_prompt+= PI_Model_Hire_Prompt_Return
            retry_count = 0
            while retry_count < 3:
                try:
                    response_llm = get_llm_response(model_name = Agent_Selection_Model, prompt = hire_prompt, json_mode=True, logger = logger)
                    response = json.loads(response_llm)
                    hired_model = int(response["Model"])
                    reasoning = response["Reasoning"]
                    selected_model = mixed_models[hired_model-1]
                    log(hire_prompt, log_name, iter, agent_information, selected_model, f"\n{"-"*30}\n Selection Reasoning:\n{reasoning}")
                    return selected_model
                except:
                    print("Failed in parsing Json output, Retrying !!!")
                    retry_count+=1
        
        except Exception as e:
            print(f"PI Failed to choose a proper model for {agent_information["name"]}, exception: ", e)


    else:
        raise ValueError(f"Unsupported model selection mode: {mode}")
    
