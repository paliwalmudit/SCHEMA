import json
import time
from llm_api import get_llm_response
from prompts import INITIAL_EXPERT_GENERATION_PROMPT, CREW_RESPONSE_PROMPT, CRITIQUE_PROMPT, PI_Prompt, Pipeline_Generation_Prompt # Graph_Generation_Prompt
from prompts_logger import PromptLogger
from result_logger import save_project_files
from tavily_helper import generate_tavily_answers, generate_personality, generate_tavily_questions, generate_based_on, get_related_work_tavily

class ExpertAgent: # Expert Agent
    def __init__(self, name: str, expertise: str, goal: str, role: str, model: str, mission_name: str, iteration: int):
        self.name = name
        self.expertise = expertise
        self.goal=goal 
        self.role=role
        self.model = model
        self.mission_name = mission_name
        self.iteration = iteration
        self.logger = PromptLogger(mission_name, iteration, self.name, self.model)

    def respond(self, mission_name: str, dialogue_history: str, feedback: str, pi_guidance, pi_questions) -> str:

        # getting tavily responses here
        tavily_response = generate_tavily_answers(mission_name, self.name, pi_questions)

        prompt = CREW_RESPONSE_PROMPT.format(
            agent_name=self.name,
            research_topic=mission_name,
            feedback_summary=feedback,
            role=self.role,
            expertise=self.expertise,
            goal=self.goal,
            dialogue_history=dialogue_history,
            guidance=pi_guidance,
            questions=pi_questions,
            tavily_answers=tavily_response
        )
        response = get_llm_response(self.model, prompt, temperature=0.75, logger=self.logger)
        return response
    
class SCAgent: # Scientific Critique Agent
    def __init__(self, model: str, mission_name: str, iteration: int, satisfaction_threshold: float = 7.5):
        based_on = generate_based_on(mission_name, "SC")
        self.model = model
        self.name= mission_name + "_Scientific_Critique_Agent"
        self.logger = PromptLogger(mission_name, iteration, self.name, self.model)
        self.satisfaction_threshold = satisfaction_threshold
        self.personality = generate_personality(mission_name, based_on, "Scientific Critique Agent")
        self.BasedOn = based_on


    def evaluate_and_summarize(self, expert_outputs: list) -> dict:
        """
        Reviews Agents outputs, rates satisfaction, and provides feedback for improvement if needed.
        Returns dict with satisfaction_score and feedback.
        """
        critique_prompt = CRITIQUE_PROMPT.format(based_on=self.BasedOn, personality=self.personality)

        for i, output in enumerate(expert_outputs, start=1):
            critique_prompt += f"Expert {i} Output:\n{output}\n\n"

        critique_prompt += """
            Now respond strictly ONLY in JSON format, never give the score of 0.0 or 0, but be strict while scoring [Priority: High, be very strict and adhere to your personality traits while giving scores, but also remember to be fair and don't just give bad scores in the name of strictness], 
            dont give everyone more than 0.9 too unless you think they really did a good job and they deserve it. Please provide a satisfaction score between 1.0 and 10.0. 
            The structure should be as follows:
            {{
              "satisfaction_score": float (1.0 - 10.0),
              "feedback": "What should experts improve and a solid critic(if score < 7.5), or a general feedback on what the agents can do to improve more(if score > 7.5)?",
              "Summary": "A concise summary of the research progress and findings so far."
            }}
        """

        response = get_llm_response(
            self.model, critique_prompt, temperature=0.4, json_mode=True, logger=self.logger
        )

        try:
            critique_result = json.loads(response)
        except json.JSONDecodeError:
            critique_result = {
                "satisfaction_score": 0.0,
                "feedback": "Could not parse critique JSON response.",
                "Summary": "No summary available due to parsing error."
            }
            print("Parsing error in Scientific Critique response.")

        score = critique_result.get("satisfaction_score", 0.0)
        if score >= self.satisfaction_threshold:
            print(f"✅ Scientific Critique satisfied for now, threshold is now increased (score={score:.2f})")
            self.satisfaction_threshold=self.satisfaction_threshold+0.5
        else:
            print(f"⚠️ Scientific Critique not satisfied (score={score:.2f}) — feedback given.")
            # print(f"Feedback: {critique_result.get('feedback', '')}")

        return critique_result

class PIAgent: # Principal Investigator Agent
    def __init__(self, mission_name: str, desc: str, model: str, iteration: int):
        based_on = generate_based_on(mission_name, "PI")
        self.name = mission_name + "_Principal_Investigator_Agent"
        self.desc=desc
        self.model = model
        self.personality = generate_personality(mission_name, based_on, "Principal Investigator Agent") 
        self.BasedOn = based_on
        self.logger = PromptLogger(mission_name,iteration, self.name, self.model)

    def initialize_crew(self, crew_size: int, crew_model: str, available_models: list, iteration: int) -> list:
        mission_name=self.name
        print(f"Generating crew for {mission_name}...")
        # Use gpt-4o-mini for crew generation as a consistent baseline utility
        generation_model = "gpt-5-mini"
        prompt = INITIAL_EXPERT_GENERATION_PROMPT.format(based_on=self.BasedOn, personality=self.personality, research_topic=mission_name, number_of_experts=crew_size)
        
        generation_logger = PromptLogger(mission_name, iteration, "AgentGeneration", generation_model)
        
        while True:
            try:
                response_json = get_llm_response(generation_model, prompt, temperature=0.5, json_mode=True, logger=generation_logger)
                crew_data = json.loads(response_json)["agents"]   
                
                crew = []
                for i, member_data in enumerate(crew_data):
                    if crew_model == "mixed":
                        model_for_agent = available_models[i % len(available_models)]
                    else:
                        model_for_agent = crew_model
                    
                    crew.append(ExpertAgent(
                        name=member_data["name"],
                        expertise=member_data["expertise"],
                        goal=member_data["Goal"], 
                        role=member_data["Role"], 
                        model=model_for_agent,
                        mission_name=mission_name,
                        iteration=iteration
                    ))
                print("Agents generated successfully.")
                return crew
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error generating agents, Error: {e}")
                time.sleep(5)

    def questions_and_guidance(self, SC_response, day: int, mission_name: str):
        # Generate questions and guidance based on Scientific Critique response
        SC_feedback=SC_response["feedback"]
        SC_score=SC_response["satisfaction_score"]
        SC_summary=SC_response["Summary"]
        questions = generate_tavily_questions(self.name, SC_summary, SC_feedback, self.logger, self.BasedOn, self.personality)
        prompt = PI_Prompt.format(based_on=self.BasedOn, personality = self.personality , SC_feedback=SC_feedback, SC_score=SC_score, SC_summary=SC_summary, questions=questions, mission_name=mission_name, day=day)
        guidance_response = get_llm_response(
            self.model, prompt, temperature=0.6, logger=self.logger
        )
        return guidance_response
    
    def pipeline_generator(self, final_report):
        """
        Generates visualization and documentation of the research pipeline based on the final report.
        """
        print("\n--- Pipeline Generator Invoked ---")
        print("\n--- Working on visualization and documentation of the research pipeline... ---")
        # graph_prompt = Graph_Generation_Prompt.format(
        #     mission_name = final_report["mission_name"],
        #     description = final_report["description"],
        #     final_talk_between_agents = final_report["final_talk_between_agents"],
        #     final_critic_score = final_report["final_critic_score"],
        #     final_feedback = final_report["final_feedback"],
        #     PI_guidance = final_report["PI_guidance"],
        # )
        # Graph = get_llm_response(self.model, graph_prompt, temperature=0.5, logger=self.logger)
        
        # if Graph:
        #     # print(Graph)
        #     Graph = json.loads(Graph)
        #     html_content = Graph["html"]
        #     graph_content = Graph["graph"]
        
        # else:
        #     print("Graph generation failed.")
        #     return
        
        # pipeline_prompt = Pipeline_Generation_Prompt.format(
        #     mission_name = final_report["mission_name"],
        #     description = final_report["description"],
        #     final_talk_between_agents = final_report["final_talk_between_agents"],
        #     final_critic_score = final_report["final_critic_score"],
        #     final_feedback = final_report["final_feedback"],
        #     PI_guidance = final_report["PI_guidance"],
        #     graph=graph_content
        # )

        # ONLY FOR TESTING PURPOSES
        prev_studies = get_related_work_tavily(final_report["mission_name"])
        html_content = "<html><body><h1>Pipeline Visualization Placeholder</h1></body></html>"
        pipeline_prompt = Pipeline_Generation_Prompt.format(
            mission_name = final_report["mission_name"],
            description = final_report["description"],
            related_work = prev_studies,
            final_talk_between_agents = final_report["final_talk_between_agents"],
            final_critic_score = final_report["final_critic_score"],
            final_feedback = final_report["final_feedback"],
            PI_guidance = final_report["PI_guidance"],
        )
        pipeline_data = get_llm_response(self.model, pipeline_prompt, temperature=0.5, logger=self.logger)
        
        # confirmation message.
        print("Final report has been processed by the pipeline generator.")
        # Save the generated files
        save_project_files(html_content, pipeline_data, self.name)
        
        return pipeline_data