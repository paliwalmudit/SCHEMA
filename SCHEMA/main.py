import copy
from collections import deque
from tqdm import tqdm
from environment import MissionEnvironment
from agents import SCAgent, PIAgent
import json
from evaluation import start_eval
import os

# Restored: The list of models for the 'mixed' crew option
MIXED_MODELS = ["gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini"]
# , "gpt-4.1-mini", "gemma3", "qwen3"

def run_simulation(base_env: MissionEnvironment, iteration: int, crew: list,  sc_model: str, PI: PIAgent):
    """
    Runs one full simulation for a given mission with a pre-initialized crew.
    The crew_model argument is used for correct file naming.
    """
    env = base_env
    research_topic = env.topic # Changed to topic for easier automation
    
    # Initialize the Scientific Critic (SC)
    sc = SCAgent(sc_model, env.topic, env.title, env.desc , iteration)
    print("Scientific Critic is ACTIVE.")

    dialogue_history = deque(maxlen=12)  # Store recent dialogue history for context
    feedback="Initial No feedback"
    pi_guidance="Initial No guidance"
    pi_questions=["Initial No questions"]
    final_report = dict()

    print(f"\n--- Starting Simulation: {research_topic} | Iteration: {iteration} | Meetings: {env.duration} ---")

    for day in tqdm(range(1, env.duration + 1), desc="Simulating Mission Days"):
        
        daily_dialogue = []
        
        for agent in crew:
            history_str = "\n".join(dialogue_history) if dialogue_history else "Initial No dialogue history"
            response = agent.respond(history_str, feedback, pi_guidance, pi_questions)
            dialogue_line = f"{agent.name}: \"{response}\""
            
            dialogue_history.append(dialogue_line)
            daily_dialogue.append(dialogue_line)
            
        # get scientific critic feedback

        critic = sc.evaluate_and_summarize(daily_dialogue)
        if critic:
            feedback=critic["feedback"]
            # print(day,"\n per meeting response:", daily_dialogue, "\n all responses till this meeting:", dialogue_history,"\n final critic:",critic)

        else:
            critic={
                "satisfaction_score":0.0,
                "feedback":"No feedback",
                "summary":"No summary"
            }
            feedback="No feedback"
            print("Critic failed to provide feedback:",feedback)
        
        # PI questions and guidance part.

        guidance_questions = PI.questions_and_guidance(critic, day)
        if guidance_questions:
            guidance_questions=json.loads(guidance_questions)
            pi_guidance=guidance_questions["guidance"]
            pi_questions=guidance_questions["questions"]
        else:
            guidance_questions={
                "guidance":"No guidance",
                "questions":"No questions"
            }
            pi_guidance="No guidance"
            pi_questions="No questions"
            print("PI failed to provide guidance and questions:", guidance_questions)

        # Final report generation at the end of the research meetings.

        if day == env.duration:
            print(f"\n--- Simulation Complete: {research_topic} | Iteration: {iteration} ---\n") 
            final_report = {
                "final_talk_between_agents": "\n Agent output: \n".join(dialogue_history),
                "final_critic_score": critic["satisfaction_score"],
                "final_feedback": feedback,
                "PI_guidance": guidance_questions
            }
        
    # PI Generates the final report. 
    PI.pipeline_generator(final_report)

    # for evaluation
    start_eval(research_topic.lower())


def get_research_topics_from_folder(folder_path: str):
    topics = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            topic_name = os.path.splitext(file)[0]
            topics.append(topic_name)
    return topics


if __name__ == "__main__":
    MISSIONS_FOLDER = "INPUT"   # folder containing files
    CREW_MODEL = "mixed"       # or any specific model
    SC_MODEL = "gpt-5-mini"
    PI_MODEL = "gpt-5-mini"
    ITERATIONS = 1 # [number of rounds of meetings] [always set to 1 to only have 1 round of meeting.]  
    model_seclection_mode = "Interview"  #["Interview", "Voting", "model_profile", "PI_Choice", "round_robin"]
    
    research_topics = get_research_topics_from_folder(MISSIONS_FOLDER)

    if not research_topics:
        raise RuntimeError("No mission files found.")

    for topic in research_topics:
        print(f"\n=== Running mission: {topic} ===\n")
        if os.path.exists(f"results/eval_results/eval_{topic}_VL.json"):
            print("moving to next !!!")
            continue

        print(f"\n=== Running mission: {topic} ===\n")

        print("PI MODEL:", PI_MODEL)
        print("SC MODEL:", SC_MODEL)
        print("Crew MODEL:", CREW_MODEL)
        print("Model Selection Mode:", model_seclection_mode)

        base_env = MissionEnvironment(topic)

        PI = PIAgent(
            mission_name = base_env.topic,
            title = base_env.title,
            desc = base_env.desc,
            model = PI_MODEL,
            iteration = ITERATIONS,
            model_seclection_mode=model_seclection_mode
        )

        print("--- PI ACTIVE ---\n")

        print("--- Initializing base agents for all iterations... ---")

        base_crew = PI.initialize_crew(
            crew_size = base_env.crew_size,
            crew_model = CREW_MODEL,
            available_models = MIXED_MODELS,
            iteration=ITERATIONS
        )

        print("--- Base crew initialized. Starting simulations. ---")

        crew_for_this_run = copy.deepcopy(base_crew)
        
        for i in range(1, ITERATIONS + 1):

            run_simulation(
                base_env=base_env,
                iteration=i,
                crew=crew_for_this_run,
                sc_model=SC_MODEL,
                PI=PI
            )

    print("\n--- All automated simulations completed ---\n")

