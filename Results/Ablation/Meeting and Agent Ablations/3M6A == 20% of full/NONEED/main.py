import argparse
import copy
from collections import deque
from tqdm import tqdm
from environment import MissionEnvironment
from agents import SCAgent, PIAgent
import json
from evaluation import start_eval
from llm_api import AVAILABLE_MODELS

# Restored: The list of models for the 'mixed' crew option
MIXED_MODELS = ["gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini"]
# , "gpt-4.1-mini", "gemma3", "qwen3"
def run_simulation(research_topic: str, iteration: int, crew_model: str, crew: list,  sc_model: str, PI: PIAgent):
    """
    Runs one full simulation for a given mission with a pre-initialized crew.
    The crew_model argument is used for correct file naming.
    """
    env = MissionEnvironment(research_topic)
    # logger = SimulationLogger(mission_name, crew_model, iteration)
    
    # Initialize the Scientific Critic (SC)
    sc = SCAgent(sc_model, env.mission_name, iteration)
    print("Scientific Critic is ACTIVE.")

    dialogue_history = deque(maxlen=12)  # Store recent dialogue history for context
    feedback="Initial No feedback"
    pi_guidance="Initial No guidance"
    pi_questions=["Initial No questions"]
    final_report = dict()

    print(f"\n--- Starting Simulation: {research_topic} | Iteration: {iteration} | Meetings: {env.duration} ---")

    for day in tqdm(range(1, env.duration + 1), desc="Simulating Mission Days"):
        # event = env.get_event_for_day(day)
        
        daily_dialogue = []
        
        for agent in crew:
            history_str = "\n".join(dialogue_history) if dialogue_history else "No recent conversation."
            response = agent.respond(env.mission_name, history_str, feedback, pi_guidance, pi_questions)
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

        guidance_questions = PI.questions_and_guidance(critic, day, env.mission_name)
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
                "mission_name": env.mission_name,
                "description": env.desc,
                "final_talk_between_agents": "\n Agent output: \n".join(dialogue_history),
                "final_critic_score": critic["satisfaction_score"],
                "final_feedback": feedback,
                "PI_guidance": guidance_questions
            }
        
    # PI.pipeline_generator(final_report)
    report = PI.pipeline_generator(final_report)
    # for evaluation
    start_eval(report, research_topic.lower())
    # logger.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Virtual ICE Lab multi-agent LLM simulation.")
    
    model_choices = list(AVAILABLE_MODELS.keys())
    mission_choices = ["Agent_Drift","Null_shot_prompting","Attention_is_all_you_need","LLM_Plays_Pokemon","Predictive_AI_use_in_Rollback_Netcode","FER-Driven_Visual_Novel","Pets-in-horror-game-for-mood-upliftment", "Poker-Face-Detection-using-Multimodal-Large-Language-Models-and-Heart-Rate-Variation", "Poker_Face_Detection", "Poker_Face_Detection_without_HRV"]

    parser.add_argument("--crew-model", type=str, required=True, choices=model_choices + ["mixed"], help="LLM for crew agents.")
    parser.add_argument("--sc-model", type=str, required=True, choices=model_choices, help="LLM for the SC agent.")
    parser.add_argument("--pi-model", type=str, required=False, choices=model_choices, help="LLM for the PI agent.")
    parser.add_argument("--topic", type=str, required=True, choices=mission_choices, help="Research idea to simulate.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of full Simulations to run.")

    args = parser.parse_args()

    # from analysis import analyze_results
    print("\n--- Research Simulation Starting ---\n")
    print("--- PI ACTIVE ---\n")
    print("--- Initializing base agents for all iterations... ---")
    base_env = MissionEnvironment(args.topic)
    pi_model=args.pi_model if args.pi_model else "gpt-5-mini"
    PI=PIAgent(base_env.mission_name, base_env.desc, pi_model, args.iterations)
    base_crew = PI.initialize_crew(
        base_env.crew_size,
        args.crew_model,
        MIXED_MODELS,
        iteration=args.iterations
    )
    print("--- Base crew initialized. Starting simulations. ---")
    # duration is the number of meetings in test mission. (ex. duration is 12 then number of meetings is also 12 in test mission.)
    # iteration is kind of number of days/ Rounds (like if iteration is 2 then there will be 2 * 12 meetings in test mission.)
    
    crew_for_this_run = copy.deepcopy(base_crew)
     
    for i in range(1, args.iterations + 1):
        crew_for_this_run = copy.deepcopy(base_crew)
        
        run_simulation(
            research_topic=args.topic,
            iteration=i,
            crew=crew_for_this_run,
            crew_model=args.crew_model,  # Pass the original crew_model argument for logging
            sc_model=args.sc_model,
            PI=PI
        )

    if args.iterations > 0:
        print("\n--- All simulations are completed... ---")
        # analyze_results(
        #     results_dir="results",
        #     mission_name=args.mission,
        #     crew_model=args.crew_model
        # )


#python main.py --crew-model gpt-4.1-nano--sc-model gpt-4.1-nano --topic test