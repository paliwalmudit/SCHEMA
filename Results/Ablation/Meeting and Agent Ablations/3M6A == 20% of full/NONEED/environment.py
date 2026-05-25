class MissionEnvironment:
    def __init__(self, mission_name: str):
        if mission_name not in MISSION_PLANS:
            raise ValueError(f"Mission '{mission_name}' not found.")

        self.mission_name = mission_name
        self.plan = MISSION_PLANS[mission_name]
        self.desc=self.plan["desc"]
        self.duration = self.plan["duration"]
        self.crew_size = self.plan["crew_size"]

    def get_event_for_day(self, day: int):
        """Returns a scheduled event, or randomly decides between a random or routine event."""
        return {
            "day": day,
            "type": "routine",
            "description": "A day of routine tasks: system checks, scientific experiments, and habitat maintenance."
        }

# Mission data database
# update the missions plans as the research you want to do.
MISSION_PLANS = {
    "Pets-in-horror-game-for-mood-upliftment": {
        "duration": 4, #number of meetings.
        "crew_size": 6,
        "desc": """Topic = A game with a horror environment consisting of a virtual pet as a companion of user. description = Implementation of this idea has two pathways currently: 1)Evaluating user's emotions through natural text through a chat, and classifying them into positive and negative emotions. Visuals are generated as game characters, negative emotions battle against the virtual pet, with the pet defeating them, giving the user a feeling that they defeated their negative emotion. While positive emotions' characters interact with the pet in a mini game. 2) A 2-d maze game with a horror castle environment, where evil spirits try to harm the user. These evil spirits are negative emotions, which cause different negative effects on being interacted with. But the user is assisted by a pet in a ghost form, which guides user through the maze, helping them find keys and avoid harmful entities. Come up with the best approach among these, and refine them if possible (to make them more interactive and such that they are effective in mood upliftment). Find relevant papers and open source games that can be used for implementation.""",
    },
    "Poker_Face_Detection":{
        "duration": 4, #number of meetings.
        "crew_size": 6,
        "desc":"""Topic = Poker Face Detection using Multimodal Large Language Models (with images captured by camera), Galvanic skin response, eye tracking and Heart Rate Variation. description = The model will use facial detection powered by images and facial landmarks as well GSR(galvanic skin response), HRV(heart rate variations) and Eye Tracking. The facial expression recognition part will help to detect when the person has a neutral expression and using that information combined with the GSR, HRV and Eye Tracking should be able to tell a poker face.""",
    },
    "Poker_Face_Detection_without_HRV":{
        "duration": 4, #number of meetings.
        "crew_size": 6,
        "desc":"""Topic = Poker Face Detection using Multimodal Large Language Models (with images captured by camera) and pattern recognition. description = The model will use facial detection powered by images and facial landmarks as well as pattern recognition. The facial expression recognition part will help to detect when the person has a neutral expression, the pattern recognition side will help to detect differences between a real neutral face and a poker face.""",
    },
    "FER-Driven_Visual_Novel":{
        "duration": 4, #number of meetings.
        "crew_size": 6,
        "desc":"""Topic = Emotion-Driven Narrative Adaptation in Horror Visual Novels Using Multimodal Emotion Recognition and Large Language Models. description = This research aims to develop an affect-aware interactive horror visual novel system in which narrative content can be dynamically adapted according to the player’s emotional state. By integrating facial emotion recognition (FER) with multimodal physiological signal analysis, including galvanic skin response (GSR) and heart rate variability (HRV), the system estimates players’ emotional responses during gameplay—particularly fear and engagement—and uses this information to dynamically adjust NPC personality traits, narrative branching, dialogue generation, and overall pacing. In the initial implementation stage, the study will first construct a visual novel text prototype with explicit horror elements and narrative intent. In parallel, the outputs of FER, GSR, and HRV will be incorporated into a large language model (e.g., the GPT family) as prompt-based or structured inputs to regulate emotional intensity, enabling the generation of emotionally consistent and narratively coherent adaptive story modifications.""",
    },
    "Predictive_AI_use_in_Rollback_Netcode":{
        "duration": 4, #number of meetings.
        "crew_size": 6,
        "desc":"""Topic = Rollback Netcode in Fighting Games. description = Rollback netcode is a technique used in online multiplayer games, particularly fighting games, to reduce the perceived latency between players. It works by predicting player inputs and "rolling back" the game state if the predictions are incorrect when the actual inputs arrive. This approach helps maintain a smooth gameplay experience even with high latency connections. Kindly help me in coming up with more research oriented idea and a research plan for this topic. The goal is to create a comprehensive research plan that outlines the objectives, methodology, and expected outcomes of the study.""",
    },
    "LLM_Plays_Pokemon":{
        "duration": 4, #number of meetings.
        "crew_size":6,
        # "desc":"""Topic: LLM plays pokemon Red. description = We will use an LLM as the sole agent to play Pokémon Red. The LLM will handle all decisions, including Pokémon selection, navigation, exploration, battles, leveling, and overall game progression. The objective is to complete the game as quickly as possible while achieving reliable and accurate navigation. We will not use ROMs or emulators. Instead, the system will run on a physical game cartridge and console, with the game screen captured in real time. The LLM will receive only visual input from this screen capture and will generate actions based on that input. Your task is to propose efficient methods to enable this setup. This includes designing approaches for perception, navigation, memory, and decision-making. One initial idea is to apply vision-based SLAM to construct and maintain a map of the game environment, but you are encouraged to critically evaluate this approach and propose alternatives if more effective. You may either build on the SLAM concept or introduce different navigation strategies and a research plan for enabling an LLM to complete the game reliably end to end."""
        "desc":"""Topic: LLM plays pokemon Red. description = We will use LLM as the sole agent which plays pokemon red (no human intervention), LLM will be incharge of all the things from pokemon selection, navigation, exploration, level-up, game completion. the goal is to complete the game as fast as possible by llms and make the navigation perfect. We will not be using any roms, instead we will be using real game cartraige with SNES and capture the game screen, then we will use those visual input to ask llm to play the game. your task is to help us in coming up with effecient ideas on how to do this task and how to make the exploration in the game Pokemon Red possible, i have a simple idea of using "SLAM" with vision for this (something new), you can either work on this or come-up with different ideas for navigation."""
    },
    "Attention_is_all_you_need":{
        "duration": 3, #number of meetings.
        "crew_size":6,
        "desc":"""Topic: Attention is all you need, it is a research about attention mechanism and how to implement it in transformer architecture and its uses in language related tasks. description = Idea: The core research idea is that sequence modeling does not require step-by-step processing and that language can be understood more effectively by allowing every token to directly interact with every other token through attention, thereby modeling global dependencies in a single operation rather than accumulating context through recurrence or convolution. Core description: The idea is to build a model where attention is the primary mechanism for learning relationships between words, enabling the system to assign relevance between all parts of a sequence at once, use explicit positional information to preserve order, and exploit parallel computation to train faster and scale better than traditional sequential architectures."""
    },
    "Null_shot_prompting":{
        "duration": 3, #number of meetings.
        "crew_size":6,
        "desc":"""Topic: Null-shot prompting, it is a research about how to prompt a large language model to perform tasks without any examples or fine-tuning. description = Null-shot prompting exploits hallucination in large language models (LLMs) by instructing LLMs to utilize information from the "Examples" section that never exists within the provided context to perform a task. Idea: The core research idea is that a sufficiently large language model trained on diverse natural language can perform new tasks without task-specific training, simply by conditioning on instructions and examples provided in text, because task understanding can emerge implicitly from scale and exposure rather than explicit supervision. Core description: The idea is to treat task execution as a continuation of language modeling, where the model interprets natural language prompts as task specifications and uses in-context examples to infer what is required, enabling zero-shot (often informally called null-shot) and few-shot task performance without updating model parameters."""
    },
    "Agent_Drift":{
        "duration": 3, #number of meetings.
        "crew_size":6,
        "desc":"""Topic: Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions. description = Multi-agent Large Language Model (LLM) systems have emerged as powerful architectures for complex task decomposition and collaborative problem-solving. However, their long-term behavioral stability remains largely unexamined. This study introduces the concept of agent drift, defined as the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences. We present a comprehensive theoretical framework for understanding drift phenomena, proposing three distinct manifestations: semantic drift (progressive deviation from original intent), coordination drift (breakdown in multi-agent consensus mechanisms), and behavioral drift (emergence of unintended strategies).
We introduce the Agent Stability Index (ASI), a novel composite metric framework for quantifying drift across twelve dimensions, including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates. Through simulation-based analysis and theoretical modeling, we demonstrate how unchecked agent drift can lead to substantial reductions in task completion accuracy and increased human intervention requirements.
We propose three mitigation strategies: episodic memory consolidation, drift-aware routing protocols, and adaptive behavioral anchoring. Theoretical analysis suggests these approaches can significantly reduce drift-related errors while maintaining system throughput. This work establishes a foundational methodology for monitoring, measuring, and mitigating agent drift in production agentic AI systems, with direct implications for enterprise deployment reliability and AI safety research."""
    },
}