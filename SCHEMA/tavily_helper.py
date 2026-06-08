from llm_api import get_llm_response
from prompts import PERSONALITY_GENERATION_PROMPT, TAVILY_CHAR_GENERATION_PROMPT, TAVILY_HELPER_PROMPT
from tavily import TavilyClient
from tavily_logger import log_tavily
from datetime import date    
import os
from dotenv import load_dotenv 

load_dotenv()
# ONLY IN CASE OF MULTIPLE KEYS, OTHERWISE WE CAN JUST USE THE SINGLE KEY DIRECTLY.
# # getting the tavily api key
# # Load all keys
# TAVILY_KEYS = [
#     os.getenv("TAVILY_API_KEY_1"),
#     os.getenv("TAVILY_API_KEY_2"),
#     os.getenv("TAVILY_API_KEY_3"),
#     os.getenv("TAVILY_API_KEY_4"),
# ]

# # remove empty values
# TAVILY_KEYS = [k for k in TAVILY_KEYS if k]

# if not TAVILY_KEYS:
#     raise Exception("No Tavily API keys found")

# current_key_index = 0

# tavily_api_key = TAVILY_KEYS[0]  # Start with the first key

tavily_api_key = os.getenv("TAVILY_API_KEY")  # Use a single key directly

tavily_client = TavilyClient(api_key=tavily_api_key)


# We need to have simpler question otherwise tavily will just answer "None".

# to get the static var for logging (might change later/ might shift it to the agent class.)
static_var=0
static_ans=dict()
prev_questions=set()
things_discussed = ""


def switch_api():
    global current_key_index
    global tavily_client
    current_key_index += 1

    if current_key_index >= len(TAVILY_KEYS):
        raise Exception("All Tavily API keys exhausted")

    tavily_client = TavilyClient(api_key=TAVILY_KEYS[current_key_index])

    print(f"[INFO] Switched to Tavily key #{current_key_index} total keys = 0 to {len(TAVILY_KEYS)-1}")


def api_exhausted(e):
    error_text = str(e).lower()

    print(f"[ERROR] {e}")

    # Detect quota/rate-limit exhaustion
    if (
        "429" in error_text
        or "rate limit" in error_text
        or "quota" in error_text
        or "credits" in error_text
        or "too many requests" in error_text
    ):

        print("[INFO] Current Tavily key exhausted")

        attempts += 1

        # switch to next key
        switch_api()

    else:
        # unknown error -> don't rotate blindly
        raise e

    raise Exception("All Tavily API keys failed")

def generate_tavily_questions(research_topic: str, dialogue_history: str, feedback: str, self_logger, based_on, personality) -> str:
    """
    Generates questions to be sent to the Tavily client based on the research topic,
    dialogue history, and feedback from the scientific critic.
    """
    global static_var, things_discussed
    # use tavily to get real world data and ask llm to generate 3 question based on that data. then use llm as the judge to pick the best question.
    # (done)
    # Construct the prompt for generating Tavily questions

    query_prompt = TAVILY_HELPER_PROMPT.format(research_topic=research_topic, dialogue_history=dialogue_history)
    query_prompt = get_llm_response("gpt-4.1-nano", query_prompt, temperature=0)
    things_discussed = query_prompt
    if len(query_prompt)>400:
        query_limitter = 390
        while len(query_prompt)>390:
            query_prompt = query_prompt.split()[:query_limitter]
            query_prompt = " ".join(query_prompt)
            query_limitter -= 10
            
    # query_resp = tavily_client.search(query= query_prompt, include_answer=True)
    # To get the data excluding the most recent data of this month. (for evaluation purposes)
    try:
        query_resp = tavily_client.search(query= query_prompt, include_answer=True, end_date="2025-10-01") # for research purposes we want to exclude the most recent data, so we set end_date.
    except Exception as e:
        api_exhausted(e)
        
    resp= query_resp.get('answer', '')
    log_tavily(query_prompt, query_resp, static_var, research_topic)
    static_var+=1

    # to generate a creative and quality question.
    # we can just use the question too but then it might loose the creative aspect of PI.

    prompt = """you are {based_on} with scientific traits {personality} is tasked to Generate 7 creative and informative questions on research topic {research_topic} based on the following context: {dialogue_history} Feedback: {feedback} real-world trends and data: {resp}""".format(based_on=based_on, personality=personality, research_topic=research_topic, dialogue_history=dialogue_history, feedback=feedback, resp=resp)
    response_questions = get_llm_response("gpt-5-nano", prompt, temperature=0)
    judge_promot = """which of the following questions is the most creative and informative for gathering information on the research topic: {research_topic}
        list of questions: {response_questions}
        respond with only the best 3 questions.
        answer in string only.
    """.format(research_topic=research_topic ,response_questions=response_questions)
    response=get_llm_response("gpt-5-nano", judge_promot, temperature=0, logger= self_logger)
    
    return response

def generate_based_on(log_name: str, research_topic: str, agent_type: str) -> str:
    global static_var
    if agent_type == "SC":
        prompt = TAVILY_CHAR_GENERATION_PROMPT.format(
            agent_type="SC",
            research_topic=research_topic,
        )   
    else:
        prompt = TAVILY_CHAR_GENERATION_PROMPT.format(
            agent_type="PI",
            research_topic=research_topic,
        )

    try:
        response = tavily_client.search(query=prompt, include_answer=True, end_date="2025-10-01")
    except Exception as e:
        api_exhausted(e)
        
    answer = response.get('answer', '')
    # print("\nquestion: \n",prompt,"\nbased on:\n", answer)
    log_tavily(prompt, response, static_var, log_name)
    static_var+=1

    return answer

def generate_personality(log_name: str, research_topic: str, based_on: str, description: str, agent_type: str) -> str:
    """
    Generates a personality description for the agent based on the research topic
    and the context provided.
    """
    global static_var

    question = """
        Generate a scintific personality traits description of this real-world scientist : {based_on}
    """.format(based_on=based_on)
    question = question.split()[:350]
    question = " ".join(question)

    try:
        agent_traits = tavily_client.search(query=question, include_answer=True, end_date="2025-10-01")
    except Exception as e:
        api_exhausted(e)
    answer = agent_traits.get('answer', '')
    # print("\nquestion: \n",question,"\ntavily response:\n", answer)
    log_tavily(question, agent_traits, static_var, log_name)
    static_var+=1


    prompt = PERSONALITY_GENERATION_PROMPT.format(
        research_topic=research_topic,
        based_on=based_on,
        agent_type = agent_type,
        description = description,
        traits = answer
    )

    personality = get_llm_response("gpt-5-nano", prompt, temperature=0)
    
    return personality


def generate_tavily_answers(research_topic, agent_name , pi_questions: list = ["no questions so help in question generation"]) -> str:
    """
    Sends the generated questions to the Tavily client and retrieves answers.
    This is a placeholder function; actual implementation will depend on Tavily's API.
    """
    # imposing word limit on question while we are using tavily. (word limit of 350-370 words)

    global static_var, static_ans, prev_questions
    answer_list= []
    response = ""
    # question="""As a {agent_name}, answer this question as based on {research_topic}:""".format(agent_name=agent_name, research_topic=research_topic)
    if "Initial No questions" in pi_questions or "no questions so help in question generation" in pi_questions or "No questions" in pi_questions:
        return ["continue, and try to be better."]
    try:
        for i in pi_questions:
            # question += i
            if i in prev_questions:
                answer = static_ans[i]
            else:
                if len(i)>400:
                    query_limitter = 390
                    while len(query_prompt)>390:
                        query_prompt = query_prompt.split()[:query_limitter]
                        query_prompt = " ".join(query_prompt)
                        query_limitter -= 10
                prev_questions.add(i)
                question = i
                # response = tavily_client.search(query=question, include_answer=True)
                # To get the data excluding the most recent data of this month. (for evaluation purposes)
                
                try:
                    response = tavily_client.search(query=question, include_answer=True, end_date="2025-10-01") #, end_date="2025-10-01"
                except Exception as e:
                    api_exhausted(e)
                answer = response.get('answer', '')
                static_ans[i]=answer
                
                # for tavily research (beta)
                # response = tavily_client.research(query=question)
                # request_id = response.get('request_id', '')
                # answer = tavily_client.get_research(request_id=request_id)
                
                log_tavily(question, response, static_var, research_topic)
                static_var+=1
            answer_list.append(answer)
            # print("\nquestion: \n",question,"\ntavily response:\n", answer)
        if len(answer_list): return answer_list
        results = response.get('results', [])
        return results[0].get('content', "No information found") if results else "No information found"
    except Exception as e: 
        print("\n exceeded 400 character limit")
        return f"Search error: {str(e)}"

def get_related_work_tavily(research_topic, description):
    """
    Retrieves related work for the given research topic using the Tavily client.
    """
    global static_var, things_discussed
    llm_prompt = """Provide studies relevant to: {research_topic}, research description: {description}, topics discussed: {alr_discussed}, return a simple prompt which will be used for Tavily Search and it should be under 380 characters, return the answer only in STRING format (SOFT CONSTRAINT: 350 Character at max, HARD CONSTRAINT: 380 Character at max)""".format(research_topic=research_topic, description=description, alr_discussed=things_discussed)
    prompt = get_llm_response(model_name = "gpt-4.1-nano", prompt=llm_prompt)
    if len(prompt)>400:
        query_limitter = 390
        while len(prompt)>390:
            prompt = prompt.split()[:query_limitter]
            prompt = " ".join(prompt)
            query_limitter -= 10
    try:
        response = tavily_client.search(query=prompt, include_answer=True, end_date="2025-10-01") #, end_date="2025-10-01"
    except Exception as e:
        api_exhausted(e)
    answer = response.get('answer', '')
    log_tavily(prompt, response, static_var, research_topic)
    static_var+=1

    return answer