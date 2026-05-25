from llm_api import get_llm_response, tavily_api_key
from prompts import PERSONALITY_GENERATION_PROMPT, TAVILY_CHAR_GENERATION_PROMPT, TAVILY_HELPER_PROMPT
from tavily import TavilyClient
from tavily_logger import log_tavily
from datetime import date     

# getting the tavily api key
tavily_key = tavily_api_key 
tavily_client = TavilyClient(api_key=tavily_api_key)

# We need to have simpler question otherwise tavily will just answer "None".

# to get the static var for logging (might change later/ might shift it to the agent class.)
static_var=0
static_ans=dict()
prev_questions=set()
things_discussed = ""

# To get the date one month before the current date
def one_month_before(d: date) -> date:
    year = d.year
    month = d.month
    day = d.day

    # Move one month back
    if month == 1:
        month = 12
        year -= 1
    else:
        month -= 1

    # Handle cases like 31 March → 28 Feb (or 29 in leap year)
    while True:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1

# Example:
today = date.today()
one_month_ago = one_month_before(today)


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
    # query_resp = tavily_client.search(query= query_prompt, include_answer=True)
    # To get the data excluding the most recent data of this month. (for evaluation purposes)
    query_resp = tavily_client.search(query= query_prompt, include_answer=True, end_date=str(one_month_ago))
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

def generate_based_on(research_topic: str, agent_type: str) -> str:
    global static_var
    if agent_type == "SC":
        prompt = TAVILY_CHAR_GENERATION_PROMPT.format(
            agent_type="SC",
            research_topic=research_topic
        )   
    else:
        prompt = TAVILY_CHAR_GENERATION_PROMPT.format(
            agent_type="PI",
            research_topic=research_topic
        )

    response = tavily_client.search(query=prompt, include_answer=True)
    answer = response.get('answer', '')
    # print("\nquestion: \n",prompt,"\nbased on:\n", answer)
    log_tavily(prompt, response, static_var, research_topic)
    static_var+=1

    return answer

def generate_personality(research_topic: str, based_on: str, agent_type: str) -> str:
    """
    Generates a personality description for the agent based on the research topic
    and the context provided.
    """
    global static_var

    question = """
        Generate a scintific personality traits description of this real-world scientist : {based_on}
    """.format(based_on=based_on)

    agent_traits = tavily_client.search(query=question, include_answer=True)
    answer = agent_traits.get('answer', '')
    # print("\nquestion: \n",question,"\ntavily response:\n", answer)
    log_tavily(question, agent_traits, static_var, research_topic)
    static_var+=1


    prompt = PERSONALITY_GENERATION_PROMPT.format(
        research_topic=research_topic,
        based_on=based_on,
        agent_type = agent_type,
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
                prev_questions.add(i)
                question = i
                # response = tavily_client.search(query=question, include_answer=True)
                # To get the data excluding the most recent data of this month. (for evaluation purposes)
                response = tavily_client.search(query=question, include_answer=True, end_date=str(one_month_ago))
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

def get_related_work_tavily(research_topic):
    """
    Retrieves related work for the given research topic using the Tavily client.
    """
    global static_var, things_discussed
    prompt = """Provide studies relevant to: {research_topic}, topics discussed: {alr_discussed}""".format(research_topic=research_topic, alr_discussed=things_discussed)

    response = tavily_client.search(query=prompt, include_answer=True)
    answer = response.get('answer', '')
    log_tavily(prompt, response, static_var, research_topic)
    static_var+=1

    return answer