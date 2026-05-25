import fitz  # This is for PDF
import os
from llm_api import get_llm_response
from prompts import Conversion_prompt
from conversion_eval import check
from evaluation import structured_json_to_text, compare_rep
from paper_logger import save_paper
import json
import unicodedata
import re

def clean_json_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)      # Normalize fancy quotes/hyphens
    t = re.sub(r'\s+', ' ', t)               # Collapse all whitespace to single space
    return t.strip() 


dummy = """{
  "research_title": "Closed-loop Auto-Research Framework for Autonomous Scientific Discovery",
  "Abstract": "This research explores a fully automated scientific research framework (DOLPHIN) that closes the loop between idea generation, experiment execution, and feedback. The system generates novel hypotheses from relevant literature, implements and tests them via automated code generation and debugging, and uses results to refine subsequent ideas. This cycle aims to approach human-level scientific research automation and demonstrate performance on diverse benchmark tasks. DOLPHIN shows comparable results with state-of-the-art human-designed methods in image classification, 3D point classification, and sentiment analysis. ",
  "Complete_Pipeline_:_Introduction": "Traditional research is human-driven or AI-assisted. Recent work automates parts of idea generation or experiments but lacks integrated feedback from results back into research cycles. This project proposes a closed-loop system that retrieves and ranks literature, generates ideas, synthesizes experimental plans and code, verifies them, and incorporates performance feedback into future idea generation. The pipeline mimics human iterative research refinement. ",
  "methodology": "The system has three main components:1) **Idea Generation**: Retrieve relevant literature, rank papers using topic and task attributes, and prompt an LLM to generate multiple candidate research ideas. Filter for novelty and independence. 2) **Experimental Verification**: Use LLM templates to generate code for each idea. Use exception-traceback-guided debugging to automatically fix code errors and execute experiments. 3) **Result Feedback**: Categorize results by performance change and feed embeddings of successful ideas back into idea prompts to discourage redundant or ineffective hypotheses. ",
  "recent_researches_and_gaps": "AI-assisted research (GPT-Researcher, AI-Scientist) can break down tasks or generate plans, but often relies on human evaluation or lacks systematic validation on real datasets. Prior automatic frameworks may not tightly integrate feedback from results into new idea generation, reducing iterative improvement efficiency. Existing work also uses simple or synthetic benchmarks, limiting real-world performance comparison. ",
  "proposed_idea_novelty": "Close the research loop by integrating experiment outcomes directly into idea generation prompts. Task-attribute-guided paper ranking and exception-traceback debugging improve idea quality and execution success. This approach goes beyond single-stage automation — it continually refines research objectives based on actual experimental feedback. ",
  "implementation_pipeline": "1) **Input Problem Definition** → 2) **Retrieve Literature** from APIs and rank by relevance → 3) **Generate Candidate Ideas** via LLM → 4) **Filter Ideas** (novel, independent) → 5) **Generate Code Plans** for each idea → 6) **Debug & Compile Experiments** via traceback analysis → 7) **Run Experiments** on benchmark datasets → 8) **Analyze Results** → 9) **Incorporate Feedback** into idea prompts → 10) **Repeat Loop** until performance stagnates. ",
  "evaluation_and_benchmarking_criteria": "Evaluate on standard benchmarks: CIFAR-100 (image classification), ModelNet40 (3D shape classification), SST-2 (sentiment analysis), and a wider MLE-bench suite. Compare accuracy/performance to baseline models (PointNet, WideResNet, BERT-base). Evaluate idea quality by improvement over baselines, novelty filtering effectiveness, and execution success rate. Quantify iteration-to-improvement rates and compute efficiency in continuous improvement cycles. "
}
"""
conversion_eval = {}
to_store = {}

def is_json(my_string):
    try:
        json.loads(my_string)
    except ValueError:
        return False
    return True

def read_pdf(ref_report):
    global conversion_eval, static_num
    path = os.path.join("Reference_Reports", f"{ref_report}.pdf")
    with fitz.open(path) as doc:
        # Join text from all pages into one string
        text = "".join(page.get_text() for page in doc)
    query = Conversion_prompt.format(ref_report=text)
    to_store["PAPER"] = text
    j_son = get_llm_response("gpt-5-mini", query, temperature=0)
    if not is_json(j_son):
        read_pdf(ref_report)
        
    j_text = clean_json_text(structured_json_to_text(j_son))
    dum = clean_json_text(structured_json_to_text(dummy))
    conversion_eval["converted_proposal_and_the_paper"] = compare_rep(j_text, text)
    conversion_eval["dummy_and_the_paper"] = compare_rep(dum, text)
    conversion_array = []
    for i in range(10):
        if i%2:
            try:
                res = check(text, j_son)
                conversion_array.append(res)
            except Exception as e:
                print("Error in checking conversion:", e)
        else:
            try:
                res = check(text, dummy)
                conversion_array.append(not res)
            except Exception as e:
                print("Error in checking conversion with dummy:", e)
    if conversion_array.count(True)>conversion_array.count(False):
        conversion_eval["llm_checking results"] = conversion_array
        to_store["JSON"] = j_text
        to_store["EVALUATION"] = conversion_eval
        save_paper(to_store, ref_report)
    else:
        print("Retry...")
        read_pdf(ref_report)

read_pdf("Agent_Drift")