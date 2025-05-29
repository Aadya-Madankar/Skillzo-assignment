#Utilities for LangGraph definition, state, nodes, and compilation


from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For in-memory checkpointing

# --- Configuration and Initialization ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY not found. LangGraph AI features might not work.")
else:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"Error configuring Google Generative AI in graph_utils: {e}")

# --- LangGraph State Definition ---
class LangGraphResumeState(TypedDict):
    # Inputs
    resume_text: Optional[str]
    input_summary: Optional[str]
    input_insights: Optional[List[str]]

    # Intermediate and Output values
    work_experience: Optional[str] # Note: These are currently simple strings from LLM, not structured JSON from extraction_nodes.py
    education: Optional[str]     # This is by design for this specific LangGraph example to keep it simpler.
                                 # If you wanted structured data here, these nodes would need to call your
                                 # extraction_nodes.py functions (which expect vector_store_path, not raw text)
                                 # or be redesigned for raw text JSON extraction.
    generated_summary: Optional[str]
    extracted_insights: Optional[List[str]]
    generated_questions: Optional[List[str]]

    # Control flow and metadata
    error: Optional[str]
    current_node: Optional[str]
    execution_id: Optional[str] 
    
    skip_summary_generation: bool
    skip_insights_extraction: bool


# --- LangGraph Node Functions ---
LLM_MODEL_NAME_GRAPH = "gemini-2.5-flash-preview-05-20"  

def initialize_state_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "initialize_state"
    # Ensure booleans are set correctly based on presence of input
    state['skip_summary_generation'] = bool(state.get('input_summary'))
    state['skip_insights_extraction'] = bool(state.get('input_insights'))
    
    print(f"\n--- Initializing State for execution_id: {state.get('execution_id')} ---")
    print(f"  Input resume_text present: {bool(state.get('resume_text'))}")
    print(f"  Input input_summary: {state.get('input_summary')}")
    print(f"  Input input_insights: {state.get('input_insights')}")
    print(f"  Calculated skip_summary_generation: {state['skip_summary_generation']}")
    print(f"  Calculated skip_insights_extraction: {state['skip_insights_extraction']}")

    if not state.get('resume_text') and not state.get('input_summary') and not state.get('input_insights'):
        state['error'] = "No resume text, input summary, or input insights provided to start analysis."
        print(f"  ERROR set in initialize_state: {state['error']}")
    return state

def extract_work_experience_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "extract_work_experience"
    print(f"\n--- Entering extract_work_experience_node for execution_id: {state.get('execution_id')} ---")
    if state.get('error') or not state.get('resume_text'):
        state['work_experience'] = "Skipped work experience extraction due to prior error or missing resume text."
        print(f"  Skipping work experience: error='{state.get('error')}', resume_text_present={bool(state.get('resume_text'))}")
        return state
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GRAPH, temperature=0.2)
        prompt = PromptTemplate.from_template(
            "Extract all work experience from this resume text:\n\n{resume_text}\n\n"
            "Present it as a structured summary. If none, state 'No work experience found'."
        )
        chain = prompt | llm
        # Limiting context size for safety, though gemini-2.5-flash-preview-05-20 has a large context window
        response = chain.invoke({"resume_text": state['resume_text'][:30000]}) 
        state['work_experience'] = response.content
        print(f"  Work experience extracted (first 100 chars): {state['work_experience'][:100]}...")
    except Exception as e:
        state['error'] = f"Work experience extraction error: {str(e)}"
        state['work_experience'] = "Error during work experience extraction."
        print(f"  ERROR during work experience extraction: {e}")
    return state

def extract_education_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "extract_education"
    print(f"\n--- Entering extract_education_node for execution_id: {state.get('execution_id')} ---")
    if state.get('error') or not state.get('resume_text'):
        state['education'] = "Skipped education extraction due to prior error or missing resume text."
        print(f"  Skipping education: error='{state.get('error')}', resume_text_present={bool(state.get('resume_text'))}")
        return state
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GRAPH, temperature=0.2)
        prompt = PromptTemplate.from_template(
            "Extract all education information from this resume text:\n\n{resume_text}\n\n"
            "Present as a structured summary. If none, state 'No education information found'."
        )
        chain = prompt | llm
        response = chain.invoke({"resume_text": state['resume_text'][:30000]})
        state['education'] = response.content
        print(f"  Education extracted (first 100 chars): {state['education'][:100]}...")
    except Exception as e:
        state['error'] = f"Education extraction error: {str(e)}"
        state['education'] = "Error during education extraction."
        print(f"  ERROR during education extraction: {e}")
    return state

def generate_summary_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "generate_summary"
    print(f"\n--- Entering generate_summary_node for execution_id: {state.get('execution_id')} ---")
    print(f"  Initial error state: {state.get('error')}")
    print(f"  skip_summary_generation: {state.get('skip_summary_generation')}")
    print(f"  input_summary: {state.get('input_summary')}")

    if state.get('error'):
        state['generated_summary'] = "Skipped summary generation due to prior error."
        print("  Skipping summary due to prior error.")
        return state
    if state.get('skip_summary_generation') and state.get('input_summary'):
        state['generated_summary'] = state['input_summary']
        print(f"  Skipping summary generation, using input_summary: {state['generated_summary'][:100]}...")
        return state
    
    work_exp = state.get('work_experience', "Not available or error during extraction.")
    edu = state.get('education', "Not available or error during extraction.")
    
    # Check if critical inputs are truly missing or just placeholders from errors/skips
    is_work_exp_valid = isinstance(work_exp, str) and work_exp.strip() and not any(err_msg in work_exp for err_msg in ["Not available", "Skipped", "Error during"])
    is_edu_valid = isinstance(edu, str) and edu.strip() and not any(err_msg in edu for err_msg in ["Not available", "Skipped", "Error during"])

    if not is_work_exp_valid and not is_edu_valid: # If BOTH are invalid/missing
        state['generated_summary'] = "Cannot generate summary: Work experience and education details are missing, invalid, or errored."
        print(f"  Cannot generate summary due to missing/errored work_exp ('{work_exp[:50]}...') and edu ('{edu[:50]}...').")
        return state
    elif not is_work_exp_valid:
        print(f"  Warning: Work experience is missing/invalid for summary ('{work_exp[:50]}...'). Proceeding with education only.")
    elif not is_edu_valid:
        print(f"  Warning: Education is missing/invalid for summary ('{edu[:50]}...'). Proceeding with work experience only.")


    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GRAPH, temperature=0.5)
        prompt_template = PromptTemplate.from_template(
            "Create a professional summary (2-3 paragraphs) based on the following information:\n\n"
            "Work Experience:\n{work_experience}\n\n"
            "Education:\n{education}\n\n"
            "If some information is missing or says 'Not available' or 'Error', acknowledge that and summarize based on the available valid information. "
            "Focus on professionalism and conciseness.\n"
            "Summary:"
        )
        chain = prompt_template | llm
        response = chain.invoke({"work_experience": work_exp, "education": edu})
        state['generated_summary'] = response.content
        print(f"  Summary generated (first 100 chars): {state['generated_summary'][:100]}...")
    except Exception as e:
        state['error'] = f"Summary generation error: {str(e)}"
        state['generated_summary'] = "Error during summary generation."
        print(f"  ERROR during summary generation: {e}")
    return state

def extract_insights_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "extract_insights"
    print(f"\n--- Entering extract_insights_node for execution_id: {state.get('execution_id')} ---")
    print(f"  Initial error state: {state.get('error')}")
    print(f"  skip_insights_extraction: {state.get('skip_insights_extraction')}")
    print(f"  input_insights: {state.get('input_insights')}")
    print(f"  generated_summary (input to this node, first 100): {str(state.get('generated_summary'))[:100]}...")

    if state.get('error'):
        state['extracted_insights'] = ["Skipped insights extraction due to prior error."]
        print("  Skipping insights due to prior error.")
        return state
    if state.get('skip_insights_extraction') and state.get('input_insights') and isinstance(state.get('input_insights'), list):
        # Ensure input_insights are valid before using them to skip
        input_insights_val = state.get('input_insights', [])
        if not any("Error" in str(i) or "Cannot extract" in str(i) for i in input_insights_val):
            state['extracted_insights'] = input_insights_val
            print(f"  Skipping insights extraction, using valid input_insights: {state['extracted_insights']}")
            return state
        else:
            print(f"  Input insights provided but seem invalid: {input_insights_val}. Proceeding to generate new insights.")
            # Do not return, let it try to generate new ones.

    summary_to_use = state.get('generated_summary', "")
    if not isinstance(summary_to_use, str) or not summary_to_use.strip() or \
       ("Error" in summary_to_use or "Skipped" in summary_to_use or "Cannot generate" in summary_to_use):
         state['extracted_insights'] = ["Cannot extract insights: The resume summary is missing, invalid, or contains errors."]
         print(f"  Cannot extract insights, summary issue: '{summary_to_use[:100]}...'")
         return state
    
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GRAPH, temperature=0.3)
        prompt_template = PromptTemplate.from_template(
            "Based on the resume summary below, extract 3-5 key insights as a JSON list of strings.\n\n"
            "Resume Summary:\n{summary}\n\n"
            "JSON Output Example: {{\"insights\": [\"Insight 1 about experience.\", \"Insight 2 highlighting a key skill.\", \"Insight 3 about achievements.\"]}}\n"
            "Ensure the output is ONLY the JSON object, with no introductory text or markdown code fences.\n"
            "JSON Output:"
        )
        chain = prompt_template | llm
        response_content = chain.invoke({"summary": summary_to_use}).content.strip()
        
        # Clean potential markdown fences
        if response_content.startswith("```json"): response_content = response_content[7:]
        if response_content.endswith("```"): response_content = response_content[:-3]
        response_content = response_content.strip()
            
        try:
            insights_data = json.loads(response_content)
            if "insights" in insights_data and isinstance(insights_data["insights"], list):
                # Filter out empty strings or non-string items just in case
                processed_insights = [str(item).strip() for item in insights_data["insights"] if isinstance(item, str) and str(item).strip()]
                if processed_insights:
                    state['extracted_insights'] = processed_insights
                    print(f"  Successfully extracted insights: {state['extracted_insights']}")
                else:
                    state['extracted_insights'] = ["LLM returned an empty list of insights or non-string items."]
                    print("  Warning: LLM returned empty or non-string insights list.")
            else:
                state['extracted_insights'] = [f"LLM insights format unexpected (missing 'insights' list): {response_content}"]
                print(f"  Warning: LLM insights format unexpected: {response_content}")
        except json.JSONDecodeError:
            state['extracted_insights'] = [f"Insights JSON parsing failed. Raw LLM output: {response_content}"]
            print(f"  Warning: Insights JSON parsing failed. Raw: {response_content}")

    except Exception as e:
        state['error'] = f"Insights extraction error: {str(e)}"
        state['extracted_insights'] = [f"Error during insights extraction: {str(e)}"]
        print(f"  ERROR during insights extraction: {e}")
    return state


def generate_questions_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "generate_questions"
    print(f"\n--- Entering generate_questions_node for execution_id: {state.get('execution_id')} ---")
    print(f"  Initial error state: {state.get('error')}")
    print(f"  extracted_insights (input to this node): {state.get('extracted_insights')}")

    if state.get('error'):
        state['generated_questions'] = ["Skipped question generation due to prior error."]
        print("  Skipping questions due to prior error.")
        return state
    
    insights_list = state.get('extracted_insights', [])
    # Stricter check for valid insights: must be a list of non-empty strings, and no error messages.
    is_insights_list_valid = isinstance(insights_list, list) and \
                             all(isinstance(i, str) and i.strip() for i in insights_list) and \
                             not any("Error" in i or "Raw insights" in i or "Cannot extract" in i or "format unexpected" in i for i in insights_list)


    if not is_insights_list_valid:
        state['generated_questions'] = ["Cannot generate questions: Insights are missing, invalid, empty, or contain error messages."]
        print(f"  Cannot generate questions, insights issue. Provided insights: {insights_list}")
        return state

    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GRAPH, temperature=0.5)
        prompt_template = PromptTemplate.from_template(
            "Based on these resume insights, generate 3 tailored interview questions as a JSON list of strings:\n\n"
            "Insights:\n{insights_text}\n\n"
            "JSON Output Example: {{\"questions\": [\"Tell me more about [specific insight context]?\", \"How did you achieve [specific result mentioned in insight]?\", \"Can you elaborate on your experience with [skill from insight]?\"]}}\n"
            "Ensure the output is ONLY the JSON object, with no introductory text or markdown code fences.\n"
            "JSON Output:"
        )
        chain = prompt_template | llm
        insights_text_for_prompt = "\n- ".join(insights_list)
        response_content = chain.invoke({"insights_text": "- " + insights_text_for_prompt}).content.strip()
        
        if response_content.startswith("```json"): response_content = response_content[7:]
        if response_content.endswith("```"): response_content = response_content[:-3]
        response_content = response_content.strip()

        try:
            questions_data = json.loads(response_content)
            if "questions" in questions_data and isinstance(questions_data["questions"], list):
                processed_questions = [str(q).strip() for q in questions_data["questions"] if isinstance(q, str) and str(q).strip()]
                if processed_questions:
                    state['generated_questions'] = processed_questions[:5] # Limit to 5
                    print(f"  Successfully generated questions: {state['generated_questions']}")
                else:
                    state['generated_questions'] = ["LLM returned an empty list of questions."]
                    print("  Warning: LLM returned an empty list of questions.")
            else:
                state['generated_questions'] = [f"LLM questions format unexpected (missing 'questions' list): {response_content}"]
                print(f"  Warning: LLM questions format unexpected: {response_content}")
        except json.JSONDecodeError:
            state['generated_questions'] = [f"Questions JSON parsing failed. Raw LLM output: {response_content}"]
            print(f"  Warning: Questions JSON parsing failed. Raw: {response_content}")

    except Exception as e:
        state['error'] = f"Question generation error: {str(e)}"
        state['generated_questions'] = [f"Error during question generation: {str(e)}"]
        print(f"  ERROR during question generation: {e}")
    return state

def final_node(state: LangGraphResumeState) -> LangGraphResumeState:
    state['current_node'] = "END"
    print(f"--- LangGraph execution reached END node for execution_id: {state.get('execution_id')} ---")
    print(f"  Final error state: {state.get('error')}")
    return state

# --- LangGraph Definition ---
graph_checkpointer = MemorySaver()

def build_resume_analysis_graph():
    workflow = StateGraph(LangGraphResumeState)

    workflow.add_node("initialize_state", initialize_state_node)
    workflow.add_node("extract_work_experience", extract_work_experience_node)
    workflow.add_node("extract_education", extract_education_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("extract_insights", extract_insights_node)
    workflow.add_node("generate_questions", generate_questions_node)
    workflow.add_node("final_node_marker", final_node) 

    workflow.set_entry_point("initialize_state")

    def should_extract_full_details(state: LangGraphResumeState):
        print(f"[Router after initialize_state] exec_id: {state.get('execution_id')}, error: {state.get('error')}, resume_text: {bool(state.get('resume_text'))}, skip_summary: {state.get('skip_summary_generation')}, skip_insights: {state.get('skip_insights_extraction')}")
        if state.get('error'): return "final_node_marker"
        
        # If insights are provided AND valid, jump to generating questions
        if state.get('skip_insights_extraction') and isinstance(state.get('input_insights'), list) and state.get('input_insights') and not any("Error" in str(i) for i in state.get('input_insights', [])):
            print("  Routing: initialize_state -> generate_questions (due to valid input_insights)")
            return "generate_questions" 
        
        # If summary is provided, jump to extracting insights (unless insights were also provided and handled above)
        if state.get('skip_summary_generation') and state.get('input_summary'):
            print("  Routing: initialize_state -> extract_insights (due to input_summary, will generate insights from it)")
            return "extract_insights" 

        if state.get('resume_text'):
            print("  Routing: initialize_state -> extract_work_experience (standard flow with resume_text)")
            return "extract_work_experience"
            
        print("  Routing: initialize_state -> final_node_marker (fallback/no valid input path)")
        state['error'] = state.get('error', "Cannot determine starting point from initialize_state.") 
        return "final_node_marker"

    workflow.add_conditional_edges(
        "initialize_state",
        should_extract_full_details,
        {
            "extract_work_experience": "extract_work_experience",
            "extract_insights": "extract_insights", 
            "generate_questions": "generate_questions", 
            "final_node_marker": "final_node_marker"
        }
    )
    
    workflow.add_edge("extract_work_experience", "extract_education")
    workflow.add_edge("extract_education", "generate_summary")
    
    def route_after_summary(state: LangGraphResumeState):
        print(f"[Router after generate_summary] exec_id: {state.get('execution_id')}, error: {state.get('error')}, generated_summary_valid: {not('Error' in str(state.get('generated_summary')) or 'Cannot generate' in str(state.get('generated_summary')))}")
        if state.get('error'): return "final_node_marker"
        # If summary generation itself failed, it might not be useful to proceed.
        current_summary = state.get('generated_summary', "")
        if not isinstance(current_summary, str) or "Error" in current_summary or "Cannot generate" in current_summary:
            print("  Routing: generate_summary -> final_node_marker (summary generation failed or invalid)")
            return "final_node_marker"
        print("  Routing: generate_summary -> extract_insights")
        return "extract_insights"

    workflow.add_conditional_edges("generate_summary", route_after_summary, {
        "extract_insights": "extract_insights", 
        "final_node_marker": "final_node_marker"
    })
    
    def route_after_insights(state: LangGraphResumeState):
        print(f"[Router after extract_insights] exec_id: {state.get('execution_id')}, error: {state.get('error')}, extracted_insights: {state.get('extracted_insights')}")
        if state.get('error'): return "final_node_marker"
        insights_list = state.get('extracted_insights', [])
        is_valid = isinstance(insights_list, list) and insights_list and not any("Error" in i or "Cannot extract" in i or "Raw insights" in i for i in insights_list)
        if not is_valid:
            print("  Routing: extract_insights -> final_node_marker (insights are invalid or errored)")
            return "final_node_marker"
        print("  Routing: extract_insights -> generate_questions")
        return "generate_questions"

    workflow.add_conditional_edges("extract_insights", route_after_insights, {
        "generate_questions": "generate_questions", 
        "final_node_marker": "final_node_marker"
    })
    
    workflow.add_edge("generate_questions", "final_node_marker")
    workflow.add_edge("final_node_marker", END)

    return workflow.compile(checkpointer=graph_checkpointer)

compiled_resume_analyzer_graph = build_resume_analysis_graph()
