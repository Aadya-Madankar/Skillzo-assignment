# This file contains nodes for summary generation, insights extraction, and question generation

# Part B: Summary, Insight Extraction, and Question Generation Nodes

from typing import Dict, Any, List, Iterator
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf_utils import load_vector_store 
import json
import time


def summary_generation_node(vector_store_path: str) -> Dict[str, Any]:
    try:
        vector_store = load_vector_store(vector_store_path)
        if not vector_store:
            return {"success": False, "error": f"Failed to load vector store from {vector_store_path}"}
        
        # Search for comprehensive resume content
        docs = vector_store.similarity_search("overall summary of professional experience, education, and key skills", k=7) # Increased k
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        if not context.strip():
            return {"success": False, "error": "No relevant context found in vector store for summary generation."}

        prompt = f"""
        Based on the following resume content, create a comprehensive professional summary.
        This summary should act as an "elevator pitch" for the candidate. 
    
        
        Resume Context:
        ---
        {context}
        ---
        
        Instructions for the Summary:
        1. Start with a concise professional overview (2-3 sentences) that captures the candidate's main role, years of experience, and core expertise.
        2. Highlight 2-3 key work experiences or significant achievements, mentioning specific roles or projects if prominent.
        3. Briefly mention the highest or most relevant educational qualifications.
        4. Weave in 3-5 core skills or areas of expertise that are evident from the context.
        5. The summary should be well-structured, ideally 5-6 paragraphs, and maintain a highly professional tone.
        6. Ensure the summary is engaging and accurately reflects the candidate's profile based *only* on the provided context. Do not invent information.
        
        Professional Summary:
        """
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) 
        summary = llm.invoke(prompt).content
        
        return {
            "success": True,
            "summary": summary.strip(),
            "word_count": len(summary.strip().split())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Summary generation failed: {str(e)}"
        }

def streaming_summary_generation_node(vector_store_path: str) -> Iterator[str]:
    try:
        node_result = summary_generation_node(vector_store_path)
        
        if node_result.get("success"):
            summary_text = node_result["summary"]
            words = summary_text.split()
            buffer = ""
            
            for i, word in enumerate(words):
                buffer += word + " "
                # Stream moderately sized chunks
                if (i + 1) % 7 == 0 or i == len(words) - 1: 
                    yield buffer.strip()
                    buffer = ""
                    time.sleep(0.02) 
            if buffer.strip():
                 yield buffer.strip()
        else:
            yield f"Error: {node_result.get('error', 'Failed to generate summary.')}"
                
    except Exception as e:
        yield f"Error during summary streaming: {str(e)}"

def insights_extraction_node(vector_store_path: str) -> Dict[str, Any]:
    response_content = ""
    try:
        vector_store = load_vector_store(vector_store_path)
        if not vector_store:
            return {"success": False, "error": f"Failed to load vector store from {vector_store_path}"}

        docs = vector_store.similarity_search("key skills, notable achievements, significant projects, career highlights, unique qualifications", k=6)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        if not context.strip():
            return {"success": False, "error": "No relevant context found in vector store for insights extraction."}
        
        prompt = f"""
        Analyze the provided resume context and extract a list of 3-5 concise, impactful insights.
        These insights should highlight key strengths, significant experiences, notable achievements, or unique aspects of the candidate's profile.
        Return ONLY a valid JSON object adhering to the specified schema.

        Resume Context:
        ---
        {context}
        ---
        
        Schema for JSON output:
        {{
            "insights": [
                "string: Insight 1 about the candidate.",
                "string: Insight 2 highlighting a key skill or achievement.",
                "string: Insight 3 about their experience or qualification."
                // ... up to 5 insights
            ]
        }}
        
        Examples of good insights:
        - "Over 7 years of progressive experience in software engineering with a focus on cloud technologies."
        - "Demonstrated leadership by managing a team of 5 developers on a critical project."
        - "Successfully launched 2 major products, resulting in significant user growth."
        - "Possesses a strong academic background in Computer Science with a Master's degree."
        - "Certified AWS Solutions Architect with hands-on experience in deploying scalable applications."

        Important Rules:
        - Focus on quantifiable achievements where possible (e.g., years of experience, team size, project impact).
        - Highlight key technologies, methodologies, or domain expertise.
        - Note leadership, management, or significant contributions.
        - Keep each insight concise, impactful, and factual, based SOLELY on the provided context.
        - Aim for 3 to 5 distinct insights.
        - The output MUST be ONLY the JSON object, with no introductory text, explanations, or markdown code fences.
        
        JSON Output:
        """
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) 
        ai_response_obj = llm.invoke(prompt)
        response_content = ai_response_obj.content.strip()
        
        json_str = response_content
        if json_str.startswith('```json'):
            json_str = json_str[len('```json'):].strip()
        if json_str.endswith('```'):
            json_str = json_str[:-len('```')].strip()
        
        insights_data = json.loads(json_str)
        
        # Validate that 'insights' key exists and is a list
        if "insights" not in insights_data or not isinstance(insights_data["insights"], list):
            raise ValueError("LLM response for insights did not match expected structure ('insights' key missing or not a list).")

        return {
            "success": True,
            "insights": insights_data.get("insights", []),
            "count": len(insights_data.get("insights", []))
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parsing error for insights: {str(e)}",
            "raw_response": response_content
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Insights extraction failed: {str(e)}",
            "raw_response": response_content
        }

def interview_question_generation_node(insights: List[str]) -> Dict[str, Any]:
    response_content = ""
    if not insights:
        return {"success": False, "error": "No insights provided to generate interview questions."}

    try:
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        
        prompt = f"""
        Based on the following key insights extracted from a candidate's resume, generate a list of 3-5 tailored interview questions.
        These questions should help an interviewer delve deeper into the candidate's experience, skills, and achievements.
        Return ONLY a valid JSON object adhering to the specified schema.

        Key Resume Insights:
        ---
        {insights_text}
        ---
        
        Schema for JSON output:
        {{
            "questions": [
                "string: Question 1 tailored to the insights.",
                "string: Question 2 exploring a specific point from the insights.",
                "string: Question 3 (behavioral or situational, based on insights)."
                // ... up to 5 questions
            ]
        }}
        
        Important Rules:
        - Generate 3 to 5 relevant interview questions.
        - Each question MUST be directly related to one or more of the provided insights.
        - Include a mix of behavioral, technical, and experience-probing questions.
        - Make questions open-ended to encourage detailed responses.
        - Avoid generic questions; personalize them based on the candidate's specific profile as reflected in the insights.
        - The output MUST be ONLY the JSON object, with no introductory text, explanations, or markdown code fences.
        
        JSON Output:
        """
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        ai_response_obj = llm.invoke(prompt)
        response_content = ai_response_obj.content.strip()
        
        json_str = response_content
        if json_str.startswith('```json'):
            json_str = json_str[len('```json'):].strip()
        if json_str.endswith('```'):
            json_str = json_str[:-len('```')].strip()
        
        questions_data = json.loads(json_str)

        if "questions" not in questions_data or not isinstance(questions_data["questions"], list):
             raise ValueError("LLM response for questions did not match expected structure ('questions' key missing or not a list).")
        
        return {
            "success": True,
            "questions": questions_data.get("questions", []),
            "count": len(questions_data.get("questions", []))
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parsing error for questions: {str(e)}",
            "raw_response": response_content
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Question generation failed: {str(e)}",
            "raw_response": response_content
        }

def streaming_first_question_node(insights: List[str]) -> Iterator[str]:
    if not insights:
        yield "Error: No insights provided for question generation."
        return

    try:
        questions_result = interview_question_generation_node(insights) # Generate all questions first
        
        if questions_result.get("success") and questions_result.get("questions"):
            first_question = questions_result["questions"][0]
            words = first_question.split()
            buffer = ""
            
            for i, word in enumerate(words):
                buffer += word + " "
                if (i + 1) % 5 == 0 or i == len(words) - 1:  # Stream every 5 words or at the end
                    yield buffer.strip()
                    buffer = ""
                    time.sleep(0.02) # Small delay
            if buffer.strip():
                yield buffer.strip()
        else:
            yield f"Error: {questions_result.get('error', 'Could not generate interview questions based on the resume.')}"
            
    except Exception as e:
        yield f"Error generating first interview question: {str(e)}"