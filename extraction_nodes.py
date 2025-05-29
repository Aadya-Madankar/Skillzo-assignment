# This file contains nodes for extracting structured data from resumes
# Part A

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf_utils import load_vector_store 
from models import WorkExperienceList, EducationList
import json



def work_experience_extraction_node(vector_store_path: str) -> Dict[str, Any]:
    response_content = ""
    try:
        # Load vector store
        vector_store = load_vector_store(vector_store_path)
        if not vector_store:
            return {"success": False, "error": f"Failed to load vector store from {vector_store_path}"}
        
        # Search for work experience related content
        docs = vector_store.similarity_search("work experience, employment history, job titles, companies, responsibilities, professional experience", k=7) # Increased k for more context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        if not context.strip():
            return {"success": False, "error": "No relevant context found in vector store for work experience."}

        # Create extraction prompt
        prompt = f"""
        Extract all work experience entries from the provided resume context.
        Return ONLY a valid JSON object adhering to the specified schema.

        Resume Context:
        ---
        {context}
        ---
        
        Schema for JSON output:
        {{
            "work_experiences": [
                {{
                    "company": "string, name of the company",
                    "role": "string, job title or role",
                    "start_date": "string or null, format YYYY-MM. If only year is available, use YYYY-01. If month is word, convert to MM.",
                    "end_date": "string or null, format YYYY-MM or 'Present'. If only year is available, use YYYY-12 (or YYYY-MM if month known).", 
                    "description": "string, concise summary of responsibilities and achievements"
                }}
                // ... more experiences
            ]
        }}
        
        Important Rules:
        - If no work experience is found, return {{"work_experiences": []}}.
        - Dates MUST be in "YYYY-MM" format or the literal string "Present". Use null if a date is truly unextractable.
        - Convert month names (e.g., "June", "Jun") to their numeric representation (e.g., "06").
        - Descriptions should be detailed but not overly verbose. Capture key actions and outcomes.
        - The output MUST be ONLY the JSON object, with no introductory text, explanations, or markdown code fences.
        
        JSON Output:
        """
        
        # Get AI response
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.2)
        ai_response_obj = llm.invoke(prompt)
        response_content = ai_response_obj.content.strip()
        
        # Clean and parse JSON response
        json_str = response_content
        # Basic cleaning for potential markdown, though the prompt tries to prevent it
        if json_str.startswith('```json'):
            json_str = json_str[len('```json'):].strip()
        if json_str.endswith('```'):
            json_str = json_str[:-len('```')].strip()
        
        # Parse and validate with Pydantic
        work_data = json.loads(json_str)
        validated_data = WorkExperienceList(**work_data)
        
        return {
            "success": True,
            "data": validated_data.dict(),
            "count": len(validated_data.work_experiences)
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parsing error for work experience: {str(e)}",
            "raw_response": response_content
        }
    except Exception as e: # Catches Pydantic validation errors too
        return {
            "success": False,
            "error": f"Work experience extraction failed: {str(e)}",
            "raw_response": response_content 
        }

def education_extraction_node(vector_store_path: str) -> Dict[str, Any]:
    response_content = ""
    try:
        # Load vector store
        vector_store = load_vector_store(vector_store_path)
        if not vector_store:
            return {"success": False, "error": f"Failed to load vector store from {vector_store_path}"}

        # Search for education related content
        docs = vector_store.similarity_search("education, degree, university, college, school, academic qualifications, courses, certifications", k=6)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        if not context.strip():
            return {"success": False, "error": "No relevant context found in vector store for education."}
        
        # Create extraction prompt
        prompt = f"""
        Extract all education entries from the provided resume context.
        Return ONLY a valid JSON object adhering to the specified schema.

        Resume Context:
        ---
        {context}
        ---
        
        Schema for JSON output:
        {{
            "education": [
                {{
                    "institution": "string, name of the university, college, or institution",
                    "degree": "string, e.g., 'Bachelor of Science', 'Master of Arts', 'PhD', 'Certificate'",
                    "field": "string, field of study, e.g., 'Computer Science', 'Data Analytics'. If not specified, try to infer or use 'N/A'.",
                    "start_year": "integer or null, year education started",
                    "end_year": "integer or null, year of graduation or completion"
                }}
                // ... more education entries
            ]
        }}
        
        Important Rules:
        - If no education information is found, return {{"education": []}}.
        - Years MUST be integers (e.g., 2020, not "2020"). Use null if a year is truly unextractable.
        - Include all formal educational qualifications: degrees, diplomas, significant certifications, relevant bootcamps.
        - The output MUST be ONLY the JSON object, with no introductory text, explanations, or markdown code fences.
        
        JSON Output:
        """
        
        # Get AI response
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.2)
        ai_response_obj = llm.invoke(prompt)
        response_content = ai_response_obj.content.strip()
        
        # Clean and parse JSON response
        json_str = response_content
        if json_str.startswith('```json'):
            json_str = json_str[len('```json'):].strip()
        if json_str.endswith('```'):
            json_str = json_str[:-len('```')].strip()
        
        # Parse and validate with Pydantic
        education_data = json.loads(json_str)
        validated_data = EducationList(**education_data) 
        
        return {
            "success": True,
            "data": validated_data.dict(),
            "count": len(validated_data.education)
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parsing error for education: {str(e)}",
            "raw_response": response_content
        }
    except Exception as e: 
        return {
            "success": False,
            "error": f"Education extraction failed: {str(e)}",
            "raw_response": response_content
        }

# Combined extraction node that calls both work and education extraction nodes.
def combined_extraction_node(vector_store_path: str) -> Dict[str, Any]:
    print(f"Starting combined extraction for: {vector_store_path}")
    work_result = work_experience_extraction_node(vector_store_path)
    print(f"Work Experience Result: {work_result.get('success')}")
    education_result = education_extraction_node(vector_store_path)
    print(f"Education Result: {education_result.get('success')}")
    
    return {
        "work_experience": work_result,
        "education": education_result,
        "combined_success": work_result.get("success", False) and education_result.get("success", False)
    }