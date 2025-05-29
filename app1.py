# app1.py
# Main FastAPI application file, combining features of app.py and focused_app.py
# Handles all API endpoints and coordinates between different modules

import os
import shutil
import tempfile
import time # For streaming delays
import json
import asyncio # For combined streaming
from typing import List, AsyncGenerator, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

import uuid # For generating unique thread_ids for LangGraph runs

# Import the new model from models.py
from models import (
    FileUploadResponse, StoredFilesListResponse, StoredFileMeta, StreamEvent,
    WorkExperienceList, EducationList, InsightsResponse, InterviewQuestionsResponse, SummaryResponse,
    ResumeQuestionLangGraphRequest
)

from graph_utils import (
    LangGraphResumeState,
    compiled_resume_analyzer_graph,
)

# Import our custom modules
from file_manager import FileManager
from pdf_utils import process_pdf_to_vector_store, get_pdf_text
from extraction_nodes import (
    work_experience_extraction_node,
    education_extraction_node,
    combined_extraction_node
)
from analysis_nodes import (
    streaming_summary_generation_node,
    insights_extraction_node,
    interview_question_generation_node,
    streaming_first_question_node
)

# For Graph Visualization
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server environments
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import base64


# Initialize FastAPI app
app = FastAPI(
    title="Comprehensive AI Resume Analyzer API",
    description="Structured resume analysis with data extraction, insights, question generation, and direct analysis endpoint.",
    version="3.0.4" # Updated version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global file manager instance
STORAGE_DIRECTORY = "all_resume_files_storage"
file_manager_instance = FileManager(storage_dir=STORAGE_DIRECTORY)

GRAPH_VIS_DIR = Path("graph_visualizations")
GRAPH_VIS_DIR.mkdir(exist_ok=True)
_graph_image_path_str = None


# --- Helper Function to get vector store path ---
def get_valid_vector_store_path(file_id: str) -> str:
    file_meta = file_manager_instance.get_file_metadata(file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail=f"File with ID '{file_id}' not found or has expired.")

    vector_store_path = file_meta.get("vector_store_path")
    if not vector_store_path or not Path(vector_store_path, "index.faiss").exists():
        file_manager_instance.cleanup_expired_files()
        file_meta = file_manager_instance.get_file_metadata(file_id)
        if not file_meta or not Path(file_meta.get("vector_store_path",""), "index.faiss").exists():
             raise HTTPException(status_code=404, detail=f"Vector store for file ID '{file_id}' is missing or corrupted after cleanup attempt.")
        vector_store_path = file_meta.get("vector_store_path")
    return vector_store_path


# --- API Endpoints ---

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy", "message": "Comprehensive AI Resume Analyzer API is running."}

# --- File Management Endpoints ---
@app.post("/upload-resume/", response_model=FileUploadResponse, summary="Upload and Process Resume for Persistent Storage")
async def upload_and_process_resume_persistent(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    operation_temp_dir = None
    try:
        operation_temp_dir = tempfile.mkdtemp(prefix="resume_upload_")
        
        temp_pdf_paths_for_processing = []
        saved_temp_pdf_paths_for_fm = []
        uploaded_file_names = []

        for uploaded_file in files:
            if not uploaded_file.filename.lower().endswith('.pdf'):
                if operation_temp_dir and Path(operation_temp_dir).exists():
                    shutil.rmtree(operation_temp_dir)
                raise HTTPException(status_code=400, detail=f"File '{uploaded_file.filename}' is not a PDF.")

            temp_pdf_path = Path(operation_temp_dir) / uploaded_file.filename
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
            
            temp_pdf_paths_for_processing.append(str(temp_pdf_path))
            saved_temp_pdf_paths_for_fm.append(str(temp_pdf_path))
            uploaded_file_names.append(uploaded_file.filename)

        if not temp_pdf_paths_for_processing:
            if operation_temp_dir and Path(operation_temp_dir).exists():
                shutil.rmtree(operation_temp_dir)
            raise HTTPException(status_code=400, detail="No PDF files were successfully saved temporarily.")

        temp_vector_store_dir_path_for_fm = str(Path(operation_temp_dir) / "temp_faiss_index")
        Path(temp_vector_store_dir_path_for_fm).mkdir(exist_ok=True)

        chunks_count = process_pdf_to_vector_store(temp_pdf_paths_for_processing, temp_vector_store_dir_path_for_fm)

        primary_file_name = uploaded_file_names[0] if uploaded_file_names else "untitled_resume.pdf"
        # Ensure primary_file_name is just the filename for FileManager
        if isinstance(primary_file_name, Path):
            primary_file_name = primary_file_name.name
        elif isinstance(primary_file_name, str) and "/" in primary_file_name or "\\" in primary_file_name:
             primary_file_name = Path(primary_file_name).name


        first_temp_pdf_path_for_fm = saved_temp_pdf_paths_for_fm[0] if saved_temp_pdf_paths_for_fm else None
        if not first_temp_pdf_path_for_fm:
            if operation_temp_dir and Path(operation_temp_dir).exists():
                shutil.rmtree(operation_temp_dir)
            raise HTTPException(status_code=500, detail="Could not determine temporary PDF path for storage.")

        file_id = file_manager_instance.save_file_with_metadata(
            file_name=primary_file_name, 
            temp_pdf_source_path=first_temp_pdf_path_for_fm, 
            temp_vector_store_source_path=temp_vector_store_dir_path_for_fm,
            expiry_days=7
        )

        return FileUploadResponse(
            message="Resume(s) processed successfully. PDF and Vector store created and saved persistently.",
            file_id=file_id,
            original_name=primary_file_name, # This should be the final name stored
            chunks_count=chunks_count
        )
    except ValueError as ve: # From process_pdf_to_vector_store for example
        if operation_temp_dir and Path(operation_temp_dir).exists(): shutil.rmtree(operation_temp_dir)
        raise HTTPException(status_code=400, detail=str(ve))
    except IOError as ioe: # From FileManager or shutil operations
        if operation_temp_dir and Path(operation_temp_dir).exists(): shutil.rmtree(operation_temp_dir)
        raise HTTPException(status_code=500, detail=f"File operation error: {ioe}")
    except HTTPException: # Re-raise if it's already an HTTPException
        if operation_temp_dir and Path(operation_temp_dir).exists(): shutil.rmtree(operation_temp_dir)
        raise
    except Exception as e:
        print(f"Unhandled error during persistent resume upload: {e}")
        import traceback
        traceback.print_exc()
        if operation_temp_dir and Path(operation_temp_dir).exists(): shutil.rmtree(operation_temp_dir)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during upload: {str(e)}")
    finally:
        if operation_temp_dir and Path(operation_temp_dir).exists():
            try: shutil.rmtree(operation_temp_dir)
            except Exception as e_clean: print(f"Error cleaning up temporary directory {operation_temp_dir}: {e_clean}")


@app.get("/stored-files/", response_model=StoredFilesListResponse, summary="List Stored Resumes")
async def list_stored_resumes():
    try:
        cleaned_count = file_manager_instance.cleanup_expired_files()
        if cleaned_count > 0: print(f"Performed cleanup: {cleaned_count} files removed.")
        active_files_meta = file_manager_instance.get_active_files()
        validated_files = {}
        for fid, meta_dict in active_files_meta.items():
            try:
                if all(k in meta_dict for k in StoredFileMeta.model_fields):
                    validated_files[fid] = StoredFileMeta(**meta_dict)
                else: print(f"Warning: Metadata for file_id {fid} is incomplete. Skipping. Meta: {meta_dict}")
            except Exception as e_val: print(f"Error validating metadata for file_id {fid}: {e_val}. Meta: {meta_dict}. Skipping.")
        return StoredFilesListResponse(files=validated_files)
    except Exception as e:
        print(f"Error in list_stored_resumes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stored files: {str(e)}")

@app.delete("/delete-file/{file_id}", summary="Delete a Stored Resume")
async def delete_resume_file(file_id: str):
    try:
        deleted = file_manager_instance.delete_file(file_id)
        if deleted: return {"message": f"File with ID '{file_id}' deleted successfully."}
        else:
            file_manager_instance.cleanup_expired_files()
            if not file_manager_instance.get_file_metadata(file_id):
                 raise HTTPException(status_code=404, detail=f"File with ID '{file_id}' not found or already deleted.")
            else: raise HTTPException(status_code=500, detail=f"Failed to delete file ID '{file_id}', but it might still exist. Check logs.")
    except HTTPException: raise
    except Exception as e:
        print(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    

class AnalyseResumeLangGraphRequest(BaseModel):
    file_id: Optional[str] = None
    resume_text: Optional[str] = None
    input_summary: Optional[str] = None
    input_insights: Optional[List[str]] = None

@app.post("/analyse-resume", summary="Analyze Resume via LangGraph and Stream Results")
async def analyse_resume_with_langgraph(request_data: AnalyseResumeLangGraphRequest):
    thread_id = f"resume_run_{uuid.uuid4()}"
    initial_lg_state = LangGraphResumeState(
        resume_text=None, input_summary=request_data.input_summary, input_insights=request_data.input_insights,
        work_experience=None, education=None, generated_summary=None, extracted_insights=None, generated_questions=None,
        error=None, current_node=None, execution_id=thread_id,
        skip_summary_generation=bool(request_data.input_summary), skip_insights_extraction=bool(request_data.input_insights)
    )
    if request_data.file_id:
        file_meta = file_manager_instance.get_file_metadata(request_data.file_id)
        if not file_meta or not file_meta.get("pdf_path"):
            raise HTTPException(status_code=404, detail=f"PDF for file_id '{request_data.file_id}' not found in metadata or path missing.")
        pdf_path_to_load = file_meta["pdf_path"]
        try:
            text_from_pdf = await asyncio.to_thread(get_pdf_text, [pdf_path_to_load])
            initial_lg_state['resume_text'] = text_from_pdf.strip() if text_from_pdf else ""
            if not initial_lg_state['resume_text']: print(f"Warning: No text extracted from PDF for file_id '{request_data.file_id}'.")
        except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to read PDF from '{pdf_path_to_load}': {e}")
    elif request_data.resume_text: initial_lg_state['resume_text'] = request_data.resume_text
    if not initial_lg_state.get('resume_text') and not initial_lg_state.get('input_summary') and not initial_lg_state.get('input_insights'):
        print("Warning: /analyse-resume called with no actionable input. Graph's initialize_state_node should set an error.")
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps(StreamEvent(type='graph_start', content={'checkpoint_id': thread_id, 'execution_id': thread_id}).model_dump())}\n\n"
        summary_done_sent, insights_done_sent, first_question_done_sent = False, False, False
        print(f"Starting LangGraph stream for execution_id: {thread_id}...")
        try:
            async for event in compiled_resume_analyzer_graph.astream_events(initial_lg_state, config, version="v1"):
                kind, node_name = event["event"], event.get("name", "unknown_node")
                event_data_output = event.get("data", {}).get("output")
                current_graph_state_dict = event_data_output if isinstance(event_data_output, dict) else {}
                if not isinstance(event_data_output, dict):
                    print(f"Warning: event['data']['output'] for node '{node_name}' is not a dict: {type(event_data_output)}, value: {str(event_data_output)[:200]}")
                    if isinstance(event_data_output, str) and "error" in event_data_output.lower():
                         yield f"data: {json.dumps(StreamEvent(type='error', content=f"Error from node '{node_name}': {event_data_output}").model_dump())}\n\n"
                
                if kind == "on_chain_end":
                    if node_name == "generate_summary" and not summary_done_sent:
                        summary_text = current_graph_state_dict.get("generated_summary")
                        if summary_text and not (isinstance(summary_text, str) and ("Error" in summary_text or "Cannot generate" in summary_text)):
                            yield f"data: {json.dumps(StreamEvent(type='summary_chunk', content=summary_text).model_dump())}\n\n"; yield f"data: {json.dumps(StreamEvent(type='summary_done', content='Summary generation complete.').model_dump())}\n\n"; summary_done_sent = True
                        elif summary_text: yield f"data: {json.dumps(StreamEvent(type='error', content=f'Summary Issue: {summary_text}').model_dump())}\n\n"; summary_done_sent = True
                    if node_name == "extract_insights" and not insights_done_sent:
                        insights_list = current_graph_state_dict.get("extracted_insights")
                        is_valid = isinstance(insights_list, list) and insights_list and not any("Error" in str(i) or "Cannot extract" in str(i) or "Raw insights" in str(i) for i in insights_list)
                        if is_valid: yield f"data: {json.dumps(StreamEvent(type='insights_generated', content={'count': len(insights_list)}).model_dump())}\n\n"; insights_done_sent = True
                        elif insights_list: err_content = insights_list[0] if isinstance(insights_list, list) and insights_list else str(insights_list); yield f"data: {json.dumps(StreamEvent(type='error', content=f'Insights Issue: {err_content}').model_dump())}\n\n"; insights_done_sent = True
                    if node_name == "generate_questions" and not first_question_done_sent:
                        questions_list = current_graph_state_dict.get("generated_questions")
                        is_valid = isinstance(questions_list, list) and questions_list and not any("Error" in str(q) or "Cannot generate" in str(q) or "Raw questions" in str(q) for q in questions_list)
                        if is_valid: first_q = questions_list[0]; yield f"data: {json.dumps(StreamEvent(type='question_chunk', content=first_q).model_dump())}\n\n"; yield f"data: {json.dumps(StreamEvent(type='question_done', content='First question generated.').model_dump())}\n\n"; first_question_done_sent = True
                        elif questions_list: err_content = questions_list[0] if isinstance(questions_list, list) and questions_list else str(questions_list); yield f"data: {json.dumps(StreamEvent(type='error', content=f'Question Issue: {err_content}').model_dump())}\n\n"; first_question_done_sent = True
                    if current_graph_state_dict.get("error"):
                        error_message = current_graph_state_dict["error"]
                        already_rep = (node_name=="generate_summary" and summary_done_sent) or (node_name=="extract_insights" and insights_done_sent) or (node_name=="generate_questions" and first_question_done_sent)
                        if not already_rep: yield f"data: {json.dumps(StreamEvent(type='error', content=f"Error in node '{node_name}': {error_message}").model_dump())}\n\n"
                        print(f"Error in graph (exec_id: {thread_id}, node: {node_name}): {error_message}")
                elif kind == "on_graph_end":
                    final_state = await asyncio.to_thread(compiled_resume_analyzer_graph.get_state, config)
                    final_vals = final_state.values if final_state and hasattr(final_state, 'values') else {}
                    if not summary_done_sent and final_vals.get("generated_summary"): yield f"data: {json.dumps(StreamEvent(type='summary_done', content='Summary done (at graph end).').model_dump())}\n\n"
                    if not insights_done_sent and final_vals.get("extracted_insights"): count = len(final_vals['extracted_insights']) if isinstance(final_vals['extracted_insights'], list) else 0; yield f"data: {json.dumps(StreamEvent(type='insights_generated', content={'count': count, 'at_graph_end': True}).model_dump())}\n\n"
                    if not first_question_done_sent and final_vals.get("generated_questions") and isinstance(final_vals["generated_questions"],list) and final_vals["generated_questions"]: yield f"data: {json.dumps(StreamEvent(type='question_done', content='First question done (at graph end).').model_dump())}\n\n"
                    yield f"data: {json.dumps(StreamEvent(type='graph_complete', content={'checkpoint_id': thread_id, 'final_state_summary': {'error': final_vals.get('error'), 'last_node': final_vals.get('current_node')}}).model_dump())}\n\n"
                    print(f"LangGraph exec_id: {thread_id} completed. Final node: {final_vals.get('current_node')}"); break
        except Exception as e:
            print(f"Error during LangGraph event streaming (exec_id: {thread_id}): {e}"); import traceback; traceback.print_exc()
            yield f"data: {json.dumps(StreamEvent(type='error', content=f'LangGraph streaming error: {str(e)}').model_dump())}\n\n"
        finally: print(f"Event generator for exec_id {thread_id} finished.")
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/resume-question", response_model=InterviewQuestionsResponse, summary="Resume LangGraph for Question Generation")
async def resume_question_generation_langgraph(request_data: ResumeQuestionLangGraphRequest):
    checkpoint_id_to_resume = request_data.checkpoint_id
    config_to_resume = {"configurable": {"thread_id": checkpoint_id_to_resume}}
    print(f"\n--- /resume-question for Checkpoint ID: {checkpoint_id_to_resume} ---")

    current_graph_state_snapshot = await asyncio.to_thread(compiled_resume_analyzer_graph.get_state, config_to_resume)
    if current_graph_state_snapshot is None:
        print(f"ERROR: Checkpoint {checkpoint_id_to_resume} not found (get_state returned None).")
        raise HTTPException(status_code=404, detail=f"Checkpoint with ID '{checkpoint_id_to_resume}' not found (get_state returned None).")

    current_graph_state_values = {}
    if hasattr(current_graph_state_snapshot, 'values') and isinstance(current_graph_state_snapshot.values, dict):
        current_graph_state_values = current_graph_state_snapshot.values
    elif isinstance(current_graph_state_snapshot, dict): current_graph_state_values = current_graph_state_snapshot
    else:
        print(f"ERROR: Unexpected state structure for {checkpoint_id_to_resume}. Type: {type(current_graph_state_snapshot)}")
        raise HTTPException(status_code=500, detail=f"Unexpected state structure for checkpoint '{checkpoint_id_to_resume}'. Type: {type(current_graph_state_snapshot)}")
    if not current_graph_state_values:
        print(f"ERROR: Checkpoint {checkpoint_id_to_resume} found but state values are empty.")
        raise HTTPException(status_code=404, detail=f"Checkpoint with ID '{checkpoint_id_to_resume}' found but state is empty.")
    
    print(f"Retrieved current state for {checkpoint_id_to_resume}:")
    for key, value in current_graph_state_values.items(): print(f"  {key} (type {type(value)}): {str(value)[:150]}")

    effective_input_summary = request_data.resume_summary or current_graph_state_values.get('generated_summary') or current_graph_state_values.get('input_summary')
    effective_input_insights = request_data.resume_insights or current_graph_state_values.get('extracted_insights') or current_graph_state_values.get('input_insights')

    are_effective_insights_valid = isinstance(effective_input_insights, list) and \
                                   effective_input_insights and \
                                   not any("Error" in str(i) or "Cannot extract" in str(i) or "Raw insights" in str(i) for i in effective_input_insights)
    force_insight_regen = False
    if not are_effective_insights_valid:
        print(f"INFO (resume-question): No valid insights for {checkpoint_id_to_resume}. Will attempt insights extraction.")
        is_summary_valid_for_regen = isinstance(effective_input_summary, str) and \
                                     effective_input_summary.strip() and \
                                     not ("Error" in effective_input_summary or "Cannot generate" in effective_input_summary)
        if not is_summary_valid_for_regen:
            print(f"ERROR (resume-question): Cannot generate Qs for {checkpoint_id_to_resume}: No valid insights & no valid summary for regen.")
            raise HTTPException(status_code=400, detail="Cannot generate questions: No valid insights, and no valid summary to regenerate insights.")
        force_insight_regen = True

    updated_checkpoint_state = LangGraphResumeState(
        resume_text=current_graph_state_values.get('resume_text'), 
        work_experience=current_graph_state_values.get('work_experience'),
        education=current_graph_state_values.get('education'),    
        input_summary=effective_input_summary,
        input_insights=effective_input_insights if are_effective_insights_valid and not force_insight_regen else None,
        generated_summary=current_graph_state_values.get('generated_summary'),
        extracted_insights=effective_input_insights if are_effective_insights_valid and not force_insight_regen else None,
        generated_questions=None, error=None,               
        current_node="targeting_questions_via_resume_question_endpoint", # Marker for this specific resume operation
        execution_id=checkpoint_id_to_resume,
        skip_summary_generation=True,
        skip_insights_extraction= (not force_insight_regen) and are_effective_insights_valid
    )
    
    print("\nPrepared 'updated_checkpoint_state' for /resume-question:")
    for key, value in updated_checkpoint_state.items(): print(f"  {key} (type {type(value)}): {str(value)[:150]}")
    await asyncio.to_thread(compiled_resume_analyzer_graph.update_state, config_to_resume, updated_checkpoint_state)
    
    questions_list_from_graph, final_error, invoked_state_values = [], None, {}
    try:
        print(f"Invoking graph for questions for {checkpoint_id_to_resume}...")
        invoked_state_snapshot = await asyncio.to_thread(compiled_resume_analyzer_graph.invoke, None, config_to_resume)
        print(f"Graph invocation completed for {checkpoint_id_to_resume}.")

        if invoked_state_snapshot is None: final_error = "Graph invocation returned None."
        elif hasattr(invoked_state_snapshot, 'values') and isinstance(invoked_state_snapshot.values, dict): invoked_state_values = invoked_state_snapshot.values
        elif isinstance(invoked_state_snapshot, dict): invoked_state_values = invoked_state_snapshot
        else: final_error = f"Unexpected state type after invoke: {type(invoked_state_snapshot)}"
        
        if not final_error: # Only proceed if invoked_state_values is a dict
            print(f"\nState after invoke for {checkpoint_id_to_resume}:")
            for key, value in invoked_state_values.items(): print(f"  {key} (type {type(value)}): {str(value)[:150]}")
            if invoked_state_values.get("error"): final_error = invoked_state_values["error"]
            if invoked_state_values.get("generated_questions"): questions_list_from_graph = invoked_state_values["generated_questions"]
    except Exception as e_invoke:
        final_error = f"Exception during graph invocation for questions: {str(e_invoke)}"
        print(f"EXCEPTION during invoke for {checkpoint_id_to_resume}: {e_invoke}"); import traceback; traceback.print_exc()

    if final_error: 
        print(f"Raising HTTPException for {checkpoint_id_to_resume} due to final_error: {final_error}")
        raise HTTPException(status_code=500, detail=f"Error during resumed graph execution for questions: {final_error}")
    
    is_valid_qs = isinstance(questions_list_from_graph, list) and questions_list_from_graph and \
                  not any("Error" in str(q) or "Cannot generate" in str(q) or "Raw questions" in str(q) for q in questions_list_from_graph)
    if not is_valid_qs:
        last_node = invoked_state_values.get('current_node', 'Unknown')
        err_detail = f"Resumed graph execution did not produce valid questions. Last node: {last_node}."
        if questions_list_from_graph and isinstance(questions_list_from_graph, list) and questions_list_from_graph : err_detail += f" Error in questions: {questions_list_from_graph[0]}"
        elif questions_list_from_graph: err_detail += f" Raw Qs content: {str(questions_list_from_graph)[:100]}"
        else: err_detail += " No questions were generated."
        print(f"Raising HTTPException for {checkpoint_id_to_resume} due to invalid questions: {err_detail}")
        raise HTTPException(status_code=500, detail=err_detail)
        
    print(f"Successfully generated questions for {checkpoint_id_to_resume}: {questions_list_from_graph}")
    return InterviewQuestionsResponse(questions=questions_list_from_graph)

# --- Part A: Data Extraction Endpoints ---
@app.get("/extract/work-experience/{file_id}", response_model=WorkExperienceList, summary="Extract Work Experience")
async def extract_work_experience(file_id: str):
    vector_store_path = get_valid_vector_store_path(file_id)
    result = await asyncio.to_thread(work_experience_extraction_node, vector_store_path)
    if result.get("success"): return WorkExperienceList(**result["data"])
    raise HTTPException(status_code=422, detail=f"Failed to extract work experience: {result.get('error', 'Unknown')}. Raw: {result.get('raw_response', '')[:200]}")

@app.get("/extract/education/{file_id}", response_model=EducationList, summary="Extract Education")
async def extract_education(file_id: str):
    vector_store_path = get_valid_vector_store_path(file_id)
    result = await asyncio.to_thread(education_extraction_node, vector_store_path)
    if result.get("success"): return EducationList(**result["data"])
    raise HTTPException(status_code=422, detail=f"Failed to extract education: {result.get('error', 'Unknown')}. Raw: {result.get('raw_response', '')[:200]}")

@app.get("/extract/all/{file_id}", summary="Extract All (Work Experience & Education)")
async def extract_all_structured_data(file_id: str):
    vector_store_path = get_valid_vector_store_path(file_id)
    result = await asyncio.to_thread(combined_extraction_node, vector_store_path)
    if result.get("combined_success"):
        return {
            "work_experience": WorkExperienceList(**result["work_experience"]["data"]).model_dump(),
            "education": EducationList(**result["education"]["data"]).model_dump(),
            "success": True
        }
    errors = []
    if not result.get("work_experience", {}).get("success"): errors.append(f"Work Exp Error: {result.get('work_experience', {}).get('error', 'Unknown')}")
    if not result.get("education", {}).get("success"): errors.append(f"Edu Error: {result.get('education', {}).get('error', 'Unknown')}")
    raise HTTPException(status_code=422, detail=f"Combined extraction failed: {'; '.join(errors)}")

# --- Part B: Analysis & Generation Endpoints ---
@app.get("/generate/summary-stream/{file_id}", summary="Generate Summary (Streaming)")
async def generate_summary_stream_persistent(file_id: str):
    vector_store_path = get_valid_vector_store_path(file_id)
    async def event_generator():
        for chunk in streaming_summary_generation_node(vector_store_path):
            await asyncio.sleep(0.01)
            if chunk.startswith("Error:"): yield f"data: {json.dumps(StreamEvent(type='error', content=chunk).model_dump())}\n\n"; return
            yield f"data: {json.dumps(StreamEvent(type='summary_chunk', content=chunk).model_dump())}\n\n"
        yield f"data: {json.dumps(StreamEvent(type='summary_done', content='Summary generation complete.').model_dump())}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/generate/insights/{file_id}", response_model=InsightsResponse, summary="Extract Resume Insights")
async def generate_insights_persistent(file_id: str):
    vector_store_path = get_valid_vector_store_path(file_id)
    result = await asyncio.to_thread(insights_extraction_node, vector_store_path)
    if result.get("success"): return InsightsResponse(insights=result["insights"])
    raise HTTPException(status_code=422, detail=f"Failed to extract insights: {result.get('error', 'Unknown')}. Raw: {result.get('raw_response', '')[:200]}")

@app.post("/generate/interview-questions/", response_model=InterviewQuestionsResponse, summary="Generate Interview Questions from Insights")
async def generate_interview_questions_from_insights(payload: InsightsResponse = Body(...)):
    if not payload.insights: raise HTTPException(status_code=400, detail="No insights provided.")
    result = await asyncio.to_thread(interview_question_generation_node, payload.insights)
    if result.get("success"): return InterviewQuestionsResponse(questions=result["questions"])
    raise HTTPException(status_code=422, detail=f"Failed to generate questions: {result.get('error', 'Unknown')}. Raw: {result.get('raw_response', '')[:200]}")

@app.get("/generate/summary-and-first-question-stream/{file_id}", summary="Stream Summary then First Question")
async def generate_summary_and_first_question_persistent(file_id: str):
    vector_store_path = get_valid_vector_store_path(file_id)
    async def combined_event_generator():
        yield f"data: {json.dumps(StreamEvent(type='status', content='Starting summary generation...').model_dump())}\n\n"
        summary_stream, summary_ok = streaming_summary_generation_node(vector_store_path), True
        for chunk in summary_stream:
            await asyncio.sleep(0.01)
            if chunk.startswith("Error:"): yield f"data: {json.dumps(StreamEvent(type='error', content=f'Summary failed: {chunk}').model_dump())}\n\n"; summary_ok=False; break
            yield f"data: {json.dumps(StreamEvent(type='summary_chunk', content=chunk).model_dump())}\n\n"
        if not summary_ok: return
        yield f"data: {json.dumps(StreamEvent(type='summary_done', content='Summary generation complete.').model_dump())}\n\n"
        
        yield f"data: {json.dumps(StreamEvent(type='status', content='Extracting insights...').model_dump())}\n\n"
        insights_result = await asyncio.to_thread(insights_extraction_node, vector_store_path)
        if not insights_result.get("success") or not insights_result.get("insights"):
            err_msg = insights_result.get('error', 'Failed to extract insights/no insights found.')
            yield f"data: {json.dumps(StreamEvent(type='error', content=err_msg).model_dump())}\n\n"; return
        extracted_insights = insights_result["insights"]
        yield f"data: {json.dumps(StreamEvent(type='insights_generated', content={'count': len(extracted_insights)}).model_dump())}\n\n"

        yield f"data: {json.dumps(StreamEvent(type='status', content='Generating first interview question...').model_dump())}\n\n"
        question_stream, first_q_streamed = streaming_first_question_node(extracted_insights), False
        for q_chunk in question_stream:
            await asyncio.sleep(0.01)
            if q_chunk.startswith("Error:"): yield f"data: {json.dumps(StreamEvent(type='error', content=f'Question generation failed: {q_chunk}').model_dump())}\n\n"; return
            yield f"data: {json.dumps(StreamEvent(type='question_chunk', content=q_chunk).model_dump())}\n\n"; first_q_streamed = True
        
        if first_q_streamed: yield f"data: {json.dumps(StreamEvent(type='question_done', content='First question streamed.').model_dump())}\n\n"
        else: yield f"data: {json.dumps(StreamEvent(type='error', content='No first question streamed. Check insights.').model_dump())}\n\n"
        yield f"data: {json.dumps(StreamEvent(type='process_complete', content='Full process finished for stored file.').model_dump())}\n\n"
    return StreamingResponse(combined_event_generator(), media_type="text/event-stream")

# --- Graph Visualization Endpoints ---
def create_static_graph_visualization(graph_app_obj, output_path: Path) -> Optional[str]:
    try:
        nodes = ["initialize_state", "extract_work_experience", "extract_education", "generate_summary", "extract_insights", "generate_questions", "final_node_marker"]
        edges = [("initialize_state", "extract_work_experience"), ("initialize_state", "generate_summary"), ("initialize_state", "extract_insights"), ("initialize_state", "final_node_marker"), ("extract_work_experience", "extract_education"), ("extract_education", "generate_summary"), ("generate_summary", "extract_insights"), ("generate_summary", "final_node_marker"), ("extract_insights", "generate_questions"), ("extract_insights", "final_node_marker"), ("generate_questions", "final_node_marker")]
        drawable_nodes = set()
        for u, v in edges: drawable_nodes.add(u); drawable_nodes.add(v)
        if "END" in drawable_nodes: drawable_nodes.remove("END")
        G = nx.DiGraph(); G.add_nodes_from(list(drawable_nodes))
        drawable_edges = [(u,v) for u,v in edges if u in drawable_nodes and v in drawable_nodes]
        G.add_edges_from(drawable_edges)
        if not G.nodes(): print("Warning: No nodes for graph visualization."); return None
        plt.figure(figsize=(16, 12)); pos = nx.spring_layout(G, k=0.9, iterations=30, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=4000, alpha=0.9, edgecolors='black')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrowsize=20, alpha=0.7, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        plt.title("Resume Analysis Workflow (Conceptual Flow)", size=18); plt.axis('off'); plt.tight_layout()
        plt.savefig(output_path, format="png", dpi=120); plt.close()
        print(f"Graph visualization saved to {output_path}"); return str(output_path)
    except Exception as e: print(f"Error creating graph visualization: {e}"); import traceback; traceback.print_exc(); return None

@app.on_event("startup")
async def startup_event():
    global _graph_image_path_str
    img_path = GRAPH_VIS_DIR / "resume_analysis_workflow_static.png"
    _graph_image_path_str = create_static_graph_visualization(compiled_resume_analyzer_graph, img_path)
    if _graph_image_path_str: print(f"Graph image generated at: {_graph_image_path_str}")
    else: print("Failed to generate graph image on startup.")

@app.get("/graph-image/", summary="Get a visualization of the analysis workflow graph")
async def get_graph_image():
    global _graph_image_path_str
    if _graph_image_path_str and Path(_graph_image_path_str).exists():
        with open(_graph_image_path_str, "rb") as fimg: image_bytes = fimg.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return JSONResponse(content={"image_data": encoded_image, "format": "png"})
    else:
        print("Graph image not found, attempting to regenerate...")
        img_path = GRAPH_VIS_DIR / "resume_analysis_workflow_static.png"
        new_path = create_static_graph_visualization(compiled_resume_analyzer_graph, img_path)
        if new_path and Path(new_path).exists():
            _graph_image_path_str = new_path
            with open(new_path, "rb") as fimg: image_bytes = fimg.read()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            return JSONResponse(content={"image_data": encoded_image, "format": "png"})
        else: raise HTTPException(status_code=404, detail="Graph image not found and could not be regenerated.")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    print(f"Attempting to start Comprehensive AI Resume Analyzer API...")
    print(f"File manager storage directory: {Path(STORAGE_DIRECTORY).resolve()}")
    file_manager_instance._ensure_metadata_file_exists() 
    print("FileManager initialized.")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")