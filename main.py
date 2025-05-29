# main.py
import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional
import base64 # For graph image

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

# --- Helper Functions for API Interaction ---
def handle_response(response: requests.Response) -> Optional[Dict[str, Any]]:
    if response.ok:
        try: return response.json()
        except json.JSONDecodeError: 
            st.error(f"Failed to decode JSON. Status: {response.status_code}, Text: {response.text[:200]}"); 
            return None
    else:
        error_message = f"API Error (Status {response.status_code})"
        try: 
            error_detail = response.json().get("detail", response.text)
            error_message += f": {error_detail}"
        except json.JSONDecodeError: 
            error_message += f": {response.text}"
        st.error(error_message)
        return None

def get_stored_files() -> Dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/stored-files/")
        data = handle_response(response) # handle_response now shows errors
        if data and "files" in data: return data["files"]
    except requests.exceptions.RequestException as e: 
        st.error(f"Connection error (get_stored_files): {e}")
    return {}

# --- Page Implementations ---
def page_part_a():
    st.header("Part A: File Management, Auto-Analysis & Extraction")

    if 'selected_file_id_a' not in st.session_state: st.session_state.selected_file_id_a = None
    if 'active_files_a' not in st.session_state: st.session_state.active_files_a = {}
    if 'trigger_combined_stream_a' not in st.session_state: st.session_state.trigger_combined_stream_a = False
    if 'part_a_auto_summary' not in st.session_state: st.session_state.part_a_auto_summary = None
    if 'part_a_auto_question' not in st.session_state: st.session_state.part_a_auto_question = None
    if 'part_a_auto_status' not in st.session_state: st.session_state.part_a_auto_status = None
    if 'part_a_auto_error' not in st.session_state: st.session_state.part_a_auto_error = None

    st.subheader("1. Upload Resume PDF")
    uploaded_file = st.file_uploader("Choose a PDF file to upload and process", type="pdf", key="uploader_part_a_unique_main")
    if uploaded_file:
        if st.button(f"Process '{uploaded_file.name}'", key="process_upload_part_a_unique_btn_main", type="primary"):
            with st.spinner("Uploading and processing PDF... This will also trigger initial analysis."):
                files = {'files': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                try:
                    response = requests.post(f"{API_BASE_URL}/upload-resume/", files=files, timeout=60) # Added timeout
                    data = handle_response(response)
                    if data and data.get("file_id"):
                        st.success(f"File '{data.get('original_name')}' processed! Initial analysis will start below.")
                        st.session_state.last_uploaded_file_id_global = data["file_id"]
                        st.session_state.last_uploaded_filename_global = data.get('original_name')
                        st.session_state.selected_file_id_a = data["file_id"] 
                        st.session_state.active_files_a = get_stored_files()
                        st.session_state.trigger_combined_stream_a = True
                        st.session_state.part_a_auto_summary, st.session_state.part_a_auto_question = None, None
                        st.session_state.part_a_auto_status, st.session_state.part_a_auto_error = None, None
                        st.rerun()
                    # handle_response shows error if !data or !file_id based on API response
                except requests.exceptions.Timeout: st.error("Upload timed out. Please try again.")
                except requests.exceptions.RequestException as e: st.error(f"Connection error during upload: {e}")
                except Exception as e: st.error(f"An unexpected error occurred during upload: {str(e)}")

    if st.session_state.last_uploaded_file_id_global:
        st.info(f"Current active file for analysis: **{st.session_state.last_uploaded_filename_global or 'N/A'}** (ID: `{st.session_state.last_uploaded_file_id_global[:8]}...`)")

    st.subheader("2. Manage & Select Stored Resumes")
    if st.button("Refresh Stored Files List", key="refresh_files_a_unique_btn_main"):
        st.session_state.active_files_a = get_stored_files()
        if not st.session_state.active_files_a: st.info("No active files found or failed to fetch.")

    if st.session_state.active_files_a:
        file_options = {fid: f"{data.get('original_name', 'Unknown Name')} (ID: {fid[:8]}...)" for fid, data in st.session_state.active_files_a.items()}
        default_selectbox_val = st.session_state.last_uploaded_file_id_global if st.session_state.last_uploaded_file_id_global in file_options else (st.session_state.selected_file_id_a if st.session_state.selected_file_id_a in file_options else None)
        current_selectbox_index = 0
        if default_selectbox_val and file_options:
            try: current_selectbox_index = list(file_options.keys()).index(default_selectbox_val)
            except ValueError: current_selectbox_index = 0

        selected_fid_in_selectbox = st.selectbox("Select a File ID to make it active (will trigger initial analysis):", options=list(file_options.keys()), format_func=lambda fid: file_options.get(fid, fid), index=current_selectbox_index, key="selectbox_file_id_a_unique_main")
        if selected_fid_in_selectbox and selected_fid_in_selectbox != st.session_state.last_uploaded_file_id_global:
            st.session_state.selected_file_id_a = selected_fid_in_selectbox
            st.session_state.last_uploaded_file_id_global = selected_fid_in_selectbox
            selected_meta = st.session_state.active_files_a.get(selected_fid_in_selectbox, {})
            st.session_state.last_uploaded_filename_global = selected_meta.get('original_name')
            st.session_state.trigger_combined_stream_a = True
            st.session_state.part_a_auto_summary, st.session_state.part_a_auto_question = None, None
            st.session_state.part_a_auto_status, st.session_state.part_a_auto_error = None, None
            st.rerun()
        elif selected_fid_in_selectbox and selected_fid_in_selectbox != st.session_state.selected_file_id_a :
            st.session_state.selected_file_id_a = selected_fid_in_selectbox

        if st.session_state.selected_file_id_a:
            display_name = file_options.get(st.session_state.selected_file_id_a, st.session_state.selected_file_id_a)
            st.write(f"Selected in list: `{display_name}`")
            if st.button(f"🗑️ Delete File: {display_name}", key=f"delete_btn_{st.session_state.selected_file_id_a}_unique_main"):
                with st.spinner(f"Deleting {display_name}..."):
                    try:
                        response = requests.delete(f"{API_BASE_URL}/delete-file/{st.session_state.selected_file_id_a}")
                        data = handle_response(response)
                        if data:
                            st.success(data.get("message", "Deletion successful."))
                            if st.session_state.selected_file_id_a == st.session_state.last_uploaded_file_id_global:
                                st.session_state.last_uploaded_file_id_global, st.session_state.last_uploaded_filename_global = None, None
                                st.session_state.part_a_auto_summary, st.session_state.part_a_auto_question = None, None
                                st.session_state.part_a_auto_status, st.session_state.part_a_auto_error = None, None
                            st.session_state.selected_file_id_a = None
                            st.session_state.active_files_a = get_stored_files(); st.rerun()
                    except requests.exceptions.RequestException as e: st.error(f"Connection error during deletion: {e}")
    else: st.info("No stored files available. Upload a resume or click 'Refresh'.")

    st.subheader("3. Initial AI Analysis (Summary & First Question)")
    analysis_output_container = st.container()
    with analysis_output_container:
        if st.session_state.part_a_auto_error: st.error(f"Error during last auto-analysis: {st.session_state.part_a_auto_error}")
        if st.session_state.part_a_auto_summary not in [None, ""]: st.markdown("**Final Summary:**"); st.markdown(st.session_state.part_a_auto_summary)
        if st.session_state.part_a_auto_question not in [None, ""]: st.markdown("**Final First Question:**"); st.markdown(st.session_state.part_a_auto_question)
        if st.session_state.part_a_auto_status: st.success(st.session_state.part_a_auto_status)

        if st.session_state.trigger_combined_stream_a and st.session_state.last_uploaded_file_id_global:
            active_file_id_for_stream = st.session_state.last_uploaded_file_id_global
            active_filename_for_stream = st.session_state.last_uploaded_filename_global or f"ID: {active_file_id_for_stream[:8]}..."
            st.write(f"🚀 Performing initial analysis for: **{active_filename_for_stream}**")
            summary_stream_placeholder, question_stream_placeholder, status_stream_placeholder = st.empty(), st.empty(), st.empty()
            current_summary_accumulator, current_question_accumulator = "", ""
            st.session_state.part_a_auto_summary, st.session_state.part_a_auto_question = "", ""
            st.session_state.part_a_auto_status, st.session_state.part_a_auto_error = None, None
            with st.spinner(f"Streaming summary & first question for {active_filename_for_stream}..."):
                try:
                    response = requests.get(f"{API_BASE_URL}/generate/summary-and-first-question-stream/{active_file_id_for_stream}", stream=True, timeout=120)
                    if response.ok:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith('data:'):
                                    try:
                                        event = json.loads(decoded_line[len('data:'):].strip())
                                        event_type, content = event.get("type"), event.get("content")
                                        if event_type == "status": status_stream_placeholder.info(content)
                                        elif event_type == "summary_chunk": current_summary_accumulator += content + " "; summary_stream_placeholder.markdown(f"**Summary (streaming):**\n\n{current_summary_accumulator.strip()}")
                                        elif event_type == "summary_done": status_stream_placeholder.success("Summary generation complete."); st.session_state.part_a_auto_summary = current_summary_accumulator.strip(); summary_stream_placeholder.markdown(f"**Final Summary:**\n\n{st.session_state.part_a_auto_summary}")
                                        elif event_type == "insights_generated": status_stream_placeholder.info(f"Insights extracted (count: {content.get('count', 'N/A')}). Generating question...")
                                        elif event_type == "question_chunk": current_question_accumulator += content + " "; question_stream_placeholder.markdown(f"**First Question (streaming):**\n\n{current_question_accumulator.strip()}")
                                        elif event_type == "question_done": status_stream_placeholder.success("First question streaming complete."); st.session_state.part_a_auto_question = current_question_accumulator.strip(); question_stream_placeholder.markdown(f"**Final First Question:**\n\n{st.session_state.part_a_auto_question}")
                                        elif event_type == "process_complete": st.session_state.part_a_auto_status = content; status_stream_placeholder.success(content); break 
                                        elif event_type == "error": st.session_state.part_a_auto_error = f"Stream Error: {content}"; status_stream_placeholder.error(st.session_state.part_a_auto_error); break
                                    except json.JSONDecodeError: st.session_state.part_a_auto_error = f"Stream parsing error: {decoded_line}"; status_stream_placeholder.warning(st.session_state.part_a_auto_error)
                    else: api_error_content = handle_response(response); st.session_state.part_a_auto_error = f"API Error before stream: {api_error_content or response.text}"
                except requests.exceptions.RequestException as e: st.session_state.part_a_auto_error = f"Connection error during combined stream: {e}"
            st.session_state.trigger_combined_stream_a = False; st.rerun() 
        elif not st.session_state.last_uploaded_file_id_global and not st.session_state.part_a_auto_summary and not st.session_state.part_a_auto_question:
             st.warning("Upload or select a PDF to start the initial analysis.")

    st.subheader("4. Extract Specific Structured Data (On Demand)")
    if not st.session_state.last_uploaded_file_id_global: st.warning("Please upload or select a File ID to perform specific extractions.")
    else:
        active_file_id_for_extraction = st.session_state.last_uploaded_file_id_global
        active_filename_for_extraction = st.session_state.last_uploaded_filename_global or f"ID: {active_file_id_for_extraction[:8]}..."
        st.write(f"Performing extractions for: **{active_filename_for_extraction}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Extract Work Experience", key="extract_work_part_a_btn_unique_main"):
                with st.spinner("Extracting work experience..."):
                    json_output_placeholder_col1 = st.empty() 
                    try: 
                        response = requests.get(f"{API_BASE_URL}/extract/work-experience/{active_file_id_for_extraction}")
                        data = handle_response(response)
                        if data: json_output_placeholder_col1.json(data)
                    except requests.exceptions.RequestException as e: st.error(f"Connection error: {e}")
        with col2:
            if st.button("Extract Education", key="extract_edu_part_a_btn_unique_main"):
                with st.spinner("Extracting education..."):
                    json_output_placeholder_col2 = st.empty()
                    try: 
                        response = requests.get(f"{API_BASE_URL}/extract/education/{active_file_id_for_extraction}")
                        data = handle_response(response)
                        if data: json_output_placeholder_col2.json(data)
                    except requests.exceptions.RequestException as e: st.error(f"Connection error: {e}")
        with col3:
            if st.button("Extract All (Work & Edu)", key="extract_all_part_a_btn_unique_main"):
                with st.spinner("Extracting all structured data..."):
                    json_output_placeholder_col3 = st.empty()
                    try: 
                        response = requests.get(f"{API_BASE_URL}/extract/all/{active_file_id_for_extraction}")
                        data = handle_response(response)
                        if data: json_output_placeholder_col3.json(data)
                    except requests.exceptions.RequestException as e: st.error(f"Connection error: {e}")
    st.markdown("---")


def page_part_b():
    st.header("Part B: Detailed Analysis & Generation")
    if 'extracted_insights_b' not in st.session_state: st.session_state.extracted_insights_b = []
    st.subheader("1. Active Resume for Detailed Analysis")
    if st.session_state.last_uploaded_file_id_global and st.session_state.last_uploaded_filename_global:
        st.info(f"Currently active: **{st.session_state.last_uploaded_filename_global}** (ID: `{st.session_state.last_uploaded_file_id_global[:8]}...`)")
        st.caption("Change active file in Part A.")
        current_analysis_file_id = st.session_state.last_uploaded_file_id_global
    else: st.warning("No resume active. Please upload or select one in Part A."); return
    st.markdown("---"); st.subheader(f"2. Actions for: {st.session_state.last_uploaded_filename_global}")
    if st.button("Re-Generate Summary (Stream)", key="regen_summary_b_btn_unique_main"):
        summary_ph_b = st.empty(); full_summary_b = ""
        with st.spinner("Streaming summary..."):
            try:
                response = requests.get(f"{API_BASE_URL}/generate/summary-stream/{current_analysis_file_id}", stream=True)
                if response.ok:
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith('data:'):
                                try:
                                    event = json.loads(decoded[len('data:'):].strip())
                                    if event.get("type") == "summary_chunk": full_summary_b += event.get("content", "") + " "; summary_ph_b.markdown(full_summary_b)
                                    elif event.get("type") == "summary_done": st.success(event.get("content", "Summary complete.")); break
                                    elif event.get("type") == "error": st.error(f"Stream error: {event.get('content')}"); break
                                except json.JSONDecodeError: st.warning(f"Parse error: {decoded}")
                else: handle_response(response) # Shows st.error
            except requests.exceptions.RequestException as e: st.error(f"Connection error: {e}")
        if full_summary_b: summary_ph_b.markdown(f"**Final Summary:**\n\n{full_summary_b}")
    st.markdown("---")
    if st.button("Extract Detailed Insights", key="gen_insights_b_btn_unique_main"):
        with st.spinner("Extracting insights..."):
            try:
                response = requests.get(f"{API_BASE_URL}/generate/insights/{current_analysis_file_id}")
                data = handle_response(response)
                if data and "insights" in data: 
                    st.session_state.extracted_insights_b = data["insights"]
                    st.success(f"Extracted {len(data['insights'])} insights.")
                    st.expander("View Insights", expanded=True).json(data["insights"])
                else: st.session_state.extracted_insights_b = [] # Clear if extraction fails or returns no insights
            except requests.exceptions.RequestException as e: st.error(f"Connection error: {e}")
    st.subheader("3. Generate Interview Questions from Extracted Insights")
    insights_input_b = st.text_area("Resume Insights (auto-filled, one per line)", value="\n".join(st.session_state.extracted_insights_b), height=150, key="insights_text_part_b_field_unique_main")
    if st.button("Generate Questions from these Insights", key="gen_questions_b_btn_unique_main"):
        if not insights_input_b.strip(): st.warning("Please provide insights.")
        else:
            insights_list = [line.strip() for line in insights_input_b.split("\n") if line.strip()]
            with st.spinner("Generating questions..."):
                try:
                    payload = {"insights": insights_list}
                    response = requests.post(f"{API_BASE_URL}/generate/interview-questions/", json=payload)
                    data = handle_response(response)
                    if data and "questions" in data: st.success(f"Generated {len(data['questions'])} questions."); st.expander("View Questions", expanded=True).json(data["questions"])
                except requests.exceptions.RequestException as e: st.error(f"Connection error: {e}")
    st.markdown("---")

def page_part_c():
    st.header("Part C: Advanced LangGraph Analysis")
    if 'checkpoint_id_c_active' not in st.session_state: st.session_state.checkpoint_id_c_active = None
    if 'stream_output_c_current' not in st.session_state: st.session_state.stream_output_c_current = []
    if 'extracted_insights_b' not in st.session_state: st.session_state.extracted_insights_b = []
    stream_output_placeholder_c = st.empty()
    def display_streamed_events_in_placeholder_c(events_list):
        with stream_output_placeholder_c.container():
            st.subheader("LangGraph Live Output:")
            if not events_list: st.info("No events from LangGraph stream yet or an action is pending."); return
            for event_obj in events_list:
                event_type, content = event_obj.get("type"), event_obj.get("content")
                if event_type == "graph_start": st.write(f"🔹 **Graph Started!** Checkpoint/Exec ID: `{content.get('checkpoint_id')}`"); st.caption("Use this ID in 'Resume with Checkpoint ID' tab.")
                elif event_type == "summary_chunk": st.markdown(f"📄 **Summary:** {content}")
                elif event_type == "summary_done": st.success(f"📄 **Summary Done:** {content}")
                elif event_type == "insights_generated": st.info(f"💡 **Insights:** Count: {content.get('count', 'N/A')}")
                elif event_type == "question_chunk": st.markdown(f"❓ **Question:** {content}")
                elif event_type == "question_done": st.success(f"❓ **First Q Done:** {content}")
                elif event_type == "graph_complete": st.balloons(); st.success(f"🎉 **Graph Complete!** Node: {content.get('final_state_summary',{}).get('last_node')}"); err = content.get('final_state_summary',{}).get('error'); st.warning(f"Graph completed with error: {err}") if err else None
                elif event_type == "error": st.error(f"❌ **LangGraph Error:** {content}")
                elif event_type == "questions_from_checkpoint": st.success("✅ **Qs from Checkpoint:**"); st.json(content)
                else: st.write(f"ℹ️ **Event ({event_type}):** `{str(content)[:300]}`")
    def stream_lg_analysis_call_c(payload: Dict[str, Any]):
        st.session_state.stream_output_c_current = []
        display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)
        with st.spinner("🚀 Engaging LangGraph Analysis Engine..."):
            try:
                response = requests.post(f"{API_BASE_URL}/analyse-resume", json=payload, stream=True, timeout=300)
                if response.ok:
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith('data:'):
                                try:
                                    event = json.loads(decoded[len('data:'):].strip())
                                    st.session_state.stream_output_c_current.append(event)
                                    if event.get("type") == "graph_start": st.session_state.checkpoint_id_c_active = event.get("content", {}).get("checkpoint_id")
                                    display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current); time.sleep(0.02)
                                except json.JSONDecodeError: st.session_state.stream_output_c_current.append({"type": "error", "content": f"Parse error: {decoded}"}); display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)
                else: err_resp = handle_response(response); st.session_state.stream_output_c_current.append({"type": "error", "content": f"API pre-stream error: {err_resp or response.text}"})
            except requests.exceptions.RequestException as e: st.session_state.stream_output_c_current.append({"type": "error", "content": f"Connection error: {e}"})
            except Exception as e_gen: st.session_state.stream_output_c_current.append({"type": "error", "content": f"General error: {e_gen}"})
        display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)

    tab1_c, tab2_c, tab3_c = st.tabs(["🧠 Analyze Selected PDF", "✍️ Analyze Raw Text", "🆔 Resume with Checkpoint ID"])
    with tab1_c:
        st.subheader("Analyze Currently Selected PDF via LangGraph")
        if st.session_state.last_uploaded_file_id_global and st.session_state.last_uploaded_filename_global:
            st.info(f"Will analyze: **{st.session_state.last_uploaded_filename_global}** (ID: `{st.session_state.last_uploaded_file_id_global[:8]}...`)")
            if st.button("Start Full LangGraph Analysis", key="analyze_selected_pdf_c_btn_unique_main", type="primary"):
                st.session_state.stream_output_c_current = []
                stream_lg_analysis_call_c({"file_id": st.session_state.last_uploaded_file_id_global})
        else: st.warning("No PDF selected/uploaded. Please use Part A to upload or select a PDF.")
    with tab2_c:
        st.subheader("Analyze Raw Resume Text via LangGraph")
        resume_text_c = st.text_area("Paste resume text here:", height=250, key="raw_text_c_input_field_unique_tab2_main")
        if st.button("Analyze Raw Text with LangGraph", key="analyze_text_c_button_field_unique_tab2_main", type="primary"):
            st.session_state.stream_output_c_current = [] 
            if resume_text_c.strip(): stream_lg_analysis_call_c({"resume_text": resume_text_c.strip()})
            else: st.warning("Please paste some resume text."); 
        display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)
    with tab3_c:
        st.subheader("Generate Questions from LangGraph Checkpoint")
        checkpoint_id_input_val = st.session_state.checkpoint_id_c_active or ""
        checkpoint_id_input = st.text_input("Enter Checkpoint ID (Thread ID):", value=checkpoint_id_input_val, key="checkpoint_input_c_field_val_unique_tab3_main")
        st.caption("This ID is auto-filled from the last LangGraph analysis run on this page.")
        override_summary_c = st.text_area("Optional: Resume Summary", key="override_summary_c_field_val_unique_tab3_main", height=100)
        default_insights_str_c = "\n".join(st.session_state.extracted_insights_b) if st.session_state.extracted_insights_b else ""
        if default_insights_str_c: st.info("Insights from Part B are pre-filled below. You can use or modify them.")
        override_insights_str_c = st.text_area("Optional: Resume Insights (one per line)", value=default_insights_str_c, key="override_insights_c_field_val_unique_tab3_main", height=100)
        if st.button("Generate Questions from Checkpoint", key="resume_graph_c_button_field_unique_tab3_main", type="primary"):
            st.session_state.stream_output_c_current = [] 
            if not checkpoint_id_input.strip(): st.warning("Please enter a Checkpoint ID."); st.session_state.stream_output_c_current.append({"type":"error", "content":"Checkpoint ID is required."})
            else:
                payload = {"checkpoint_id": checkpoint_id_input.strip()}
                if override_summary_c.strip(): payload["resume_summary"] = override_summary_c.strip()
                if override_insights_str_c.strip(): payload["resume_insights"] = [line.strip() for line in override_insights_str_c.split("\n") if line.strip()]
                with st.spinner("🔄 Requesting questions..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/resume-question", json=payload)
                        data = handle_response(response)
                        if data and data.get("questions"): st.session_state.stream_output_c_current.append({"type":"questions_from_checkpoint", "content": data["questions"]})
                        elif data and not data.get("questions"): st.session_state.stream_output_c_current.append({"type":"error", "content": f"API response did not contain questions. Details: {data.get('detail', str(data))}"})
                    except requests.exceptions.RequestException as e: st.session_state.stream_output_c_current.append({"type":"error", "content": f"Connection error (resume-question): {e}"})
            display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)
        else: display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)
    if 'part_c_just_loaded' not in st.session_state : st.session_state.part_c_just_loaded = True; display_streamed_events_in_placeholder_c(st.session_state.stream_output_c_current)
    elif not any(st.session_state.stream_output_c_current): display_streamed_events_in_placeholder_c([])
    st.markdown("---")

# --- Main App Structure ---
st.set_page_config(layout="wide", page_title="AI Resume Analyzer UI")
st.sidebar.title("Resume Analyzer")
if 'api_health_checked' not in st.session_state: st.session_state.api_health_checked = False
if 'graph_image_b64' not in st.session_state: st.session_state.graph_image_b64 = None
if 'last_uploaded_file_id_global' not in st.session_state: st.session_state.last_uploaded_file_id_global = None
if 'last_uploaded_filename_global' not in st.session_state: st.session_state.last_uploaded_filename_global = None
if 'part_a_auto_summary' not in st.session_state: st.session_state.part_a_auto_summary = None
if 'part_a_auto_question' not in st.session_state: st.session_state.part_a_auto_question = None
if 'part_a_auto_status' not in st.session_state: st.session_state.part_a_auto_status = None
if 'part_a_auto_error' not in st.session_state: st.session_state.part_a_auto_error = None
if 'extracted_insights_b' not in st.session_state: st.session_state.extracted_insights_b = []
if 'checkpoint_id_c_active' not in st.session_state: st.session_state.checkpoint_id_c_active = None
if 'stream_output_c_current' not in st.session_state: st.session_state.stream_output_c_current = []
if 'current_page_for_part_c_reset' not in st.session_state: st.session_state.current_page_for_part_c_reset = None

page_options = {"Part A: File Mgmt & Auto-Analysis": page_part_a, "Part B: Detailed Analysis": page_part_b, "Part C: LangGraph Analysis Engine": page_part_c}
selected_page_name = st.sidebar.radio("Choose a section:", list(page_options.keys()))

if not st.session_state.api_health_checked:
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_response.ok: st.sidebar.success(f"API Status: {health_response.json().get('status', 'OK')}")
        else:
            error_detail_health = f"API Status: Error ({health_response.status_code})"
            try: error_detail_health += f" - {health_response.json().get('detail', health_response.text)}"
            except json.JSONDecodeError: error_detail_health += f" - {health_response.text}"
            st.sidebar.error(error_detail_health)
    except requests.exceptions.RequestException: st.sidebar.error("API Status: Unreachable")
    st.session_state.api_health_checked = True

with st.sidebar:
    st.divider(); st.header("⚙️ Workflow Graph")
    if st.session_state.graph_image_b64:
        try: st.image(base64.b64decode(st.session_state.graph_image_b64), caption="Analysis Workflow (Conceptual)", use_container_width=True)
        except Exception as e_img: st.warning(f"Graph display error: {e_img}")
    if st.button("Load/Refresh Workflow Graph", use_container_width=True, key="refresh_graph_btn_unique_sidebar_key_final_main"):
        with st.spinner("Loading graph..."):
            try:
                img_res = requests.get(f"{API_BASE_URL}/graph-image/")
                if img_res.ok: st.session_state.graph_image_b64 = img_res.json().get("image_data"); st.rerun()
                else: st.error(f"Graph load failed (Status: {img_res.status_code}). Details: {img_res.text[:200]}")
            except requests.exceptions.RequestException as e_fgraph: st.error(f"Connection error fetching graph: {e_fgraph}")
    st.markdown("---"); st.info("AI Resume Analysis Interface.")

if 'current_page_for_part_c_reset' not in st.session_state: st.session_state.current_page_for_part_c_reset = selected_page_name
if st.session_state.current_page_for_part_c_reset != selected_page_name:
    if st.session_state.current_page_for_part_c_reset == "Part C: LangGraph Analysis Engine": st.session_state.part_c_just_loaded = False
    st.session_state.current_page_for_part_c_reset = selected_page_name
page_options[selected_page_name]()