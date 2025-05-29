# Resume Analysis Workflow with LangGraph, LangChain, and FastAPI

## Overview

This project implements a LangGraph-powered API system designed to comprehensively analyze resumes. It extracts structured data, generates summaries and insights, creates tailored interview questions, and demonstrates workflow persistence using LangGraph's checkpointing capabilities. The system is built using Python, LangGraph, LangChain, FastAPI, and Streamlit for a user interface.

The core functionalities include:
1.  **Resume Analysis:** Processing uploaded PDF resumes.
2.  **Structured Data Extraction:** Parsing work experience and education details, validated by Pydantic models.
3.  **Summarization:** Generating professional summaries of the resume content.
4.  **Insight Extraction:** Identifying key strengths and highlights from the resume.
5.  **Interview Question Generation:** Creating relevant questions based on extracted insights.
6.  **Streaming Output:** Asynchronously streaming the summary and the first interview question via a single API call.
7.  **Workflow Persistence:** Utilizing LangGraph checkpointing to resume analysis from specific stages.

## Features

*   **PDF Upload & Processing:** Securely upload PDF resumes, which are then processed to extract text and build a vector store for efficient information retrieval.
*   **Persistent Storage:** Uploaded resumes and their associated vector stores are saved, managed with expiry dates, and can be listed or deleted.
*   **Structured Data Extraction:**
    *   Extracts **Work Experience** (company, role, dates, description).
    *   Extracts **Education** (institution, degree, field, years).
    *   Uses Pydantic models for robust data validation.
*   **AI-Powered Analysis:**
    *   **Comprehensive Summary Generation:** Creates an "elevator pitch" style professional summary.
    *   **Impactful Insight Extraction:** Identifies 3-5 key insights about the candidate.
    *   **Tailored Interview Questions:** Generates 3-5 interview questions based on the extracted insights.
*   **Streaming API Endpoints:**
    *   Streams summaries word-by-word.
    *   Streams the first interview question after insights are generated.
    *   Provides a combined endpoint to stream summary, then insights status, then the first question.
*   **LangGraph Orchestration:**
    *   A defined LangGraph Directed Acyclic Graph (DAG) manages the flow of resume analysis.
    *   Nodes for initialization, text extraction (simplified for graph), summary, insights, and question generation.
    *   Conditional routing based on input data (e.g., skipping summary generation if one is provided).
*   **Checkpointing & Resumption:**
    *   The LangGraph workflow supports checkpointing, allowing the state to be saved.
    *   A dedicated API endpoint (`/resume-question`) allows resuming the graph execution, specifically targeting question generation, using a checkpoint ID and optionally providing a pre-computed summary or insights.
*   **Interactive Streamlit UI:**
    *   **Part A:** File management (upload, list, delete), auto-analysis triggering summary/first question stream, and on-demand structured data extraction.
    *   **Part B:** Detailed analysis for an active file (re-generate summary, extract insights, generate questions from insights).
    *   **Part C:** Interface for the LangGraph engine:
        *   Analyze selected PDF through the full graph.
        *   Analyze raw pasted text through the full graph.
        *   Resume graph from a checkpoint ID to generate questions, with options to override summary/insights.
*   **Graph Visualization:** Displays a conceptual image of the analysis workflow.

## Technology Stack

*   **Backend:** Python, FastAPI
*   **AI/LLM Orchestration:** LangGraph, LangChain
*   **LLM Provider:** Google Generative AI (Gemini models)
*   **Vector Store:** FAISS (via `langchain_community`)
*   **Embeddings:** GoogleGenerativeAIEmbeddings (`models/embedding-001`)
*   **PDF Processing:** PyPDF2
*   **Data Validation:** Pydantic
*   **Frontend (UI):** Streamlit
*   **API Client:** `requests`
*   **Environment Management:** `python-dotenv`
*   **Graph Visualization:** Matplotlib, NetworkX

## Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   A Google Cloud Project with the Generative AI API enabled.
*   A `GOOGLE_API_KEY` from your Google Cloud Project.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project and add your Google API Key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

## Running the Application

The application consists of two main parts: the FastAPI backend and the Streamlit frontend.

1.  **Start the FastAPI Backend:**
    Open a terminal and run:
    ```bash
    uvicorn app1:app --host 127.0.0.1 --port 8000 --reload
    ```
    The API will be available at `http://127.0.0.1:8000`. You can access the OpenAPI documentation at `http://127.0.0.1:8000/docs`.

2.  **Start the Streamlit Frontend:**
    Open another terminal and run:
    ```bash
    streamlit run main.py
    ```
    The Streamlit UI will be available at `http://localhost:8501` (or another port if 8501 is busy).

## API Endpoints

The FastAPI application (`app1.py`) exposes several endpoints. Key endpoints as per the assignment requirements and implementation:

### File Management & Basic Analysis (Vector Store Based)

*   **`POST /upload-resume/`**:
    *   Description: Uploads one or more PDF resume files. Processes the first PDF to create a text corpus and a FAISS vector store. Saves the PDF and vector store persistently.
    *   Request: `multipart/form-data` with `files` field.
    *   Response: `FileUploadResponse` (JSON with `file_id`, `original_name`, etc.)
*   **`GET /stored-files/`**:
    *   Description: Lists all actively stored (non-expired) resume files and their metadata.
    *   Response: `StoredFilesListResponse` (JSON)
*   **`GET /generate/summary-stream/{file_id}`**:
    *   Description: Streams a generated summary for the specified `file_id`.
    *   Response: Server-Sent Events (`text/event-stream`) with `summary_chunk` and `summary_done` events.
*   **`POST /generate/interview-questions/`**:
    *   Description: Generates interview questions based on a provided list of insights.
    *   Request Body: `InsightsResponse` (JSON with `{"insights": ["insight1", ...]}`)
    *   Response: `InterviewQuestionsResponse` (JSON with `{"questions": ["q1", ...]}`)
*   **`GET /generate/summary-and-first-question-stream/{file_id}`**:
    *   Description: Asynchronously streams the summary, then insight extraction status, then the first interview question for the given `file_id`.
    *   Response: Server-Sent Events (`text/event-stream`) with `status`, `summary_chunk`, `summary_done`, `insights_generated`, `question_chunk`, `question_done`, `process_complete`, `error` events.

### LangGraph Endpoints

*   **`POST /analyse-resume`**:
    *   Description: Accepts resume text (or a `file_id` to load text from an uploaded PDF) and streams the analysis results from the LangGraph workflow (summary, insights status, first question). Can also accept `input_summary` or `input_insights` to skip parts of the graph.
    *   Request Body: `AnalyseResumeLangGraphRequest` (JSON)
        ```json
        // Option 1: Provide file_id
        { "file_id": "your_file_id_here" }
        // Option 2: Provide raw text
        { "resume_text": "Detailed resume content..." }
        // Option 3: Provide text and pre-computed summary
        { "resume_text": "...", "input_summary": "A professional summary." }
        // Option 4: Provide text, summary, and insights
        { "resume_text": "...", "input_summary": "...", "input_insights": ["Insight 1", "Insight 2"] }
        ```
    *   Response: Server-Sent Events (`text/event-stream`) from the LangGraph, including `graph_start` (with `checkpoint_id`), `summary_chunk`, `summary_done`, `insights_generated`, `question_chunk`, `question_done`, `graph_complete`, `error` events.

*   **`POST /resume-question`**:
    *   Description: Accepts a `checkpoint_id` from a previous `/analyse-resume` run to resume the LangGraph. Primarily used to generate all interview questions. Can also accept `resume_summary` or `resume_insights` to override the state from the checkpoint or provide necessary data if missing.
    *   Request Body: `ResumeQuestionLangGraphRequest` (JSON)
        ```json
        // Basic resumption
        { "checkpoint_id": "your_checkpoint_id_from_analyse_resume" }

        // Resumption with overriding summary
        {
          "checkpoint_id": "your_checkpoint_id",
          "resume_summary": "An updated or specific summary to use."
        }

        // Resumption providing insights (if insights step was skipped or needs override)
        {
          "checkpoint_id": "your_checkpoint_id",
          "resume_insights": ["Insight A from prior analysis", "Insight B"]
        }
        ```
    *   Response: `InterviewQuestionsResponse` (JSON with `{"questions": ["full list of questions..."]}`)

### Other Endpoints
The API also includes endpoints for:
*   `GET /health`: Health check.
*   `DELETE /delete-file/{file_id}`: Delete a stored file.
*   `GET /extract/work-experience/{file_id}`, `GET /extract/education/{file_id}`, `GET /extract/all/{file_id}`: For specific structured data extraction using the vector store.
*   `GET /graph-image/`: Get a base64 encoded PNG of the workflow graph.

## LangGraph Workflow Overview

The core analysis pipeline is orchestrated by a LangGraph DAG defined in `graph_utils.py`.

**State (`LangGraphResumeState`):**
The graph maintains state including:
*   Input: `resume_text`, `input_summary`, `input_insights`.
*   Intermediate/Output: `work_experience`, `education`, `generated_summary`, `extracted_insights`, `generated_questions`.
*   Control/Metadata: `error`, `current_node`, `execution_id`, `skip_summary_generation`, `skip_insights_extraction`.

**Nodes:**
1.  `initialize_state`: Sets up the initial state, checks for necessary inputs, and determines skip flags.
2.  `extract_work_experience` (text-based for graph): Extracts work experience from raw text.
3.  `extract_education` (text-based for graph): Extracts education from raw text.
4.  `generate_summary`: Generates a professional summary. Can be skipped if `input_summary` is provided.
5.  `extract_insights`: Extracts key insights from the summary. Can be skipped if `input_insights` are provided and valid.
6.  `generate_questions`: Generates interview questions based on insights.
7.  `final_node_marker`: Marks the end of the process.

**Conditional Edges:**
The graph uses conditional logic to route execution:
*   After `initialize_state`:
    *   If `input_insights` are valid, jump to `generate_questions`.
    *   Else if `input_summary` is valid, jump to `extract_insights`.
    *   Else if `resume_text` is available, proceed to `extract_work_experience`.
    *   Otherwise, or if an error occurs, go to `final_node_marker`.
*   After `generate_summary`: If summary generation fails, go to `final_node_marker`; otherwise, proceed to `extract_insights`.
*   After `extract_insights`: If insights are invalid, go to `final_node_marker`; otherwise, proceed to `generate_questions`.

**Checkpointing:**
*   The graph is compiled with `MemorySaver` for in-memory checkpointing.
*   The `execution_id` (thread_id) serves as the checkpoint ID.
*   The `/analyse-resume` endpoint initiates a new graph run and returns the `checkpoint_id` in the first stream event.
*   The `/resume-question` endpoint uses this `checkpoint_id` to load the graph's state and invoke it, effectively resuming or continuing from where it makes sense to generate questions (potentially re-evaluating insights if needed).

## How to Test / Use

1.  **Start Backend and Frontend:** Follow the "Running the Application" steps.
2.  **Access Streamlit UI:** Open `http://localhost:8501` in your browser.

### Part A: File Management & Auto-Analysis
    *   **Upload:** Use the file uploader to select a PDF resume. Click "Process".
    *   The file will be uploaded, processed, and an initial analysis (summary + first question stream) will automatically start and display. The active file ID will be shown.
    *   **Manage Files:** Refresh the list to see stored files. Select a file from the dropdown to make it active for analysis (this also triggers the initial summary/question stream). Delete files using the "🗑️ Delete File" button.
    *   **Specific Extractions:** With an active file, click buttons like "Extract Work Experience" to get structured JSON output for that specific file.

### Part B: Detailed Analysis
    *   Ensure a file is active (selected/uploaded in Part A).
    *   **Re-Generate Summary:** Click to get a new streamed summary for the active file.
    *   **Extract Detailed Insights:** Click to get a list of insights. These will be auto-filled into the text area below.
    *   **Generate Interview Questions:** Modify the auto-filled insights if needed, then click to generate questions based on them.

### Part C: LangGraph Analysis Engine
    *   **Analyze Selected PDF:** If a PDF is active from Part A, click "Start Full LangGraph Analysis". Output from the graph (summary, insights, first question, checkpoint ID) will stream in the "LangGraph Live Output" area.
    *   **Analyze Raw Text:** Paste resume text into the text area and click "Analyze Raw Text with LangGraph".
    *   **Resume with Checkpoint ID:**
        *   After running an analysis in the first two tabs of Part C, a "Checkpoint ID" will be displayed in the live output. This ID is usually auto-filled here.
        *   Optionally, provide a summary or insights to override/supplement the state from the checkpoint.
        *   Click "Generate Questions from Checkpoint". This will call the `/resume-question` endpoint to get a full list of questions, resuming the LangGraph. The results will appear in the live output area.

### Direct API Testing (e.g., using `curl` or Postman)

*   **Upload a resume:**
    ```bash
    curl -X POST -F "files=@/path/to/your/resume.pdf" http://127.0.0.1:8000/upload-resume/
    ```
    Note the `file_id` from the response.

*   **Run LangGraph analysis with `file_id` (streaming):**
    ```bash
    curl -N -X POST -H "Content-Type: application/json" -d '{"file_id": "YOUR_FILE_ID_FROM_UPLOAD"}' http://127.0.0.1:8000/analyse-resume
    ```
    Observe the Server-Sent Events. Note the `checkpoint_id` from the `graph_start` event.

*   **Resume LangGraph for questions:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"checkpoint_id": "YOUR_CHECKPOINT_ID"}' http://127.0.0.1:8000/resume-question
    ```
    This should return a JSON list of interview questions.

*   **Test combined summary & first question stream (non-LangGraph):**
    ```bash
    curl -N http://127.0.0.1:8000/generate/summary-and-first-question-stream/YOUR_FILE_ID_FROM_UPLOAD
    ```

## Future Improvements / Notes

*   **Error Handling:** While basic error handling is present, it could be further enhanced for robustness across all components.
*   **LangGraph Extraction Nodes:** The current LangGraph nodes for work experience and education extraction (`extract_work_experience_node`, `extract_education_node` in `graph_utils.py`) are simplified to take raw text and return text. For full Pydantic-validated JSON output *within the graph*, these nodes would need to be adapted to perform JSON extraction and validation similar to `extraction_nodes.py`.
*   **Security:** For a production environment, add authentication and authorization to API endpoints.
*   **Scalability:** Consider more robust checkpointing (e.g., Redis, SQL database) and task queuing for handling many concurrent requests.
*   **Model Configuration:** Allow selection of different LLM models or temperature settings via API/UI.
