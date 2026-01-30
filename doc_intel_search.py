import os
import re
import hashlib
import time
from pathlib import Path

from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPowerPointLoader
import pandas as pd

from config.config_validator import validate_ini_file
from config.prompts import Final_Prompt
from utils import (
    list_directories_sorted_by_modified_time,
    return_embeddings_model,
    create_document_loader,
    get_file_types_csv,
    calculate_percentiles,
    inject_user_info,
    query_openai_mult,
)
from logger.logger_setup import logger

CONFIG_LOC = os.environ.get("CONFIG_LOC", str(Path(__file__).parent / "config" / "config.ini"))
CONFIG = validate_ini_file(CONFIG_LOC)
doc_directory_option = CONFIG["files_location"]["uploads_dir"]

last_two_questions = ["", ""]


def generate_unique_string_hash(input_string):
    hash_object = hashlib.sha1(input_string.encode())
    return hash_object.hexdigest()


def list_files(directory):
    """List pdf/docx/pptx filenames in directory for hash building."""
    filename_str = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            m = re.search(r"\.([^.]+)$", file)
            if m and m.group(1).lower() in ("pdf", "docx", "pptx"):
                filename_str.append(file + "_$$_")
    return sorted(filename_str)


def pptx_direct_loaded_chunks():
    """Load pptx files as chunks (directory loader does not handle pptx)."""
    directory_path = doc_directory_option
    files_info = get_file_types_csv(directory_path)
    pptx_files_info = files_info[files_info["Document Type"] == ".pptx"].reset_index(drop=True)
    if pptx_files_info.empty:
        return None
    pptx_loader = UnstructuredPowerPointLoader(pptx_files_info["Path"][0]).load_and_split()
    for path in pptx_files_info["Path"][1:]:
        load_pptx = UnstructuredPowerPointLoader(path).load_and_split()
        pptx_loader = pptx_loader + load_pptx
    return pptx_loader


def query_pdfs_intel(cust_query, outside_context, email, correlation_id, verbosity, recent_chat_history=None):

    print("Inside query_pdfs_intel function ")

    global last_two_questions

    # -----------------------------
    # Build hash from current files
    # -----------------------------
    doc_directory = doc_directory_option
    index_name_list = list_files(doc_directory)
    index_name = "".join(index_name_list)
    hash_index = generate_unique_string_hash(index_name)

    # -----------------------------
    # Resolve vector DB to use
    # -----------------------------
    vector_location = CONFIG['embeddings']['faiss_index_storeloc']
    vectors_found = list_directories_sorted_by_modified_time(vector_location)

    if not vectors_found:
        return [
            cust_query,
            "We haven't found any Document to query on. Please upload some documents first.",
            []
        ]

    for vector_db in vectors_found:
        if hash_index in vector_db:
            print("Found vector for current files:", hash_index)
            break
    else:
        # fallback to latest available vector
        latest_vector = vectors_found[0]
        hash_index = os.path.basename(latest_vector)
        print("Vector not found for current files, using latest:", hash_index)

    print("Loading vector db with hash:", hash_index)

    # -----------------------------
    # Load embeddings
    # -----------------------------
    embeddings = return_embeddings_model()

    # -----------------------------
    # SAFE FAISS LOAD / REBUILD
    # -----------------------------
    vector_path = os.path.join(vector_location, hash_index)
    faiss_file = os.path.join(vector_path, "index.faiss")
    pkl_file = os.path.join(vector_path, "index.pkl")

    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        db = FAISS.load_local(vector_path, embeddings)
        print("FAISS index loaded successfully")
    else:
        print("FAISS index missing. Rebuilding index...")

        allowed_dtypes = CONFIG["files_location"]["unstructured_dtype"]
        full_docs = []

        for dtype in allowed_dtypes:
            docs = create_document_loader(documents_loc=doc_directory, type=dtype)
            if docs:
                full_docs.extend(docs)

        pptx_docs = pptx_direct_loaded_chunks()
        if pptx_docs:
            full_docs.extend(pptx_docs)

        if not full_docs:
            return [
                cust_query,
                "No valid documents found to build knowledge base.",
                []
            ]

        db = FAISS.from_documents(full_docs, embeddings)
        os.makedirs(vector_path, exist_ok=True)
        db.save_local(vector_path)

        print("FAISS index rebuilt and saved")

    print("Total chunks:", len(db.index_to_docstore_id))

    # -----------------------------
    # Similarity search
    # -----------------------------
    start_time = time.time()
    docs_with_score = db.similarity_search_with_score(cust_query, k=7)

    distance_scores = [score for _, score in docs_with_score]
    percentile_threshold = [CONFIG['citation']['dynamic_score_threshold']]
    percentiles_result = calculate_percentiles(distance_scores, percentile_threshold)
    threshold_score = percentiles_result[0]

    docs = [doc for doc in docs_with_score if doc[1] < threshold_score]

    print("Context chunks selected:", len(docs))

    # -----------------------------
    # Build context
    # -----------------------------
    docs_context = ""
    for doc in docs:
        docs_context += "\n\n" + doc[0].page_content

    prompt = (
        Final_Prompt
        .replace("{context}", docs_context)
        .replace("{question}", cust_query)
        .replace("{last_question}", last_two_questions[1])
    )
    if recent_chat_history:
        prompt += "\n\nRecent conversation with user (for context only):\n" + recent_chat_history

    # -----------------------------
    # Verbosity handling
    # -----------------------------
    low_verbose = "**Summarize response in strictly maximum 25 words.**"
    high_verbose = "**Provide as much detailed information as possible.**"

    if verbosity == "low":
        prompt = prompt.replace("{verbosity_control}", low_verbose)
    elif verbosity == "high":
        prompt = prompt.replace("{verbosity_control}", high_verbose)
    else:
        prompt = prompt.replace("{verbosity_control}", "")

    usr_df_loc = CONFIG['files_location']['usr_data_dir']
    prompt = inject_user_info(prompt, email, usr_df_loc)

    # -----------------------------
    # LLM Call
    # -----------------------------
    textdata_response = None

    if 'azure' in CONFIG['model']['openai_api_type']:
        for i in range(len(CONFIG['azure_llm_instances']['model_name'])):
            try:
                textdata_response = query_openai_mult(prompt, i)
                logger.info(f"{email}, {correlation_id}, Query executed successfully")
                break
            except Exception as e:
                logger.error(f"Azure instance {i} failed: {e}")

    # -----------------------------
    # Citations
    # -----------------------------
    citations = []
    for doc in docs:
        meta = doc[0].metadata
        score = doc[1]
        filename = os.path.basename(meta.get("source", "document"))
        page = meta.get("page", 0)

        citations.append({
            "filename": filename,
            "label": f"{filename}, Page : {page + 1}",
            "description": doc[0].page_content,
            "link": None,
            "textToHighlight": None
        })

    # -----------------------------
    # Memory
    # -----------------------------
    last_two_questions.pop(0)
    last_two_questions.append(cust_query)

    return [cust_query, textdata_response, citations]
