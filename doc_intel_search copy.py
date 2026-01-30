import os
import json
from pathlib import Path
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory

import hashlib

from langchain.document_loaders import UnstructuredPowerPointLoader
import pandas as pd
from config.config_validator import validate_ini_file

from utils import query_openai, query_openai_gpt35_backup, return_embeddings_model, create_document_loader, create_document_splitter, create_vector_store, inject_user_info, calculate_percentiles, get_file_types_csv
from config.prompts import Final_Prompt, Context_Question_Prompt, leaves_mapping, new_template
#from deploy import email

from utils import list_directories_sorted_by_modified_time, query_openai
from logger_setup import logger
import time
import concurrent.futures
import itertools
CONFIG_LOC = os.environ.get('CONFIG_LOC', str(Path('config/config.ini')))
print("Using config file: ", CONFIG_LOC)
# validate and read config file:
CONFIG = validate_ini_file(CONFIG_LOC)


base_path = Path(__file__).parent
print(base_path)


# LLM_option = "OpenAI"
# standalone Open AI
# os.environ["OPENAI_API_KEY"] = "sk-w2JBDWObHx6xxxxxxxxxxxxxxxxxxxx"
# OpenAI_model = "gpt-3.5-turbo" 
# OpenAI_model = "gpt-4"

LLM_option = "Azure OpenAI"

# ------- uploads directory
# doc_directory_option = "/var/www/html/salesbuddy/wp-content/uploads" # UAT
# doc_directory_option = "/var/www/html/wp-content/uploads"
# doc_directory_option = "../sales_buddy_uploads"
doc_directory_option = "../chatbot-dev/uploads"

# prod


def delete_files(directory):
  for file in os.listdir(directory):
    path = os.path.join(directory, file)
    if os.path.isfile(path):
      os.remove(path)

def create_document_splitter(document_loader) -> list[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " "],
        # length_function = lambda x: x*0.75 # token count is roughly 0.75 times the character count
    )
    print("DOcument splitter inprogress------------------")
    documents = text_splitter.split_documents(document_loader)
    return documents


import time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.1f} seconds to execute.")
        return result
    return wrapper


def generate_unique_string_hash(input_string):
    hash_object = hashlib.sha1(input_string.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex
    # return "DB"

import os
import re


def list_files(directory):
    '''this will be used to check if any files have changed or not in the complete directory'''
    filename_str = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_type = re.search(r"\.([^\.]+)$", file).group(1)
            if file_type == "pdf" or file_type == "docx" or file_type=="pptx":
                filename_str.append(file + "_$$_")
            
    return sorted(filename_str)


# pptx handle
@measure_time
def pptx_direct_loaded_chunks():

    directory_path = doc_directory_option
    get_file_types_csv(directory_path)

    # files_info = pd.read_csv("pptx_loader_helper.csv")
    files_info = get_file_types_csv(directory_path)      ## Changing this to skip creation and use of df pptx_loader_helper
    print(files_info.head())
    pptx_files_info = files_info[files_info["Document Type"]==".pptx"]
    pptx_files_info = pptx_files_info.reset_index(drop=True)

    if pptx_files_info.empty:
        return None

    print("pptx_files_info ", pptx_files_info.head())

    print("path 0th overall ", files_info["Path"][0])
    print("path 0th", pptx_files_info["Path"][0])

    pptx_loader = UnstructuredPowerPointLoader( pptx_files_info["Path"][0] )
    pptx_loader = pptx_loader.load_and_split()

    for pptx in pptx_files_info["Path"][1:]:
        load_pptx = UnstructuredPowerPointLoader(pptx)
        load_pptx = load_pptx.load_and_split()
        pptx_loader = pptx_loader + load_pptx

    return pptx_loader


@measure_time
def create_embedding():

    doc_directory = doc_directory_option
    index_name_list = list_files(doc_directory)
    index_name = ""
    for filename in index_name_list:
    #    index_name += filename[:-4]
        index_name += filename

    hash_index = generate_unique_string_hash(index_name)
    print(" hash_index name ", hash_index)

    if os.path.exists("Vector_DB/"+hash_index):
            #load
            print("found existing embeddings so skipping")
            return None
    
    documents_loc = doc_directory_option

    ###***
    allowed_dtypes = CONFIG["files_location"]["unstructured_dtype"]
    full_split_doc_list = []
    for dtype in allowed_dtypes:
        docs = create_document_loader(documents_loc=documents_loc, type=dtype)
        if docs is not None:
            full_split_doc_list.extend(docs)
#            docs_split = create_document_splitter(docs)   ##to be used when creating chunks at token level
        print("Split Document pages count: ", len(full_split_doc_list))

    #extra processing to handle pptx as directroy loader is not working with pptx
    pptx_direct_pages = pptx_direct_loaded_chunks()
    print('\n \n \n pptx pages: \n', pptx_direct_pages)
    if pptx_direct_pages:
        full_split_doc_list = full_split_doc_list + pptx_direct_pages        
    chunks = full_split_doc_list

    # chunks=infuse_context_into_document(chunks)   # being infused in create_vector_store function from utils

    print("chunks type ", type(chunks) )
    print("len ", len(chunks))

    db = create_vector_store(split_documents=chunks) 

    print("Embeddings created Successfully with name ", index_name, " Hashed Name ", hash_index)
    vector_location = CONFIG['embeddings']['faiss_index_storeloc']
    db.save_local(vector_location + hash_index)
    print("Embeddings saved Successfully with name ", hash_index,  " Hashed Name ", hash_index)

    # return index_name
    return [index_name, hash_index]

last_two_questions = ["", ""]

# Define a load balancing strategy (round-robin)
def round_robin(iterable):
    # Cycle through the iterable indefinitely
    pool = itertools.cycle(iterable)
    while True:
        yield next(pool)

def query_pdfs_intel(cust_query, outside_context, email, correlation_id):
    global last_two_questions

    doc_directory = doc_directory_option
    index_name_list = list_files(doc_directory)
    index_name = ""
    for filename in index_name_list:
        index_name += filename

    hash_index = generate_unique_string_hash(index_name)

    # To handle a case when there is an update in the files and vector DB is not created for that new set of files
    # We will use the latest available vector_DB with us from the options

    # vector_db_path = "../sales_buddy_vector_db/"
    vector_location = CONFIG['embeddings']['faiss_index_storeloc']
    vectors_found = list_directories_sorted_by_modified_time(vector_location)

    if len(vectors_found)==0:
        return [cust_query, "We haven't found any Document to query on. Please upload some documents first before asking a question. Thanks!"]


    for vector_db in vectors_found:
        if hash_index in vector_db:
            print("Found the vector from already created--------", hash_index)
            break
    else:
        hash_index = vectors_found[0]
        pattern = r'/([^/]+)\Z'
        hash_index = re.search(pattern, hash_index).group(1)
        print("Not found, --- using the vector from already created--------", hash_index)

    print("Loading the vector db with hashed name--> ", hash_index)

    global docs
    global textdata_response

    # Using return embeddings model from utils to get model from date mentioned in config.ini
    embeddings = return_embeddings_model() 
    vector_location = CONFIG['embeddings']['faiss_index_storeloc']
    db = FAISS.load_local( vector_location+hash_index, embeddings)
    print("Loaded")
    print("index_to_docstore_id total documents/chunks ---> ", len(db.__dict__["index_to_docstore_id"]))        

    # Getting relevant docs for dynamic citations
    #docs_with_score = db.similarity_search_with_score(cust_query, k=7)
## Chekcing if the cust_query is a relevant connecting question to the last question
    Context_Question_Prompt_Final = Context_Question_Prompt.replace("{leaves_mapping}", str(leaves_mapping)).replace("{question_1}", last_two_questions[1]).replace("{question_2}", cust_query)
    start_time = time.time()
    if query_openai(Context_Question_Prompt_Final)== "Yes":
        ques = 'Q.1' + last_two_questions[1] + '? ' + 'Q.2' + cust_query + '/n' + "**Strictly follow the Note: The 'Q.1' is just for your reference to take context for 'Q.2' so that you can answer it even if user ask follow up or related question to 'Q.1' in short. If 'Q.1' and 'Q.2' are not relevant to each other i.e. doesn't share any context, you should ignore 'Q.1' and answer 'Q.2' as usual. Also, provide answer only for 'Q.2' in all scenarios without mentioning 'Q.1', 'Q.2' in the answer."
        docs_with_score = db.similarity_search_with_score(ques, k=7)
        distance_scores = []
        for doc in docs_with_score:
            distance_scores.append(doc[1])
        distance_scores

        percentile_threshold = [CONFIG['citation']['dynamic_score_threshold']]
        percentiles_result = calculate_percentiles(distance_scores, percentile_threshold)

        threshold_score = percentiles_result[0]
        print(" threshold_score is -->", threshold_score)

        # filtered
        docs = [doc for doc in docs_with_score if doc[1]< threshold_score]

        # Fetching Prompts
        docs_context = ""
        for doc in docs:
            docs_context = docs_context + f" \n\
            \n\
                {doc[0].page_content}"
#        prompt = Final_Prompt.replace("{context}", docs_context).replace("{question}", ques).replace("{leaves_mapping}", str(leaves_mapping)).replace("{last_question}", last_two_questions[1])
        prompt = new_template.replace("custom_db",docs_context).replace('leaves_mapping', str(leaves_mapping).replace('last_question', last_two_questions[1]).replace('{', '[').replace('}', ']'))
    else:        
        docs_with_score = db.similarity_search_with_score(cust_query, k=7)
        distance_scores = []
        for doc in docs_with_score:
            distance_scores.append(doc[1])
        distance_scores

        percentile_threshold = [CONFIG['citation']['dynamic_score_threshold']]
        percentiles_result = calculate_percentiles(distance_scores, percentile_threshold)

        threshold_score = percentiles_result[0]
        print(" threshold_score is -->", threshold_score)

        # filtered
        docs = [doc for doc in docs_with_score if doc[1]< threshold_score]

        # Fetching Prompts
        docs_context = ""
        for doc in docs:
            docs_context = docs_context + f" \n\
            \n\
                {doc[0].page_content}"
#        prompt = Final_Prompt.replace("{context}", docs_context).replace("{question}", cust_query).replace("{leaves_mapping}", str(leaves_mapping))
        prompt = new_template.replace("custom_db",docs_context).replace('leaves_mapping', str(leaves_mapping).replace('last_question', last_two_questions[1]).replace('{', '[').replace('}', ']'))
    usr_df_loc = CONFIG['files_location']['usr_data_dir']    
    print('The curated prompt is *****#####:', prompt)
    prompt = inject_user_info(prompt=prompt, email=email, user_df_loc=usr_df_loc)
    prompt_template = PromptTemplate(input_variables=['entities', 'history', 'input'], output_parser=None, partial_variables={}, template=prompt)

    textdata_response = None
    if 'azure' in CONFIG['model']['openai_api_type']:       
        # For using multiple LLM instances
        # Fetching details from config file
        model_names = CONFIG['azure_llm_instances']['model_name']
        deployment_names = CONFIG['azure_llm_instances']['deployment_name']
        ns = CONFIG['azure_llm_instances']['n']
        temperatures = CONFIG['azure_llm_instances']['temperature']
        openai_api_types = CONFIG['azure_llm_instances']['openai_api_type']
        openai_api_bases = CONFIG['azure_llm_instances']['openai_api_base']
        openai_api_versions = CONFIG['azure_llm_instances']['openai_api_version']
        openai_api_keys = CONFIG['azure_llm_instances']['openai_api_key']
        instances = []
        for i in range(len(model_names)):
            llm = AzureChatOpenAI(model_name=model_names[i],
                    deployment_name=deployment_names[i],
                    n=ns[i],
                    temperature=temperatures[i],
                    openai_api_type = openai_api_types[i],
                    openai_api_base = openai_api_bases[i],
                    openai_api_version = openai_api_versions[i],
                    openai_api_key = openai_api_keys[i])
            
            instances.append(llm)

            # Create a pool of workers for concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as executor:
            # Use the round-robin strategy to select an instance for each request
                load_balancer = round_robin(instances)
                try:
                    # Submit requests concurrently
                    for i in range(len(model_names)):
                        llm = next(load_balancer)
                        conversation = ConversationChain(llm=llm, verbose=True, prompt = prompt_template,
                                                        memory=ConversationEntityMemory(llm=llm, k=3))

                    futures = [executor.submit(conversation.predict, input=cust_query) for _ in range(len(instances))]
                    # Wait for the first successful response
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            textdata_response = future.result()
                            logger.info(f'{email}, {correlation_id}, Query: {cust_query}, Query_Type: {"Normal"}, Execution_Time: {time.time() - start_time}, Status: {"Query Executed Successfully"}')
                            print(" Azure Open AI Response --- ", textdata_response)
                            break
                        except Exception as e:
                            # Log the error and continue to the next instance
                            logger.error(f'{email}, {correlation_id}, Query: {cust_query}, Query_Type: {"Normal"}, Execution_Time: {time.time() - start_time}, Status: {f"Instance failed. Trying the next instance"}')
                except Exception as e:
                    print(e)
#                    logger.error(f'Error during concurrent execution: {str(e)}')
        
        if textdata_response is None:
            print("All LLM instances failed to provide a response.")
    else:
        BASE_LLM = ChatOpenAI(temperature=0, model = OpenAI_model)
        # using OpenAI Base LLM
        chain = load_qa_chain(BASE_LLM, chain_type="stuff") 
        if outside_context =="yes":
            out_context_prompt = "You will be given a question along with the context to answer below. \n\
        You can use the information from your knowledge base or internet to answer the question in best possible manner. \n\
            Give the grammatically correct and structured response. \n\
            "
        else:
            # check input_documents dtype (list/str)
            textdata_response = chain.run(input_documents=docs, question=prompt)
            print(" Open AI response ---- ", textdata_response)
            logger.info(f'{email}, {cust_query}, Query_Type: {"Normal"}, Execution_Time: {time.time() - start_time}, Status: {"Query Executed Successfully"}')

    # Preparing Dynamic Citations
    citation_texts = []
    if outside_context=="yes":
        pass
    else:
        for doc in docs:            
            citation_texts.append([doc[0].page_content , doc[0].metadata, doc[1] ] )
            #content, metadata, score
        citations = []

        for cit in citation_texts:
            # better approach by checking if page is there
            if cit[1].get("page") is not None:
                # to get just the filename instead of complete path
                filename = re.search(r'[\\/]+([^\\/]+)$', cit[1]["source"])
                # re.search(r'[\\/]+([^\\/]+)$', file)
                print("filename label out ", filename)
                if filename:
                    filename = filename.group(1)
                    print("filename label ", filename)
                # added filename field for download_file API
                citations.append({'filename':str(filename), 'label':str(filename) + ", Page : " + str(cit[1]["page"] +1) , "description":cit[0], 'link':None, 'textToHighlight':None })
            else:
                filename = re.search(r'[\\/]+([^\\/]+)$', cit[1]["source"])
                print("filename label out ", filename)
                if filename:
                    filename = filename.group(1)
                    print("filename label ", filename)
                citations.append({'filename':str(filename), 'label':str(filename) + ", Page : 0 "  , "description":cit[0], 'link':None, 'textToHighlight':None })
    # Remember the last two questions
    last_two_questions.pop(0)
    last_two_questions.append(cust_query)
    return [cust_query, textdata_response, citations]


    
