from pathlib import Path
import uuid
import re
import pandas as pd
import numpy as np
import json
import csv
import openai
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from transformers import GPT2TokenizerFast
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from fuzzywuzzy import fuzz
from config.config_validator import validate_ini_file
from langchain.chat_models import AzureChatOpenAI
import os
import collections
from logger.logger_setup import logger
CONFIG_LOC = os.environ.get('CONFIG_LOC', str(Path('config/config.ini')))
print("Using config file: ", CONFIG_LOC)
# validate and read config file:
CONFIG = validate_ini_file(CONFIG_LOC)



os.environ["OPENAI_API_KEY"] = "sk-w2JBDWObHx6BEQxxxxxxxxxxxxxxxxxxxxxx"
#openai.api_key = "sk-w2JBDWObHx6BEQxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


def query_openai_mult(prompt, n_model):

    try:
    
        print("\t Inside query_openai_mult function ")

        model_names = CONFIG['azure_llm_instances']['model_name']
        deployment_names = CONFIG['azure_llm_instances']['deployment_name']
        ns = CONFIG['azure_llm_instances']['n']
        temperatures = CONFIG['azure_llm_instances']['temperature']
        openai_api_types = CONFIG['azure_llm_instances']['openai_api_type']
        openai_api_bases = CONFIG['azure_llm_instances']['openai_api_base']
        openai_api_versions = CONFIG['azure_llm_instances']['openai_api_version']
        openai_api_keys = CONFIG['azure_llm_instances']['openai_api_key']
        response = openai.ChatCompletion.create(
        engine=deployment_names[n_model], # replace this value with the deployment name you chose when you deployed the asso>
                        n=int(ns[n_model]),
                        temperature=int(temperatures[n_model]),
                        api_type = openai_api_types[n_model],
                        api_base = openai_api_bases[n_model],
                        api_version = openai_api_versions[n_model],
                        api_key = openai_api_keys[n_model],
                        messages=[{"role": "user", "content": prompt}])
        

        print("===========")
        print(response)
        print(response.choices[0]['message']['content'])
        print("===========")
        return response.choices[0]['message']['content']         

    except Exception as e:
        print("Error::::::", e)
        return None
                         
                         
def _get_azure_llm_credentials():
    """Get API key and endpoint: from env first, else first instance from config.ini."""
    api_key = (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY") or "").strip()
    api_base = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    if api_key and api_base:
        return api_key, api_base
    # Use first Azure LLM instance from config (same as RAG path)
    inst = CONFIG.get("azure_llm_instances") or {}
    keys = inst.get("openai_api_key")
    bases = inst.get("openai_api_base")
    if keys and bases:
        key = keys[0] if isinstance(keys, list) else keys
        base = (bases[0] if isinstance(bases, list) else bases).rstrip("/")
        if key and base:
            return key, base
    return "", ""


def query_openai(prompt):
    try:
        api_key, api_base = _get_azure_llm_credentials()
        if not api_key or not api_base:
            raise ValueError("Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT, or configure config.ini [azure_llm_instances]")
        inst = CONFIG.get("azure_llm_instances") or {}
        deployment = (inst.get("deployment_name") or ["gpt-4o-mini"])
        deployment = deployment[0] if isinstance(deployment, list) else deployment
        api_version = (inst.get("openai_api_version") or ["2023-03-15-preview"])
        api_version = api_version[0] if isinstance(api_version, list) else api_version
        response = openai.ChatCompletion.create(
            engine=deployment,
            n=1,
            temperature=0,
            api_type="azure",
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0]['message']['content']

    except Exception as e:
        print("*******************")
        print(f"Got into an error when using the GPT-4 ---------- {e}")
        print("*******************")
        return query_openai_gpt35_backup(prompt)

def query_openai_gpt35_backup(prompt):
    """Fallback LLM call using same credentials as query_openai (config or env)."""
    api_key, api_base = _get_azure_llm_credentials()
    if not api_key or not api_base:
        raise ValueError("Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT, or configure config.ini [azure_llm_instances]")
    inst = CONFIG.get("azure_llm_instances") or {}
    deployment = (inst.get("deployment_name") or ["gpt-4o-mini"])
    deployment = deployment[0] if isinstance(deployment, list) else deployment
    api_version = (inst.get("openai_api_version") or ["2023-03-15-preview"])
    api_version = api_version[0] if isinstance(api_version, list) else api_version
    response = openai.ChatCompletion.create(
        engine=deployment,
        n=1,
        temperature=0,
        api_type="azure",
        api_base=api_base,
        api_version=api_version,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0]['message']['content']

def _check_existance_filetype(documents_loc, file_type='pdf'):
    pdf_count = 0
    for root, _, files in os.walk(documents_loc):
        for filename in files:
            if filename.endswith("." + file_type):
                pdf_count += 1
                if pdf_count > 0:
                    return True
    print(f"GOt {pdf_count} files of type {file_type}")
    return False

def create_document_loader(documents_loc:str, type='pdf'):
    """Split a document (txt, pdf, docx) into chunks using langchain"""
    from langchain.document_loaders import DirectoryLoader
    #check if more than one .pdf file is there in documents_loc:
    loader = None
    if "pdf" in type and _check_existance_filetype(documents_loc, file_type='pdf'):
        print('loading pdfs')
        try:
            from langchain.document_loaders import PyPDFLoader
        except ImportError:
            raise Exception('PyPDF2 not installed: install using "pip install pypdf"')
        loader = DirectoryLoader(documents_loc, glob="**/*.pdf", 
                                 loader_cls=PyPDFLoader,
                                 show_progress=True, 
                                 use_multithreading=True,
                                 recursive=True)
    elif "docx" in type and _check_existance_filetype(documents_loc, file_type='docx'):
        print('loading docx')
        try:
            from langchain.document_loaders import Docx2txtLoader
        except ImportError:
            raise Exception('python-docx not installed')
        loader = DirectoryLoader(documents_loc, glob="**/*.docx", 
                                 loader_cls=Docx2txtLoader,
                                 show_progress=True, 
                                 use_multithreading=True,
                                 recursive=True)
    elif "html" in type and _check_existance_filetype(documents_loc, file_type='html'):
        try:
            from langchain.document_loaders import BSHTMLLoader
        except ImportError:
            raise Exception('beautifulsoup not installed: "pip install beautifulsoup4"')
        loader = DirectoryLoader(documents_loc, glob="**/*.html", 
                                 loader_cls=BSHTMLLoader,
                                 show_progress=True, 
                                 use_multithreading=True)
    elif "azure" in type:
        try:
            from langchain.document_loaders import AzureBlobStorageContainerLoader
        except ImportError:
            raise Exception('azure-storage-blob not installed: "pip install azure-storage-blob"')
        assert CONFIG['embeddings']['azure_blob_doc_loader_connection_string'] != "None", "Please set the azure_blob_doc_loader_connection_string in the config file"
        assert CONFIG['embeddings']['azure_blob_doc_loader_container'] != "None", "Please set the azure_blob_doc_loader_container in the config file"
        loader = AzureBlobStorageContainerLoader(conn_str=CONFIG['embeddings']['azure_blob_doc_loader_connection_string'], 
                                                 container=CONFIG['embeddings']['azure_blob_doc_loader_container'])
    elif "text" in type:
        from langchain.document_loaders import TextLoader
        loader = DirectoryLoader(documents_loc, glob="**/*.txt", 
                                 loader_cls=TextLoader,
                                 show_progress=True, 
                                 use_multithreading=True)
    else:
        print(f'could not find any documents of type {type} in the given directory')
    if loader is not None:
        docs = loader.load()
        return docs

def create_document_splitter(document_loader) -> list[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    chunk_size = CONFIG['embeddings']['chunk_size']
    chunk_overlap = CONFIG['embeddings']['chunk_overlap']
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(document_loader)
    return documents

def return_embeddings_model(embeddings_model_type=None):
    """Return the embeddings model"""
    embeddings_model_type = CONFIG['embeddings']['embeddings_model_type'] if embeddings_model_type is None else embeddings_model_type
    if "azure" in embeddings_model_type:

        from langchain.embeddings import OpenAIEmbeddings
        deployments = CONFIG['azure_embeddings']['azure_embeddings_deployment']
        openai_api_bases = CONFIG['azure_embeddings']['azure_embeddings_api_base']
        open_api_types = CONFIG['azure_embeddings']['azure_embeddings_api_type']
        open_api_versions = CONFIG['azure_embeddings']['azure_embeddings_api_version']
        open_api_keys = CONFIG['azure_embeddings']['azure_embeddings_api_key']
        for i in range(len(deployments)):
            deployment = deployments[i]
            openai_api_base = openai_api_bases[i]
            openai_api_type = open_api_types[i]
            openai_api_version = open_api_versions[i]
            openai_api_key = open_api_keys[i]
            embeddings = OpenAIEmbeddings(deployment= deployment,
                                    openai_api_base=openai_api_base,
                                    openai_api_type=openai_api_type,
                                    openai_api_version=openai_api_version,
                                    openai_api_key=openai_api_key)
            try:
                test = 'hi'
                test_embeddings = embeddings.embed_query(test) 
                break
            except Exception as e:
                print(e) 
    elif embeddings_model_type=='openai':
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                                      openai_api_key=CONFIG['embeddings']['embeddings_model_api_key'])
    else:
        raise Exception('embeddings_model_type currently not supported, change to "azureopenai" or "openai" in config file')
    return embeddings


def infuse_context_into_document(split_documents):
    """Give the split documents, add the name of document in the page_content of each document to main context while vectorizing"""
    print('infusing context into document before vectorizing....')
    for doc in split_documents:
        document_name = doc.metadata["source"]
        document_name = re.search(r"[\\/]([^\\/]+)$", document_name).group(1)

        document_content = doc.page_content
        doc.page_content = f"Document Name : {document_name}\n\nDocument Content :\n{document_content}"
    print("split_documents lenght  --> ", len(split_documents))
    return split_documents

def create_vector_store(split_documents=None, type=None):
    """Create a vector store for the given documents"""
    embeddings = return_embeddings_model()
    type = type if type is not None else CONFIG["embeddings"]["vector_store_type"]
    # print(f"Creating vector store of type {type} with index_name {index_name}")
    print(f"Creating vector store of type {type}")
    if type=='faiss':
        # check existance of 'faiss_cpu'in python environment:
        try:
            from langchain.vectorstores import FAISS
        except ImportError:
            raise Exception('faiss not installed: install using "pip install faiss-cpu"')

        assert split_documents is not None, "Please provide split_documents"
        split_documents = infuse_context_into_document(split_documents)

        # when we don't have any documents
        if len(split_documents)==0:
            return None
        
        docsearch = FAISS.from_documents([split_documents[0]], embeddings)
        for doc in split_documents[1:]:
            docsearch.add_documents([doc])

    return docsearch


def inject_user_info(prompt, email, user_df_loc):
    """Inject user information into the prompt"""
    print("Injecting user information into the prompt...")
    return prompt
    # user_df = pd.read_csv(user_df_loc + '/user_data.csv')
    # user_info = user_df[user_df['email'] == email]
    # # if the user is not found in the db, then return an empty string
    # if user_info.empty:
    #     print(f"User with email {email} not found in the database. Please try again.")
    #     #user_info = ""
    #     return prompt
    # else:
    #     user_info = user_info.drop(['email'], axis=1)
    #     out_str = ""
    #     for col in user_info.columns:
    #         out_str += col + " is " + str(user_info[col].values[0]) + "\n"
    #     prompt_inj = "\nAlso, here is some useful information about the person asking the question: " + out_str + "\n"
    #     return prompt + prompt_inj


def calculate_percentiles(scores_list, percentiles):
    # Sorting the data list in ascending order
    scores_list.sort()
    
    # Calculating the percentiles using numpy's percentile function
    result = np.percentile(scores_list, percentiles)
    return result


def get_file_types(directory):
    file_types_2 = collections.defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)  # Get the full file path
            _, ext = os.path.splitext(file)
            file_info = {"ext": ext, "path": file_path}
            file_types_2[file].append(file_info)
    return file_types_2

def get_file_types_csv(directory_path):
    file_types_2 = get_file_types(directory_path)
    relevant_docs = pd.DataFrame(columns=["Document Type", "Filename", "Path"])
    for file, extensions in file_types_2.items():
        if (extensions[0]["ext"]==".pdf" or extensions[0]["ext"]==".docx" or extensions[0]["ext"]==".pptx"):
            path = os.path.normpath(extensions[0]["path"])  # Normalize the path with consistent slashes
            new_row = pd.DataFrame({"Document Type": [extensions[0]["ext"] ], "Filename": [file], "Path": [ path ]})
            relevant_docs = pd.concat([relevant_docs, new_row], ignore_index=True)
    # this file contains details of file available >>> check for alternate way to get rid of this
    # relevant_docs.to_csv("download_helper.csv", index=False)    
    return relevant_docs



def list_directories_sorted_by_modified_time(vector_location):
    directories = [os.path.join(vector_location, item) for item in os.listdir(vector_location) if os.path.isdir(os.path.join(vector_location, item))]
    sorted_directories = sorted(directories, key=lambda x: - os.path.getmtime(x)) #latest first
    return sorted_directories

# Generating LLM model for Langchain Conversation Chain
llm_for_conversation = AzureChatOpenAI(
        model_name="gpt-4-32k",
        deployment_name="gpt-4-32k",
        n=1,
        temperature=0,
        openai_api_type = "azure",
        openai_api_base = "https://sagar-m5i7au59-australiaeast.services.ai.azure.com",
        openai_api_version = "2023-03-15-preview",
        openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", ""))