import os

from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

import hashlib

from langchain.document_loaders import UnstructuredPowerPointLoader
import pandas as pd
from pptx_handler import get_file_types_csv
import pandas as pd



if __name__ == '__main__':
    # ---
    from doc_intel_search import create_embedding

    # call create_embedding
    file_index = create_embedding()

    if file_index == None:
        print("Latest Embeddings are already available, not creating new")

    else:
        print("Updated Embedding as per the latest file changes")