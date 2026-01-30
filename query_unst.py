import pandasql as ps
import pandas as pd
import argparse
import re
import os
from pprint import pprint
import json
from pathlib import Path
from utils import query_openai, query_openai_mult   #, query_pdf, will use query_pdf from doc_intel search
import time
from doc_intel_search import query_pdfs_intel
from logger.logger_setup import logger
from config.config_validator import validate_ini_file
CONFIG_LOC = os.environ.get('CONFIG_LOC', str(Path('config/config.ini')))
print("Using config file: ", CONFIG_LOC)
# validate and read config file:
CONFIG = validate_ini_file(CONFIG_LOC)

greeting_chk_context = f"You have to check the following question and identify if this is normal \
    greeting or generic question for a chat bot and answer 'Yes' if the question is greeting else 'No'. \
    Also, here are few points to remeber while concluding to the answer: \
    1. Along with greetings you must consider questions from list ['what can you do?' , ' who are you?', 'What Can you do?', 'which data do you use?' , 'What is inside the data that you use?', 'What data do you have?', 'what you do?'] as generic and provide answer as 'Yes'.\
    2. If the question is just words with lack of meaning or any unrecognizable linguistic structure keep answer as 'Yes' so it can just be handled as greeting \
    Attention: Make sure that you should only give answer in one word either 'Yes' or 'No'\
    Question: "

CON = "This is a Deep Learning and Natural Language Understanding based query tool that can give you insights on nexus India employees policies. If the user \
       asks 'What can you do?' or 'who are you?', then respond exactly with 'I am chatbot-dev-nexus, a Deep Learning and Natural Language Understanding based tool developed \
       by nexus India Pvt. Ltd. I was built using the latest Azure Open AI GPT architecture, which is a state-of-the-art deep learning \
       model that uses a transformer network to generate human-like text. I am trained on the Employee Policies \
       of nexus., and my primary function is to understand, process natural language queries on \
       Sales case studies data and generate appropriate responses'. \n\
       If the user asks 'What data do you have?' or \
       'which data do you use?' or 'What is inside the data that you use?' \
       or 'What is the data summary?' or 'Summarize the data', then respond with 'The dataset for The nexus employee policies. This comprehensive dataset allows me to provide accurate and insightful answers to queries related to data you have uploaded in my knowledge Base' "
       

handle_greeting_general = """
If the question is more of general conversation, like the user is greeting you/asking a general question or greeting you with hi or hello or heyy then respond back with the general greeting with the {name} wherever possible. also, here is some information that you may find useful for certain questions: '{CON}'.\n\
    5. If the question seems gratitude, like the user is thanking you or expressing gratitude with thank you or thanks or thank you so much then respond back with My Pleasure. Glad to help you. If you have any further question then please feel free to ask. If the user wants to end the conversation like saying bye or by or bye-bye or see you later or catch you later or I am done then respond with Bye!!  \n\
    Do noi include any words at the start of response mentioning it as an answer like 'Answer:', 'A', 'A.' etc.
    \n\
Now the main Question to answer is:\n\
"""

    
def query_unstructured(cust_query, outside_context, email, correlation_id, verbosity, recent_chat_history=None):
    start_time = time.time()
    final_greeting_chk_context = (greeting_chk_context + cust_query)
    model_names = CONFIG['azure_llm_instances']['model_name']
    deployment_names = CONFIG['azure_llm_instances']['deployment_name']
    ns = CONFIG['azure_llm_instances']['n']
    temperatures = CONFIG['azure_llm_instances']['temperature']
    openai_api_types = CONFIG['azure_llm_instances']['openai_api_type']
    openai_api_bases = CONFIG['azure_llm_instances']['openai_api_base']
    openai_api_versions = CONFIG['azure_llm_instances']['openai_api_version']
    openai_api_keys = CONFIG['azure_llm_instances']['openai_api_key']
    email_parts = email.split('@')
    name_parts = email_parts[0].split('.')
    name = name_parts[0] + ' ' + name_parts[1] if len(name_parts) > 1 else email_parts[0]
#    name = str(email.split('@')[0].split('.')[0] + ' ' + email.split('@')[0].split('.')[1])
    greeting_prompt_final = handle_greeting_general.replace("{name}", name).replace("{CON}", CON) +'\n\nQ. '+ cust_query
    # print('final is ........///', greeting_prompt_final)
    if query_openai(final_greeting_chk_context) == 'Yes':
        print('>>> Greeting/Generic Question, fetching output without citation')
        for i in range(len(model_names)):
            try:
                json_str = query_openai_mult(greeting_prompt_final, i)
                logger.info(f'{email}, {correlation_id}, Query: {cust_query}, Query_Type: {"General Greeting"}, Execution_Time: {time.time() - start_time}, Status: {"Query Executed Successfully"}')
                return json.dumps({'queryType':'generalGreeting', 'citation':None, 'label':None, 'textdata': json_str, 'respType':1})
            except Exception as e:
                #Logging the error and continue to the next instance
                logger.error(f'{email}, {correlation_id}, Query: {cust_query}, Query_Type: {"Normal"}, Execution_Time: {time.time() - start_time}, Status: {f"Instance failed with deployment {deployment_names[i]}. Trying the next instance"}, Error: {e}')
    else:                    
        print('>>> The query is not generic and asking info from knowledge base, so fetching information from PDF data.')

        # response will return : [cust_query, textdata_response, citation_texts, score]
        response =  query_pdfs_intel(cust_query, outside_context, email, correlation_id, verbosity, recent_chat_history=recent_chat_history) 
        # response =  query_pdfs_intel(cust_query, outside_context, email, correlation_id) 

        json_data = json.dumps({'queryType':None,'textdata':response[1], 'citation':response[2], "respType":1})

        return json_data
