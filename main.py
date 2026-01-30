from flask import Flask, render_template, request,Response
from flask import send_file
from flask import jsonify
import re
import socket
from query_unst import query_unstructured
import os, shutil
from doc_intel_search import create_embedding
from werkzeug.utils import secure_filename
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from utils import get_file_types_csv
from config.config_validator import validate_ini_file
from flask_cors import CORS,cross_origin

import logging

logging.getLogger('flask_cors').level = logging.DEBUG

CONFIG_LOC = os.environ.get('CONFIG_LOC', str(Path('config/config.ini')))
print("Using config file: ", CONFIG_LOC)
# validate and read config file:
CONFIG = validate_ini_file(CONFIG_LOC)

host_name = socket.gethostname()
ip_address = socket.gethostbyname(host_name)
DEPLOY_PORT = CONFIG['deployment']['port']
app = Flask(__name__)
# CORS(app) 
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})
# CORS(app, resources={r"*": {"origins": "*"}})
doc_directory_option = CONFIG['files_location']['uploads_dir']
Path(doc_directory_option).mkdir(exist_ok=True)
Path(doc_directory_option + "/user_uploads").mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = CONFIG["files_location"]["unstructured_dtype"]

print("\n\n")

print("\n ---- START ---- \n")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

@app.route('/')
def gateway_test():
    return 'hi'

@app.route('/chatbot-dev-nexus')
def upload():

    return render_template('file_handler.html')


@app.route("/chatbot-dev-nexus/upload", methods=["GET", "POST"])
def upload_save_create_embedding():
    try:
        correlation_id = request.headers.get("AN-CorrelationID")
        if request.method == "POST":
            files = request.files.getlist("files")
            print("Total files ",len(files))
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    doc_directory = doc_directory_option + "/user_uploads"
                    file.save(os.path.join(doc_directory , file.filename) )
                    print("filename is", file.filename)
                else:
                    return "Invalid file type. Please upload files with extensions: pdf, docx, pptx."
            print("Files uploaded successfully!")        
            
            file_index = create_embedding()
            
            # TO store the logs for file uploads and vector_DB creation with proper timestamp
            if file_index!=None:
                files_name = file_index[0]
                hash_index = file_index[1]
                import pytz
                ist = pytz.timezone('Asia/Kolkata')
                time_now = datetime.now(ist)
                time_now = time_now.strftime('%Y-%m-%d %H:%M:%S')
                # contains details of docs uploaded and vector db creation timewise >> check and delete later if not needed
                try:
                    file_vector_index_map = pd.read_csv("file_vector_index_hash_map.csv")            
                except Exception as e:
                    details_dict = {'files_name' : [files_name], 'vector_db_hash_name' : [hash_index], "update_time":[time_now] }                                
                    pd.DataFrame(details_dict).to_csv('file_vector_index_hash_map.csv')
                # Using concat
                details_dict = {'files_name' : [files_name], 'vector_db_hash_name' : [hash_index], "update_time":[time_now] }
                details_df = pd.DataFrame(details_dict)

                file_vector_index_map = pd.concat([file_vector_index_map, details_df], ignore_index = True)
                file_vector_index_map.reset_index()

                file_vector_index_map.to_csv("file_vector_index_hash_map.csv", index=False)

                print("Updated the vector db hashed name as ", hash_index)

            selected_files = [file.filename for file in files]
            all_files = os.listdir(doc_directory_option + "/user_uploads")

            uploaded_files = [file for file in selected_files if file in all_files]
            print("uploaded_files ", uploaded_files)
            response_data = json.dumps({"selected_files":selected_files, "uploaded_files": uploaded_files})
            return response_data

        else:
            response_data = json.dumps({"selected_files":[], "uploaded_files": []})
            return response_data
    except Exception as e:
            return jsonify({"errorCode": "ds-201",
                    "message": "There was some error in uploading the files and processing them. Please try again after some time. If the issue persists, please contact the chatbot-dev-nexus support team at [chatbot-dev-nexus.support@xx.com]. They will be happy to assist you further.",
                    "AN-CorrelationID": correlation_id}), 999


@app.route('/chatbot-dev-nexus/download_files', methods=['POST'])
def download_pdf_files():
    try:
        correlation_id = request.headers.get("AN-CorrelationID")
        if request.method == "POST":
            try:
                file_selected = ""
                if request.is_json: #for API request
                    file_selected = request.get_json['filename']
                else: #for HTML form request
                    file_selected = str(request.form['filename'])

                print("file_selected ", file_selected)
                get_file_types_csv(doc_directory_option)  # find better options
    #            download_files = pd.read_csv("download_helper.csv")
                download_files = get_file_types_csv(doc_directory_option)    ## Changing this to skip creation and use of df download_helper
                selected_file_path = download_files[download_files["Filename"] == file_selected]["Path"].values[0]
                print("selected_file_path ", selected_file_path)
                # single file download for now
                # filename  = re.search(r'[\\/]+([^\\/]+)$', file_path)
                filename  = re.search(r'[\\/]+([^\\/]+)$', selected_file_path)
                print("filename label out ", filename)
                if filename:
                    filename = filename.group(1)
                    print("filename label ", filename)
                return send_file(selected_file_path, as_attachment=True, download_name=filename)            
            except Exception as e:
                return f"We got into an error while file download \n or This file doesn't exist \n Error : {e}", 999
    except Exception as e:
        return jsonify({"errorCode": "ds-202",
            "message": "There was some issue in downloading the file. Please retry after sometime. If the issue persists, please contact the ASK chatbot-dev-nexus support team at [chatbot-dev-nexus.support@nexus.com]. They will be happy to assist you further.",
            "AN-CorrelationID": correlation_id}), 999
    

@app.route('/chatbot-dev-nexus/delete_all', methods=['GET', 'POST'])
def clear_database():
    try:
        correlation_id = request.headers.get("AN-CorrelationID")
        uploads_directory_path = (doc_directory_option + '/user_uploads')
        to_be_deleted = []
        try:
            # Get a list of all the files in the directory
            files = os.listdir(uploads_directory_path)

            # Iterate through the files and delete them one by one
            
            for file_name in files:
                file_path = os.path.join(uploads_directory_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Deleted file: {file_path}")
                    to_be_deleted.append(file_name)

            print(f"All files in the directory {uploads_directory_path} have been deleted.")
        except Exception as e:
            print(f"An error occurred: {e}")

        clear_db_response_data = {"deleted_files": to_be_deleted }

        dir = '../chatbot-dev/Vector_DB'
        for files in os.listdir(dir):
            path = os.path.join(dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
        
        #file_index = create_embedding()
        # Get a list of all the files in the directory
        files = os.listdir(doc_directory_option)
        print("type of files ", type(files))
        print("len of files ", len(files))

        if (len(files)>1): # >1 as we have user_uploads directory inside uploads (make sure to update accordingly)
            file_index = create_embedding()
        else:
            file_index = None

        if file_index==None:
            clear_db_response_data["new_db_created"] =  'No'
        else:
            clear_db_response_data["new_db_created"] = 'Yes'

        return jsonify(clear_db_response_data)
    except Exception as e:
        return({"errorCode": "ds-203",
            "message": "Sorry, we could not delete the files. Please contact our support team at chatbot-dev-nexus.support@xx.com.",
            "AN-CorrelationID": correlation_id}), 999

@app.route("/chatbot-dev-nexus/home", methods=["GET", "POST"])
def ask_question():
    all_files = os.listdir(doc_directory_option + "/user_uploads")
    return render_template('home.html', files=all_files)


@app.route("/chat", methods=["GET"])
def chat():
    if request.method == "GET":
        cust_query = request.args.get("cust_query", default="NA")
    
        correlation_id = request.headers.get("AN-CorrelationID")
        email = "fake@ggmail.com"
        verbosity = "medium"

        try:
            outside_context = request.form.get('selection')
            print()
            print("context is ", outside_context)
        except:
            print("outside_context not provided, default is no")
            outside_context = "no"

        result = query_unstructured(cust_query, outside_context, email, correlation_id,verbosity)

        return jsonify(result)

    return {"error":"Please check method"}    

    



    # result = request.form
    # print()
    # try:
    #     # Attempt to parse JSON
    #     data = request.get_json()
    #     print("json",data)
    #     if data is None:
    #         return {}
    # except:
    #     # Fallback to form data
    #     data = request.form.to_dict()
    #     if data is None:
    #         return {}
        
    
    # correlation_id = request.headers.get("AN-CorrelationID")
    # cust_query = data["cust_query"]
    # email = "fake@ggmail.com"
    # verbosity = "medium"
    
    # print("^^^^^^^^^^^^^^^^^^^^^")
    # print(f"{correlation_id} :: {cust_query} :: {email} :: {verbosity}")
    # print("^^^^^^^^^^^^^^^^^^^^^")

    # try:
    #     outside_context = request.form.get('selection')
    #     print("context is ", outside_context)
    # except:
    #     print("outside_context not provided, default is no")
    #     outside_context = "no"

    # # result = query_pdfs_intel(cust_query, outside_context) # from doc_search
    # result = query_unstructured(cust_query, outside_context, email, correlation_id,verbosity)

    # return jsonify(result)
    # return request.json, request.form


# Initialize an empty string as the global email variable to use it for user_data_injection in prompt
email = ''

@app.route('/chatbot-dev-nexus/answer', methods=['POST','GET'])
@cross_origin(origins="*")
def answer():
    print("@@@@@@@@@@@@@@@@@@")
    print(request.form)
    print("@@@@@@@@@@@@@@@@@@")

#    try:
    correlation_id = request.headers.get("AN-CorrelationID")
    global email
    cust_query = str(request.form['cust_query'])
    email = str(request.form['user_id'])
    verbosity = str(request.form['verbosity'])
    try:
        outside_context = str(request.form['selection'])
        print("context is ", outside_context)
    except:
        print("outside_context not provided, default is no")
        outside_context = "no"

    # result = query_pdfs_intel(cust_query, outside_context) # from doc_search
    result = query_unstructured(cust_query, outside_context, email, correlation_id,verbosity)
    result_dict = json.loads(result)
    print(">>>>>",type(result_dict))
    
    # print(result_dict["citation"])

    # to render results in HTML for DS backend testing 
    # return render_template('result2.html', result_dict = result_dict, cust_query=cust_query)
    #result.headers["Access-Control-Allow-Origin"] = "*"
    
    #return Response(result, mimetype='text/plain')
    # print(result_dict)
    return result_dict
#    except Exception as e:
#         return jsonify({"errorCode": "ds-204",
#             "message": "Sorry, there is some issue with the Azure OpenAI services, please retry aftersometime. If the issue persists, please contact the ASK chatbot-dev-nexus support team at [chatbot-dev-nexus.support@nexus.com].",
#             "AN-CorrelationID": correlation_id}), 999

@app.route("/chatbot-dev-nexus/health", methods=["GET"])
def health_monitoring():
    health_response = {"status":"UP"}
    health_response = json.dumps(health_response)
    return health_response

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.logger.disabled = True
    #CORS(app)
    #CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

    #CORS(app, support_credentials=True)
    app.run(host='0.0.0.0', port=DEPLOY_PORT, debug=True)
   
