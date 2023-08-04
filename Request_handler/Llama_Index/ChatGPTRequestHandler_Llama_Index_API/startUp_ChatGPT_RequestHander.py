import os
import traceback
from datetime import datetime

import openai
from Cython.Compiler.PyrexTypes import memoryviewslice_type
from flask import Flask ,request
from llama_index import StorageContext, load_index_from_storage
import configparser

app = Flask(__name__)
global  index,query_engine
def logStep( msg):
    print(msg)

def initProperties():
    logcontent = str(datetime.now())+'[INFO] Initializing property file for Environment Header API'
    logStep(logcontent)
    global openapi_key,tibco_error_code_dir,hostname,port
    prop = configparser.ConfigParser()
    prop.read('EnvironmentHealer_props.ini')
    openapi_key = prop['OPENAI']['api_key']
    openai.api_key = openapi_key

    tibco_error_code_dir = prop['FILESHARE']['tibco_error_code_dir']
    hostname = prop['CONNECTIONS']['hostname']
    port = prop['CONNECTIONS']['port']
    logcontent = str(datetime.now()) + '[INFO] Initializing Successful'
    logStep(logcontent)


def getResponseFromChatGPT(input_text):
    global tibco_error_code_dir
    logcontent = str(datetime.now()) + '[INFO] Received Prompt Request'
    logStep(logcontent)

    try:
        gpt_response = query_engine.query(input_text)
        response = {'status':'success','msg':str(gpt_response)}
        logcontent = str(datetime.now()) + '[INFO] Sent Successful Completion from ChatGpt'
        logStep(logcontent)
        return response
    except Exception as e:
        traceback.print_exc()
        logcontent = str(datetime.now()) + '[ERROR] Exception Occurred'
        logStep(logcontent)
        response = {'status':'error','msg':'Technical Error'}
        return response


@app.route("/getResponseForPrompt/" ,methods=['POST'])
def getResponseForPrompt():
    input_data = request.data

    response = getResponseFromChatGPT(str(input_data))
    return response

if __name__ =='__main__':
    global hostname, port,tibco_error_code_dir
    initProperties()
    storage_context = StorageContext.from_defaults(persist_dir=tibco_error_code_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    logcontent = str(datetime.now()) + '[INFO] Hosting Environment Healer API'
    logStep(logcontent)

    app.run(hostname, port =port )



