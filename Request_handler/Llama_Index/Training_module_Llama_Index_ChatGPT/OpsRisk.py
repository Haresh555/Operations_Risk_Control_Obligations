import traceback

from langchain import OpenAI
import openai
from llama_index import ServiceContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader
import os
import sys
import time


from llama_index import StorageContext, load_index_from_storage

os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.environ["OPENAI_API_KEY"]


def construct_index(directory_path):
    index = None
    max_input_size = 4096
    num_outputs = 500
    chunk_size_limit = 1024
    path = os.listdir("./Doc_index_file")
    file_metadata = lambda x: {"filename": x}
    reader = SimpleDirectoryReader(directory_path, file_metadata=file_metadata)
    documents = reader.load_data()
    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio=0.1,chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt 3.5 turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)


    if len(path)>0:
        print('[LOG] Identified available context index for Doc_index_file. Continuing the training from where it was left ')
        storage_context = StorageContext.from_defaults(persist_dir="./Doc_index_file")
        index = load_index_from_storage(storage_context)
        print(index)
    else:
        print('[LOG] There is no existing index to pick for Doc_index_file ,Reading the document for commencing training')
        print('[LOG] Total length of  document:',len(documents))


        print('[LOG] Creating Index  for all documents of length :',len(documents))
        index = GPTVectorStoreIndex.from_documents(
             service_context=service_context,documents=[]
        )
    #documents = documents[100:100]
    for i, d in enumerate(documents):
           #print('Document : ',d )
            try:

                index.insert(document=d, service_context=service_context, overrite=False)
                index.storage_context.persist("./Doc_index_file")
                print('--->',d)
                print('INDEX :',index)

                time.sleep(30)
            except   Exception as e:
                traceback.print_exc()
                time.sleep(30)
                pass
    return index


construct_index("./docs")