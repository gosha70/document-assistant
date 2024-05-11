# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os
import argparse
import logging
import json
import time
import dash
import base64
import threading
import tempfile
from datetime import datetime
from langchain_community.vectorstores import Chroma
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from retrieval_constants import CURRENT_DIRECTORY

from models.model_info import ModelInfo

from embeddings.embeddings_constants import DEFAULT_COLLECTION_NAME
from embeddings.embedding_database import load_vector_store
from retrieval_qa import create_retrieval_qa
from prompt_info import PromptInfo
from embeddings.unstructured.document_splitter import DocumentSplitter
from embeddings.file_watcher import FileWatcher

with open('app/app_config.json', 'r') as file:
    app_config = json.load(file)

model_info = ModelInfo() # DEFAULT_MODEL_NAME = "hkunlp/instructor-large" 
temp_directory = None
verbose = False
docs_db = None  
qa_service = None
next_question_delay = app_config["next_question_delay"]
# The number of seconds passed b/w questions
if next_question_delay is None or next_question_delay < 1:
    next_question_delay = 1
print(f"Minimum wait in seconds b/w questions: {next_question_delay}")

# System prompt muat be specified for embeddings
app_system_prompt = app_config["system_prompt"]
print(f"System prompts: {app_system_prompt}")

def count_documents() -> int:
    # Count of loaded Documents
    if docs_db is None:
        return 0    
    dic = docs_db.get()["ids"]
    return len(dic)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#app.config.suppress_callback_exceptions = True

SYSTEM_ERROR = "I apologize, but I'm unable to provide a specific answer to your question based on the information currently available to me.\nMy ability to respond accurately depends on a variety of factors, including the scope of my training data and the specific details of your query.\nIf you have any other questions or need assistance with a different topic, please feel free to ask, and I'll do my best to help."

# This layout closely follows the structure of popular chat applications
app.layout = dbc.Container([    
        html.Div([
            html.Div([  # Div for image and text
                html.Img(src=app_config["chat_logo"], className='title-image'),  # Custom Image
                html.Div([  # Nested Div for title and subtitle
                        html.Span(app_config["chat_title"], className='title-span'),  # Title text
                        html.Span(app_config["chat_subtitle"], className='subtitle-span')  # Subtitle text
                ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '10px'})
            ], style={'textAlign': 'left', 'color': '#E4E4E7', 'display': 'flex', 'alignItems': 'center'})            
        ], className='header-container'),

        html.Hr(className="chat-bottom-hr"),   

        # Pevents the concurrent request to LLM
        dcc.Store(id='click-store', data={'last-click-time': 0}),
        dcc.Store(id='upload-files-store', data={'file-names': []}),

        dbc.Row(
            dbc.Col(
                html.Div(id='chat-box-id', className='chat-box chat-box-hidden'),
                width=12,
            ),
            className="mb-2"
        ),

        html.Hr(className="chat-top-hr"),     

        dbc.Row(
            [
                dbc.Col(  
                    dcc.Upload(
                        id='upload-data',                        
                        children=dbc.Button(
                            html.Img(src=app_config["file_attach_icon"], className='ask-button-img'), 
                            id='file-attach-button-id', 
                            n_clicks=0,
                            className="mb-2 ask-button", 
                            disabled=False,
                            color=app_config["color_shceme"],
                            title=app_config["attach_button_title"]
                        ),
                        # Do not allow multiple files to be uploaded
                        multiple=True
                    ),  
                    width=1,
                    className='my-col',
                ),
                dbc.Col(
                    dcc.Textarea(
                        id='message-input-id', 
                        placeholder=app_config["chat_ask_placeholder"], 
                        className='message-input',
                        disabled=False
                    ),
                    width=10,
                    className='my-col',
                ),
                dbc.Col(
                    dbc.Button(
                        html.Img(src=app_config["ask_button"], className='ask-button-img'), 
                        id='ask-button-id', 
                        n_clicks=0,
                        className="mb-2 ask-button", 
                        color=app_config["color_shceme"],
                        disabled=False
                    ),
                    width=1,
                    className='my-col',
                )
            ],
            className='g-0 align-items-center', 
        ),
                
        html.Div([
                html.Div([ 
                    html.P(
                        app_config["copyright"]
                    ),
                    # Toggle button for Info about AI model
                    html.Button(
                        html.Img(src=app_config["info_button"], className='info-button-img', id='info-button-img-id'), 
                        id='info-button-id', 
                        className="mb-2 info-button", 
                        n_clicks=0,
                        title="Show Model Information"
                    ),
                    html.Div([
                        html.Table([
                            html.Tr([html.Td('Model Name:'), html.Td(model_info.model_name)]),
                            html.Tr([html.Td('Document Splits:'), html.Td(id='document-splits-count', children=count_documents())])  
                        ])
                    ], style={'display': 'none'}, id='info-table')
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}) 
            ],
            className='custom-footer-style',
            style={'textAlign': 'left', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}
        ), 

        html.Div(id='output-track-div', hidden=True),
    ],
    fluid=True,
    className='container-style' 
)

def formatted_datetime():
    # Format the date and time to display in the chat to: 'Dec 25, 2023 12:00 PM'
    return datetime.now().strftime('%b %d, %Y %I:%M %p')

def check_get_click_time(n_clicks, data):    
    current_time = time.time()
    # Get the last click time from the store
    last_click_time = data['last-click-time']    
    if last_click_time is None:
        return current_time
        
    # Calculate the time since the last click
    time_since_last_click = current_time - last_click_time 
    if n_clicks is None or time_since_last_click <= next_question_delay:
        raise PreventUpdate
    
    return current_time

@app.callback(
    [Output('click-store', 'data'),
     Output('message-input-id', 'value', allow_duplicate=True),
     Output('message-input-id', 'disabled', allow_duplicate=True),
     Output('ask-button-id', 'disabled', allow_duplicate=True)],
    [Input('ask-button-id', 'n_clicks')],
    [State('click-store', 'data')],
    prevent_initial_call=True
)
def update_click_store(n_clicks, data):
    data['last-click-time'] = check_get_click_time(n_clicks, data)
    return data, app_config["wait_info"], True, True


@app.callback(
    [Output('info-table', 'style'),
     Output('document-splits-count', 'children'),
     Output('info-button-img-id', 'src'),
     Output('info-button-id', 'title')],
    [Input('info-button-id', 'n_clicks')]
)
def toggle_table_visibility(n_clicks):
    if n_clicks % 2 == 0:  # Even number of clicks - hides the table
        return {'display': 'none'}, 0, app_config["info_button"], "Show Model Information"
    else:  # Odd number of clicks - shows the table
        doc_count = count_documents()
        return {'display': 'inline-block'}, doc_count, app_config["hide_button"], "Hide Model Information"

def save_file_content(file_name: str, file_content):
    """
    Converts the specified file content into Documents

    Parameters:
    - file_name (str): The file name
    - file_content (str): The file content
    """
    content_type, content_string = file_content.split(',')
    # Decode the base64 content
    decoded = base64.b64decode(content_string)
    # Save decoded content to a temporary file
    temp_file_path = os.path.join(temp_directory, file_name)
    with open(temp_file_path, 'wb') as tmpfile:  # Use 'wb' for writing in binary mode
        tmpfile.write(decoded)
        print(f"The temporary file was created: {temp_file_path} ...")  

@app.callback(
    Output('upload-files-store', 'data'),    
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('upload-files-store', 'data'))
def upload_and_process_files(list_of_contents, list_of_names, list_of_dates, data):
    if list_of_contents is not None and temp_directory is not None and docs_db is not None:
        cached_files = data['file-names']
        for content, filename, date in zip(list_of_contents, list_of_names, list_of_dates):
            if any(filename in s for s in cached_files):
                continue
            else:
                save_file_content(file_name=filename, file_content=content)
                cached_files.append(filename)
                data['file-names'] = cached_files

    return data

@app.callback(
    [Output('chat-box-id', 'children'),
     Output('chat-box-id', 'className'),
     Output('message-input-id', 'value'),
     Output('message-input-id', 'disabled'),
     Output('ask-button-id', 'disabled')],
    [Input('ask-button-id', 'n_clicks')],
    [State('message-input-id', 'value'),
     State('chat-box-id', 'children'),
     State('click-store', 'data')]
)
def update_chat(n_clicks, message, chat_elements, data):   
    if message is None:
        raise PreventUpdate   
    
    data['last-click-time'] = check_get_click_time(n_clicks, data)
    # Disable the button at the start
    chat_elements = chat_elements or []
    try:
        user_icon = app_config["user_icon"]
        system_icon = app_config["system_icon"]
        system_error_icon = app_config["system_error_icon"]

        question_div = html.Div(
            [
                html.Div(
                    [
                        html.Img(src=user_icon, className="chat-icon user-icon"),
                        html.Span(formatted_datetime(), className="chat-datetime") 
                    ],
                    className="chat-header"
                ),
                html.P(message, className="chat-message user-message")
            ],
            className="chat-bubble user-bubble"
        )

        try:
            answer = get_answer(message)  # Placeholder for your answer generation logic
            answer_div = html.Div(
                [                    
                    html.Div( 
                        [
                            html.Img(src=system_icon, className="chat-icon system-icon"),
                            html.Span(formatted_datetime(), className="chat-datetime") 
                        ],
                        className="chat-header"
                    ),
                    # html.P(answer, className="chat-message system-message")
                    dcc.Markdown(answer.replace('\n', '\n\n'), className="chat-message system-message")
                ],
                className="chat-bubble system-bubble"
            )
        except Exception as error:
            logging.error(f"Failed to answer on the question: '{message}'.\nError: {str(error)}", exc_info=True)
            answer_div = html.Div(
                [
                    html.Img(src=system_error_icon, className="chat-icon system-icon"),
                    html.P(SYSTEM_ERROR, className="chat-message system-error-message")
                ],
                className="chat-bubble system-bubble"
            )
        
        chat_elements.extend([question_div, answer_div])
    except Exception as error:     
        logging.error(f"Failed to process the messsage: '{message}'.\nError: {str(error)}", exc_info=True)

    return chat_elements, 'chat-box-shown', '', False, False


# Placeholder function for generating system answers
def get_answer(question):
    # Get the answer from the chain
    logging.info(f"Asking the question:\n {question}")  
    results = qa_service(question)
    answer, docs = results["result"], results["source_documents"]
    logging.info(f"Got the answer on the question:\n {question}.") 
    if verbose: 
        log_message = f"=============\n{answer}\n"
        for document in docs:
            log_message = log_message +  f">>> {document.metadata['source']}:{document.page_content}\n"
        log_message = log_message + "=============" 
        logging.info(log_message)
    
    return answer
    
if __name__ == '__main__':
    # Set the logging level to INFO    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create the parser
    parser = argparse.ArgumentParser(description="Starting Chat Application using the vectorstore.")

    # Add the arguments
    parser.add_argument(
        '--port', 
        type=int, 
        help='Port for Chat Application. Defaults to 5001.', 
        default=5001
    )
    parser.add_argument(
        '--host', 
        type=str, 
        help='Host where Chat Application run. Defaults to 127.0.0.1. Set to 0.0.0.0 to access Application from other devices', 
        default='127.0.0.1'
    )
    parser.add_argument(
        '--persist_directory', 
        type=str, 
        help='The path to the directory where the vectorstore will be persisted. Defaults to the root directory.', 
        default=CURRENT_DIRECTORY
    )
    parser.add_argument(
        '--collection_name', 
        type=str, 
        help='The name of vectorestore.', 
        default=DEFAULT_COLLECTION_NAME
    )
    parser.add_argument(
        '--prompt_template', 
        type=str, 
        help='(Optional) the promp template type: "llama", "mistral"',
        default=None
    )
    parser.add_argument(
        '--history', 
        type=bool, 
        help='(Optional) The flag indicates if LLM supports the propmpt history.', 
        default=False
    )
    parser.add_argument(
        '--verbose', 
        type=bool, 
        help='(Optional) The flag indicates that the results must be printed out to the console.', 
        default=False
    )
    
    # Parse the arguments
    args = parser.parse_args()

    verbose = args.verbose   

    prompt_info = PromptInfo(app_system_prompt, args.prompt_template, args.history)

    logging.info(f"Input arguments:\n===\n{args}\n===\nLoading the vectorstore from {args.persist_directory} ...")
    docs_db = load_vector_store(
        model_name=model_info.model_name, 
        collection_name=args.collection_name, 
        persist_directory=args.persist_directory
    )
    
    if docs_db is None:
        logging.error(f"Failed to load the vectorstore from {args.persist_directory}.")  
    else:
        dic = docs_db.get()["ids"]
        documents_count = len(dic)
        logging.info(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nLoaded the vectorstore with {documents_count} documents.\nLLM model name: {model_info.model_name}.\nSystem Prompt:\n---\n{app_system_prompt}\n---\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        qa_service = create_retrieval_qa(model_info=model_info, prompt_info=prompt_info, vectorstore=docs_db)
        if qa_service is None:
            logging.error(f"Failed to initialize the retrieval framework for the vectorstore located in {args.persist_directory}.")  
        else:  
            temp_directory = tempfile.mkdtemp()
            if not temp_directory.endswith('/'):
                temp_directory = temp_directory + '/'
            document_splitter = DocumentSplitter(logging)

            # Start the watcher
            watcher_thread = threading.Thread(target=FileWatcher.run_file_watcher, args=(docs_db, document_splitter, temp_directory), daemon=True)
            watcher_thread.start()  

            # Start the web server
            app.run_server(debug=True, host=args.host, port=args.port, use_reloader=False)   
