import argparse
import logging
import json
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from retrieval_constants import CURRENT_DIRECTORY
from models.model_info import ModelInfo
from embeddings.embedding_database import load_vector_store
from retrieval_qa import create_retrieval_qa
from prompt_info import PromptInfo

with open('app/app_config.json', 'r') as file:
    app_config = json.load(file)

verbose = False
qa_service = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SYSTEM_ERROR = "I apologize, but I'm unable to provide a specific answer to your question based on the information currently available to me.\nMy ability to respond accurately depends on a variety of factors, including the scope of my training data and the specific details of your query.\nIf you have any other questions or need assistance with a different topic, please feel free to ask, and I'll do my best to help."
 

# This layout closely follows the structure of popular chat applications
app.layout = dbc.Container([
        html.H1([    
            html.Img(src=app_config["chat_logo"], className='title-image'),  # Custom Image        
            html.Span(app_config["chat_title"], className='title-span')  # Title with some space
        ], style={'textAlign': 'left', 'color': '#E4E4E7'}),

        html.Hr(className="chat-bottom-hr"),   
        
        dbc.Row(
            dbc.Col(
                html.Div(id='chat-box', className='chat-box', style={'display': 'none'}),
                width=12,
            ),
            className="mb-2"
        ),

        html.Hr(className="chat-top-hr"),     

        dbc.Row(
            [
                dbc.Col(
                    dcc.Textarea(
                        id='message-input', 
                        placeholder=app_config["chat_ask_placeholder"], 
                        className='message-input'
                    ),
                    width=10,
                ),
                dbc.Col(
                    dbc.Button(
                        html.Img(src=app_config["ask_button"], className='ask-button-img'), 
                        id='send-button', 
                        n_clicks=0, 
                        className="mb-2",
                        style={'background-color': 'transparent', 'border': 'none'}
                    ),
                    width=2,
                )
            ]
        ),

        html.P(
            app_config["copyright"],
            className="custom-footer-style"
        )
    ],
    fluid=True,
    className='container-style' 
)

@app.callback(
    [Output('chat-box', 'children'), Output('chat-box', 'style')], 
    [Input('send-button', 'n_clicks')],
    [State('message-input', 'value'), State('chat-box', 'children')]
)
def update_chat(n_clicks, message, chat_elements):
    if n_clicks > 0:
        chat_elements = chat_elements or []
        user_icon = app_config["user_icon"]
        system_icon = app_config["system_icon"]
        system_error_icon = app_config["system_error_icon"]

        question_div = html.Div(
            [
                html.Img(src=user_icon, className="chat-icon user-icon"),
                html.P(message, className="chat-message user-message")
            ],
            className="chat-bubble user-bubble"
        )
        
        try:
            answer = get_answer(message)  # Placeholder for your answer generation logic
            answer_div = html.Div(
                [
                    html.Img(src=system_icon, className="chat-icon system-icon"),
                    html.P(answer, className="chat-message system-message")
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
        return chat_elements, {'display': 'block'}    
    elif n_clicks == 0:
        raise PreventUpdate
    return chat_elements, {'display': 'none'}

# Placeholder function for generating system answers
def get_answer(question):
    # Get the answer from the chain
    results = qa_service(question)
    answer, docs = results["result"], results["source_documents"]
    if verbose: 
        log_message = f"=============\nAnswer: {answer}\n"
        for document in docs:
            log_message = log_message +  f">>> {document.metadata['source']}:{document.page_content}\n"
        log_message = log_message + "=============" 
        logging.info(log_message)
    
    return answer

if __name__ == '__main__':
    # Set the logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Create the parser
    parser = argparse.ArgumentParser(description="Starting Chat Application using the embedding database.")

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
        help='The path to the directory where the embedding database will be persisted. Defaults to the root directory.', 
        default=CURRENT_DIRECTORY
    )
    parser.add_argument(
        '--system_prompt', 
        type=str, 
        help='(Optional) The system instruction for Retrieval Q/A LLM.'
    )
    parser.add_argument(
        '--prompt_template', 
        type=str, 
        help='(Optional) the promp template type: "llama", "mistral"'
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

    model_info = ModelInfo() # DEFAULT_MODEL_NAME = "hkunlp/instructor-large" 

    prompt_info = PromptInfo(args.system_prompt, args.prompt_template, args.history)


    logging.info(f"Loading the embedding database from {args.persist_directory} ...")
    docs_db = load_vector_store(model_info.model_name, persist_directory=args.persist_directory)
    
    if docs_db is None:
        logging.error(f"Failed to load the embedding database from {args.persist_directory}.")  
    else:
        logging.info(f"Initializing the retrieval framework for the embedding database: {docs_db} ...")  
        qa_service = create_retrieval_qa(model_info=model_info, prompt_info=prompt_info, vectorstore=docs_db)
        if qa_service is None:
            logging.error(f"Failed to initialize the retrieval framework for the embedding database: {docs_db}.")  
        else:    
            app.run_server(debug=False, host=args.host, port=args.port)   
