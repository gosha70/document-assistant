# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import argparse
from langchain.vectorstores import Chroma

from embeddings.embeddings_constants import CHROMA_SETTINGS

from models.model_info import ModelInfo


if __name__ == "__main__": 
    """Utility to test the vectorstore."""

     # Create the parser
    parser = argparse.ArgumentParser(description="Creating the embedding database.")

    # Add the arguments
    parser.add_argument('--persist_directory', type=str, help='The path to the directory with the vectorsstore to test.')
    parser.add_argument('--test_question', type=str, help='(Optional) Indicates if the created vectordatastore should be tested.', default=None)
    parser.add_argument('--collection_name', type=str, help='The name of embedding vectorestore.', default=None)
      
    # Parse the arguments
    args = parser.parse_args()

    embedding = ModelInfo.create_embedding(model_name=None)
    docs_db = Chroma(
        persist_directory=args.persist_directory,
        embedding_function=embedding,
        collection_name=args.collection_name,
        client_settings=CHROMA_SETTINGS,
    )   
    print(f" Collection count: {docs_db._collection.count()}")
    doc_ids = docs_db.get()["ids"]
    print(f"The vectorestore stors {len(doc_ids)} documents")

    if args.test_question is not None:        
        retriever = docs_db.as_retriever()
        answer = retriever.invoke(args.test_question)
        print(f"ANSWER: {answer[0].page_content}")


    
