import argparse
from langchain.vectorstores import Chroma
from embeddings.embeddings_constants import CHROMA_SETTINGS, DEFAULT_COLLECTION_NAME
from models.model_info import ModelInfo

def create_vectorstore(persist_directory, collection_name, embedding):
    """Create and return a Chroma instance."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=collection_name or DEFAULT_COLLECTION_NAME,
        client_settings=CHROMA_SETTINGS,
    )

def main(args):
    """Utility to merge the 'from' vectorstore to the target 'to' vectorstore."""

    embedding = ModelInfo.create_embedding(model_name=None)

    from_docs_db = create_vectorstore(args.from_persist_directory, args.from_collection_name, embedding)
    print(f"The from vectorestore of '{args.from_collection_name or DEFAULT_COLLECTION_NAME}' collection count: {from_docs_db._collection.count()}")

    to_docs_db = create_vectorstore(args.to_persist_directory, args.to_collection_name, embedding)
    print(f"The target vectorestore of '{args.to_collection_name or DEFAULT_COLLECTION_NAME}' collection count: {to_docs_db._collection.count()}")

    from_db_data = from_docs_db._collection.get(include=['documents', 'metadatas', 'embeddings'])

    to_docs_db._collection.add(
        embeddings=from_db_data['embeddings'],
        metadatas=from_db_data['metadatas'],
        documents=from_db_data['documents'],
        ids=from_db_data['ids']
    )
    to_docs_db.persist()

    print(f"After the merge, the target vectorestore stores {len(to_docs_db.get()['ids'])} documents")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merging two vectorestore.")
    parser.add_argument(
        '--from_persist_directory', 
        type=str, 
        required=True, 
        help='Path to the source vectorestore directory.'
    )
    parser.add_argument(
        '--from_collection_name', 
        type=str, 
        default=None, 
        help='Collection name of the source vectorestore.'
    )
    parser.add_argument(
        '--to_persist_directory', 
        type=str, 
        required=True, 
        help='Path to the target vectorestore directory.'
    )
    parser.add_argument(
        '--to_collection_name', 
        type=str, 
        default=None, 
        help='Collection name of the target vectorestore.'
    )
    args = parser.parse_args()
    main(args)
