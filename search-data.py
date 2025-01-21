"""Module providing an example of Firestore Vector Similarity Search"""
import argparse
import logging

import os
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()

logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search documents in a Firestore vector store."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The search query prompt.",
    )
    return parser.parse_args()

def search(query:str):
    """Executes Firestore Vector Similarity Search"""
    embedding = VertexAIEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME"),
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION")
    )

    client = firestore.Client(
           project=os.getenv("PROJECT_ID"),
           database=os.getenv("DATABASE")
           )
    
    vector_store = FirestoreVectorStore(
                        client=client,
                        collection=os.getenv("COLLECTION"),
                        embedding_service=embedding)
   
    logging.info(f"Now executing query: {query}")
    results: list[Document] = vector_store.similarity_search(
                                    query=query,
                                    k=int(os.getenv("TOP_K")),
                                    include_metadata=True)
    for result in results:
        print(result.page_content)

# Example: Suggest some exercises to relieve back pain
# python search-data.py --prompt "Suggest some exercises to relieve back pain"
if __name__ == "__main__":
    args = parse_arguments()
    prompt = args.prompt
    search(prompt)

