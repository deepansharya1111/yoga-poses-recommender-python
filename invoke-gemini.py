import os
import logging
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
import vertexai


load_dotenv()
logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main():
    try:
        # Initialize Vertex AI SDK
        logging.info("Initializing Vertex AI SDK")
        vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
        logging.info("Done Initializing Vertex AI SDK")
        model = VertexAI(model_name=os.getenv("GEMINI_MODEL_NAME"), verbose=True)
        response = model.invoke("Tell me something about Yoga")
        return response
    except Exception as e:
        logging.error(f"Error invoking Gemini: {e}")
        return None
        
if __name__ == "__main__":
    print(main())