import logging
import asyncio # Needed for async operations
import numpy as np # Needed for audio data handling
import wave # To create WAV file headers
import io # To handle in-memory file
import base64 # To encode audio/image data for JSON response
from PIL import Image # For image handling
from flask import Flask, request, jsonify, render_template, make_response
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter # Import FieldFilter
from google.cloud import aiplatform # Import Vertex AI platform SDK
import os
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from google import genai # Correct import for the new unified SDK
# Import necessary types from the SDK based on the notebook
from google.genai.types import (
    Content,
    LiveConnectConfig,
    Part,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    AudioTranscriptionConfig, # Import for transcription
    Tool,                     # Import Tool
    GoogleSearch,             # Import GoogleSearch tool
    GenerateContentConfig,    # Import config for generate_content
)
import urllib
from settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Simple in-memory store for conversation history (replace with proper session management for production)
conversation_histories = {}
# Example structure: 
# { 
#   'user_session_id': [ 
#      {'role': 'user', 'parts': [{'text': 'prompt1'}]}, 
#      {'role': 'model', 'parts': [{'text': 'response1'}]} 
#   ] 
# }


def search(query: str, num_results: int, expertise_filter: str = "Any"): # Add expertise_filter parameter
    """Executes Firestore Vector Similarity Search with optional metadata filtering"""
    embedding = VertexAIEmbeddings(
        model_name=settings.embedding_model_name,
        project=settings.project_id,
        location=settings.location,
    )

    client = firestore.Client(
        project=settings.project_id, database=settings.database
    )

    vector_store = FirestoreVectorStore(
        client=client,
        collection=settings.collection,
        embedding_service=embedding,
    )
    
    # --- Build Filter --- # Removed pre_filter logic

    # Fetch a larger number of results initially to increase chances of finding filtered items
    initial_k = 50 # Fetch more initially
    logging.info(f"Now executing query: {query} for initial k={initial_k} results")
    
    # IMPORTANT: We MUST set include_metadata=True to filter on it later
    initial_results: list[Document] = vector_store.similarity_search(
        query=query, 
        k=initial_k, 
        include_metadata=True, # MUST be True to access metadata for filtering
        # pre_filter=pre_filter # Removed pre_filter
    )

    # --- Filter results in Python ---
    filtered_results = []
    if expertise_filter and expertise_filter != "Any":
        logging.info(f"Filtering results in Python for expertise: {expertise_filter}")
        for result in initial_results:
            # Check if metadata exists and contains the nested expertise level
            # Using .get() for safer access in case keys are missing
            nested_metadata = result.metadata.get('metadata', {}) if hasattr(result, 'metadata') else {}
            level = nested_metadata.get('expertise_level')

            if level == expertise_filter:
                 filtered_results.append(result)
            # Add logging for misses if needed:
            # else:
            #    logging.debug(f"Skipping result due to filter mismatch: Level='{level}', Filter='{expertise_filter}', Meta='{result.metadata}'")
        logging.info(f"Found {len(filtered_results)} results matching filter.")
    else:
        logging.info("No expertise filter applied, using all initial results.")
        filtered_results = initial_results
        
    # Take the top 'num_results' from the filtered list
    final_results = filtered_results[:num_results]
    logging.info(f"Returning final {len(final_results)} results.")
    # --- End Python Filtering ---

    #Format the results for JSON output (using final_results)
    # Re-structure metadata slightly for consistency if needed, or ensure frontend handles it
    # The metadata structure might now be result.metadata instead of result.metadata['metadata']
    # Let's keep it as is for now, assuming frontend accesses result.metadata.metadata.field
    formatted_results = [
        {"page_content": result.page_content, "metadata": result.metadata} for result in final_results
    ]

    return formatted_results

@app.route("/", methods=["GET"])
def index():
    """Renders the HTML page with the search form."""
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_api():
    """API endpoint to search documents based on the prompt provided in json"""
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing prompt in request body"}), 400
        
        prompt = data["prompt"]
        # Get num_results from request, default to settings.top_k if not provided
        num_results = data.get('num_results', settings.top_k) 
        # Ensure it's an integer
        try:
            num_results = int(num_results)
        except (ValueError, TypeError):
             logging.warning(f"Invalid num_results value received: {data.get('num_results')}. Defaulting to {settings.top_k}")
             num_results = settings.top_k
        
        # Get expertise_filter from request, default to "Any"
        expertise_filter = data.get('expertise_filter', "Any")

        results = search(prompt, num_results, expertise_filter) # Pass expertise_filter
        return jsonify({"results": results}), 200
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return jsonify({"error": "An error occurred during search"}), 500

@app.route('/generate_audio', methods=['POST'])
async def generate_audio(): # Make the route async
    data = request.get_json()
    # Get pose_name, optional follow_up_text, optional voice_name, and optional language_code
    pose_name = data.get('pose_name')
    follow_up_text = data.get('follow_up_text') 
    voice_name = data.get('voice_name', 'Aoede') # Default to Aoede if not provided
    language_code = data.get('language_code', 'en-US') # Default to en-US if not provided

    if not pose_name:
        return jsonify({'error': 'Missing pose_name'}), 400

    try:
        # --- Use Live API for Audio Instruction Generation ---
        logging.info(f"Attempting Live API audio instruction generation for: {pose_name}")
        
        # Instantiate the client, explicitly specifying Vertex AI backend
        logging.info(f"Initializing genai.Client with vertexai=True, project={settings.project_id}, location={settings.location}")
        client = genai.Client(vertexai=True, project=settings.project_id, location=settings.location)

        # Configure the Live API connection for audio output AND text transcription
        logging.info(f"Using voice: {voice_name}, language: {language_code}")
        config = LiveConnectConfig(
            response_modalities=["AUDIO"], 
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
                ),
                language_code=language_code # Add language code here
            ),
            # Request transcription of the model's audio output
            output_audio_transcription=AudioTranscriptionConfig(), 
        )

        # --- Manage Conversation History & Prompt ---
        session_id = 'user1' # Use a fixed ID for now; replace with real session ID
        history = conversation_histories.get(session_id, []) # Get history first

        # Define base prompt text based on follow-up status
        if follow_up_text:
            base_user_text = f"Regarding the {pose_name} yoga pose, the user asked for clarification: '{follow_up_text}'. Please provide further audio instructions addressing this question. Format the instructions clearly in the transcription, ideally with a double newline between distinct steps or points."
            logging.info(f"Processing follow-up text (requesting text format, lang={language_code})")
        else:
            history = [] # Start with empty history for initial request
            base_user_text = f"Generate clear, step-by-step audio instructions for performing the {pose_name} yoga pose. Format the instructions clearly in the transcription, take pauses while narrating like humans (but do not write pause in the transcript, it is for your understanding), ideally using numbered steps with a double newline \n\n separator between each step(not to be spoken in audio). At the end, ask if the user needs clarification on any steps."
            logging.info(f"Processing initial prompt text (requesting text format, lang={language_code})")
            # Clear any old history for this session on a new initial request
            if session_id in conversation_histories:
                del conversation_histories[session_id]

        # Add system instruction if language is not English
        system_instruction = ""
        if language_code != 'en-US': # Assuming en-US is the default/English
            language_name = language_code.split('-')[0].upper() # Basic attempt
            lang_map = {"DE": "GERMAN", "ES": "SPANISH", "FR": "FRENCH", "HI": "HINDI", "ID": "INDONESIAN", "IT": "ITALIAN", "JA": "JAPANESE", "KO": "KOREAN", "NL": "DUTCH", "PL": "POLISH", "PT": "PORTUGUESE", "RU": "RUSSIAN", "TH": "THAI", "TR": "TURKISH", "VI": "VIETNAMESE", "CMN": "MANDARIN CHINESE", "AR": "ARABIC", "BN": "BENGALI", "GU": "GUJARATI", "KN": "KANNADA", "ML": "MALAYALAM", "MR": "MARATHI", "TA": "TAMIL", "TE": "TELUGU"}
            language_name = lang_map.get(language_name, language_name) # Use mapped name if available
            system_instruction = f"RESPOND IN {language_name}. YOU MUST RESPOND UNMISTAKABLY IN {language_name}.\n\n"
            logging.info(f"Adding system instruction for language: {language_name}")
        
        # Combine system instruction and base text
        current_user_text = f"{system_instruction}{base_user_text}" # Now current_user_text is guaranteed to be assigned
        logging.debug(f"Final user text being sent: {current_user_text}")

        # Prepare the full turn history to send to the model
        current_turn = Content(role="user", parts=[Part(text=current_user_text)])
        turns_to_send = history + [current_turn] # Send history (retrieved or empty) + current input

        logging.debug(f"Sending turns to Live API for session {session_id}: {turns_to_send}")
        # --- End History Management ---

        audio_data_list = []
        model_transcription_parts = [] # To collect text transcription

        async with client.aio.live.connect(model=settings.gemini_live_model, config=config) as session:
            # Send the full conversation context
            await session.send_client_content(turns=turns_to_send)

            # Receive and collect audio data AND transcription
            async for message in session.receive():
                # Collect Transcription
                if (
                    message.server_content.output_transcription
                    and message.server_content.output_transcription.text
                ):
                    model_transcription_parts.append(message.server_content.output_transcription.text)

                # Collect Audio
                if (
                    message.server_content.model_turn
                    and message.server_content.model_turn.parts
                ):
                    for part in message.server_content.model_turn.parts:
                        # Check specifically for inline_data which contains audio bytes
                        if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data'):
                            logging.debug("Received audio data chunk.")
                            audio_data_list.append(
                                np.frombuffer(part.inline_data.data, dtype=np.int16)
                            )
                
                # Optional: Check for turn completion if needed, though collecting all audio might suffice
                # if message.server_content.turn_complete:
                #    logging.info("Server indicated turn complete.")
                #    break # Exit loop once turn is complete? Or wait for session close?

        # --- Process results ---
        model_response_text = "".join(model_transcription_parts).strip()
        logging.info(f"Received model transcription: {model_response_text}")

        # Update history
        if model_response_text: # Only update history if we got a text response
             history.append(current_turn) # Add user turn
             history.append(Content(role="model", parts=[Part(text=model_response_text)])) # Add model text response
             conversation_histories[session_id] = history # Store updated history
             logging.debug(f"Updated history for {session_id}: {history}")

        # Process and return collected audio (if any)
        if audio_data_list:
            logging.info(f"Collected {len(audio_data_list)} audio chunks.")
            full_audio_np = np.concatenate(audio_data_list)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            sample_rate = 24000 # From notebook example
            num_channels = 1 # Mono
            sample_width = 2 # Bytes per sample for int16

            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(full_audio_np.tobytes())
            
            wav_data = wav_buffer.getvalue()
            logging.info(f"Created WAV data of size: {len(wav_data)} bytes")

            # Encode audio data as base64
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')

            # Return JSON with audio and transcription
            return jsonify({
                'audio_base64': audio_base64,
                'transcription': model_response_text 
            })
        else:
            logging.error("No audio data received from Live API session.")
            return jsonify({'error': 'Live API did not return any audio data.'}), 500
        # --- End Live API Audio ---

    except Exception as e:
        logging.error(f"Error during Live API audio generation: {e}", exc_info=True) # Log traceback
        return jsonify({'error': f'An error occurred during audio generation: {str(e)}'}), 500


@app.route('/web_search_audio', methods=['POST'])
async def web_search_audio():
    data = request.get_json()
    query = data.get('query') # Initial query or follow-up
    follow_up_text = data.get('follow_up_text') # Optional follow-up
    language_code = data.get('language_code', 'en-US') # Default to en-US if not provided

    if not query and not follow_up_text: # Need at least one
        return jsonify({'error': 'Missing query or follow_up_text'}), 400
    
    # Use query as the base topic even for follow-ups if needed for context
    base_query = query if query else data.get('base_query', 'the previous topic') # Need frontend to send base_query on follow-up

    try:
        logging.info(f"Attempting Web Search audio generation for query: {query}")
        
        client = genai.Client(vertexai=True, project=settings.project_id, location=settings.location)

        # Configure Live API for audio output + Google Search tool
        voice_name = "Aoede" # Keep a default voice for web search for now, or make it selectable too
        logging.info(f"Using voice: {voice_name}, language: {language_code} for web search")
        config = LiveConnectConfig(
            response_modalities=["AUDIO"], 
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
                ),
                language_code=language_code # Add language code here
            ),
            tools=[Tool(google_search=GoogleSearch())], # Add Google Search tool
            output_audio_transcription=AudioTranscriptionConfig(), # Request transcription
        )

        # --- Manage Web Search Conversation History ---
        # Use a different key for web search history
        session_id = 'web_search_user1' 
        history = conversation_histories.get(session_id, [])

        # Add system instruction if language is not English
        system_instruction = ""
        if language_code != 'en-US':
            language_name = language_code.split('-')[0].upper()
            lang_map = {"DE": "GERMAN", "ES": "SPANISH", "FR": "FRENCH", "HI": "HINDI", "ID": "INDONESIAN", "IT": "ITALIAN", "JA": "JAPANESE", "KO": "KOREAN", "NL": "DUTCH", "PL": "POLISH", "PT": "PORTUGUESE", "RU": "RUSSIAN", "TH": "THAI", "TR": "TURKISH", "VI": "VIETNAMESE", "CMN": "MANDARIN CHINESE", "AR": "ARABIC", "BN": "BENGALI", "GU": "GUJARATI", "KN": "KANNADA", "ML": "MALAYALAM", "MR": "MARATHI", "TA": "TAMIL", "TE": "TELUGU"}
            language_name = lang_map.get(language_name, language_name)
            system_instruction = f"RESPOND IN {language_name}. YOU MUST RESPOND UNMISTAKABLY IN {language_name}.\n\n"
            logging.info(f"Adding system instruction for language: {language_name}")
        
        if follow_up_text:
            current_user_text = f"{system_instruction}{follow_up_text}"
            logging.info(f"Processing web search follow-up text (lang={language_code}): {current_user_text}")
        else: # Initial web search query
            current_user_text = f"{system_instruction}{query}"
            logging.info(f"Processing initial web search query (lang={language_code}): {current_user_text}")
            history = [] # Start fresh history for a new initial web search

        current_turn = Content(role="user", parts=[Part(text=current_user_text)])
        turns_to_send = history + [current_turn] 
        logging.debug(f"Sending web search turns to Live API: {turns_to_send}")
        # --- End History Management ---


        audio_data_list = []
        model_transcription_parts = [] # To collect text transcription

        async with client.aio.live.connect(model=settings.gemini_live_model, config=config) as session:
            await session.send_client_content(turns=turns_to_send)

            async for message in session.receive():
                 # Collect Transcription
                if (
                    message.server_content.output_transcription
                    and message.server_content.output_transcription.text
                ):
                    model_transcription_parts.append(message.server_content.output_transcription.text)
                
                 # Collect Audio
                if (
                    message.server_content.model_turn
                    and message.server_content.model_turn.parts
                ):
                    for part in message.server_content.model_turn.parts:
                        if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data'):
                            audio_data_list.append(
                                np.frombuffer(part.inline_data.data, dtype=np.int16)
                            )

        # Process and return collected audio
        if audio_data_list:
            logging.info(f"Collected {len(audio_data_list)} audio chunks from web search.")
            full_audio_np = np.concatenate(audio_data_list)
            
            wav_buffer = io.BytesIO()
            sample_rate = 24000 
            num_channels = 1 
            sample_width = 2 

            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(full_audio_np.tobytes())
            
            wav_data = wav_buffer.getvalue()
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')

            # Update history
            model_response_text = "".join(model_transcription_parts).strip()
            logging.info(f"Received web search model transcription: {model_response_text}")
            if model_response_text: 
                 history.append(current_turn) 
                 history.append(Content(role="model", parts=[Part(text=model_response_text)])) 
                 conversation_histories[session_id] = history 
                 logging.debug(f"Updated web search history for {session_id}: {history}")

            # Return JSON with audio and transcription
            return jsonify({
                'audio_base64': audio_base64,
                'transcription': model_response_text 
            })
        else:
            logging.error("No audio data received from Live API web search session.")
            return jsonify({'error': 'Live API did not return any audio data for web search.'}), 500

    except Exception as e:
        logging.error(f"Error during Live API web search audio generation: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred during web search audio generation: {str(e)}'}), 500


@app.route('/generate_image', methods=['POST'])
def generate_image(): # Synchronous route
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400

    try:
        logging.info(f"Attempting Image Generation for prompt: {prompt}")
        
        # Use the standard client initialization for generate_content
        client = genai.Client(vertexai=True, project=settings.project_id, location=settings.location)
        
        # Specify the image generation model (Update config.yaml if needed, or hardcode)
        # Assuming a model like 'gemini-2.0-flash-exp-image-generation' or similar exists
        # Let's use the one from settings for now, assuming it's set correctly
        # image_model_name = settings.image_generation_model_name # Use the one from settings if configured
        # Trying the model name from the user's provided example:
        image_model_name = "gemini-2.0-flash-exp" 
        logging.info(f"Using image generation model: {image_model_name}")

        response = client.models.generate_content(
            model=image_model_name, # Use the specific image gen model
            contents=prompt,
            config=GenerateContentConfig( # Use GenerateContentConfig
              response_modalities=['TEXT', 'IMAGE'] # Request both text and image
            )
        )

        image_base64 = None
        text_response = ""

        # Extract image and text from response
        # Note: response structure might vary slightly, adjust based on actual output
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
              if part.text is not None:
                text_response += part.text
              elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
                    # Convert raw image bytes to base64
                    image_bytes = part.inline_data.data
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    logging.info(f"Generated image (mime_type: {part.inline_data.mime_type}), encoded as base64.")
                    # Assuming only one image is generated
                    break 
        
        if image_base64:
             return jsonify({
                 'image_base64': image_base64,
                 'text': text_response.strip(),
                 'mime_type': part.inline_data.mime_type # Send mime type too
             })
        else:
             logging.error(f"No image found in response for prompt: {prompt}")
             # Log the full response if debugging needed
             # logging.debug(f"Full Image Gen Response: {response}") 
             return jsonify({'error': 'Failed to generate image.', 'text': text_response.strip()}), 500

    except Exception as e:
        logging.error(f"Error during image generation: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred during image generation: {str(e)}'}), 500


# Removed the old text_to_wav function and previous generate_audio implementation

if __name__ == "__main__":
    # Run using Hypercorn for ASGI support
    import hypercorn.asyncio
    import hypercorn.config

    config = hypercorn.config.Config()
    config.bind = [f"127.0.0.1:{settings.port}"] # Changed bind address
    # config.loglevel = "DEBUG" # Uncomment for more detailed server logs
    
    print(f"Starting Hypercorn server on http://{config.bind[0]}...")
    asyncio.run(hypercorn.asyncio.serve(app, config))
