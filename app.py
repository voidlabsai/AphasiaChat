import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import glob
import pickle
from pathlib import Path
import hashlib
import tempfile
import logging
from audio_transcription_app import transcribe_audio, text_to_embedding, compare_embeddings, whisper_api
import openai
from audiorecorder import audiorecorder

logging.basicConfig(level=logging.INFO)


# Initialize session state variables if they don't exist
if 'title' not in st.session_state:
    st.session_state.title = "Aphasia Therapist"  # Default title

# Define necessary embedding model, LLM, and vectorstore
# openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = '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'
OPENAI_API_KEY = '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'
text_key = "text"

def hash_content(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

@st.cache_data
def process_audio(buffers=None):
    example_audio_path = Path("example-audio")
    filenames = list(example_audio_path.glob("*.mp3")) + list(example_audio_path.glob("*.wav"))
    
    audio_data = {}
    for filename in filenames:
        audio_data[filename.name] = {}
        audio_data[filename.name]['text'] = whisper_api(str(filename))
        audio_data[filename.name]['embedding'] = text_to_embedding(audio_data[filename.name]['text'])

    buffer_names = []

    if buffers:
        # Create temporary files for buffers and load them
        for buffer in buffers:
            filetype = buffer.name.split('.')[-1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + filetype)
            buffer_names.append(buffer.name)
            with open(temp_file.name, 'wb') as f:
                f.write(buffer.getbuffer())  # Write buffer contents to a temporary file

            buffer_text = whisper_api(temp_file.name)
            buffer_embedding = text_to_embedding(buffer_text)

            audio_data[buffer.name] = {}
            audio_data[buffer.name]['text'] = buffer_text
            audio_data[buffer.name]['embedding'] = buffer_embedding
            
            temp_file.close()

    # Clean up temporary files
    for name in buffer_names:
        Path(name).unlink(missing_ok=True)

    return audio_data

audio_data = process_audio()

def initialize_conversation():
    chat = ChatOpenAI(model_name=model_version, temperature=0, openai_api_key=OPENAI_API_KEY)
    initial_ai_message = "I'd like you to look at an image and describe what you see. Here's the image:"

    template = f"""The following is a friendly conversation between a human and a speech therapist specializing in aphasia. The therapist is supportive and follows best practices from speech language therapy. The patient may be hard to understand, but the therapist tries their best and asks for clarification if the text is unclear. The therapist is not perfect, and sometimes it says things that are inconsistent with what it has said before.

    Current conversation:
    Therapist: {initial_ai_message}
    [Image: womanBakingCake.jpg]
    {{history}}
    Patient: {{input}}
    Therapist:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )
    st.session_state.conversation = ConversationChain(
        prompt=PROMPT,
        llm=chat, 
        verbose=False, 
        memory=ConversationBufferMemory(human_prefix="Patient")
    )


# Read the files from the directory using pathlib
directory = Path("./example-audio")
files = []
if directory.exists():
    for file in directory.iterdir():
        if file.suffix in [".mp3", ".wav"]:
            files.append(str(file))

def model_query(query):
    """
    Args:
        query (str): audio file name of query
    """
    # text = audio_data[query]['text']

    ai_message = st.session_state.conversation.predict(input=query)
    return ai_message
    
# Sidebar elements for file uploading and selecting options
with st.sidebar:
    st.title("Settings")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if uploaded_files:
        # Reload the vectorstore with new files including uploaded ones
        audio_data = process_audio(buffers=uploaded_files)  

        uploaded_file_names = [uploaded_file.name for uploaded_file in uploaded_files]
        files.extend(uploaded_file_names)

    # Add a way to select which files to use for the model query
    selected_files = st.multiselect("Please select the files to query:", options=files)

    model_version = st.selectbox(
        "Select the GPT model version:",
        options=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
        index=0  # Default to gpt-3.5-turbo
    )

    # Add reset button in sidebar
    if st.button('Reset Chat'):
        st.session_state.chat_history = []
        st.session_state.context_hashes = set()
        initialize_conversation()
        st.rerun()

if "context_hashes" not in st.session_state:
    st.session_state.context_hashes = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    initialize_conversation()

# Main chat container
chat_container = st.container()
user_input = None

# Handle chat input and display
with chat_container:
    st.image("woman-baking-cake.jpg", caption="Please describe what is happening in this image.", width=600)
    audio = audiorecorder("Click to record", "Click to stop recording") # st.chat_input("Say something", key="user_input")

    if len(audio) > 0:
        print("Recording saved to userRecording.wav")
        filename = "userRecording" + str(len(st.session_state.chat_history)) + ".wav"
        audio.export(filename, format="wav")
        user_input = whisper_api(filename)
        print(user_input)
        audio = None

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"> **User**: {message['content']}")
        elif message["role"] == "model":
            st.markdown(f"> **Model**: {message['content']}")
        elif message["role"] == "context":
            with st.expander("Click to see the context"):
                for doc in message["content"]:
                    st.markdown(f"> **Recording**: {doc.metadata['source']}")


# Handle chat input and display
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    model_response = model_query(user_input)
    st.session_state.chat_history.append({"role": "model", "content": model_response})
    context = None # implement context as audio recording
    if context:
        st.session_state.chat_history.append({"role": "context", "content": context})

    # Use st.rerun() to update the display immediately after sending the message
    st.rerun()