import streamlit as st
from audiorecorder import audiorecorder

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

import logging

from audio import get_response_audio, autoplay_audio, whisper_api
from constants import openai_api_key, image_and_word_descriptions, filepaths, systemMessage, initial_ai_message, exercise_examples, initial_therapist_message

logging.basicConfig(level=logging.INFO)

# Initialize session state variables if they don't exist
if 'title' not in st.session_state:
    st.session_state.title = "Aphasia Therapist"

def initialize_conversation(model_version="gpt-3.5-turbo-1106", selected_problem=1, exercise_number='Exercise 1'):
    chat = ChatOpenAI(model_name=model_version, temperature=0, openai_api_key=openai_api_key)

    template = f"""{systemMessage}
    {exercise_examples[exercise_number]}
    Current conversation:
    Therapist: {initial_ai_message[exercise_number]}
    {image_and_word_descriptions[exercise_number][selected_problem-1]}
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

def model_query(query):
    return st.session_state.conversation.predict(input=query)

def reset_audiorecorder():
    st.session_state.audio_key = 'audiorecorder_' + str(int(st.session_state.audio_key.split('_')[1]) + 1)

def reset_history(exercise_number='Exercise 1'):
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "therapist", "content": initial_therapist_message[exercise_number]})
    if exercise_number != "Exercise 1":
        st.session_state.chat_history.append({"role": "Initial Prompt", "content": exercise_number})

if 'exercise_number' not in st.session_state:
    st.session_state.exercise_number = 'Exercise 1'

if 'selected_problem' not in st.session_state:
        st.session_state.selected_problem = 1

if 'audio_key' not in st.session_state:
    st.session_state.audio_key = 'audiorecorder_1'

if 'response_filename' not in st.session_state:
    st.session_state.response_filename = None

if "chat_history" not in st.session_state:
    reset_history()
    initialize_conversation()

# Sidebar elements for file uploading and selecting options
with st.sidebar:
    st.title("Settings")

    model_version = st.selectbox(
        "Select the GPT model version:",
        options=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
        index=0  # Default to gpt-3.5-turbo
    )

    exercise_number = st.selectbox(
        "Select the exercise number:",
        options=["Exercise 1", "Exercise 2", "Exercise 3"],
        index=0  # Default to exercise 1
    )
    if exercise_number != st.session_state.exercise_number:
        st.session_state.exercise_number = exercise_number
        reset_history(exercise_number=st.session_state.exercise_number)
        initialize_conversation(model_version=model_version, selected_problem=st.session_state.selected_problem, exercise_number=st.session_state.exercise_number)

    selected_problem = st.selectbox("Please select the problem number:", options=[i+1 for i in range(len(filepaths))])
    if selected_problem != st.session_state.selected_problem:
        st.session_state.selected_problem = selected_problem
        initialize_conversation(model_version=model_version, selected_problem=st.session_state.selected_problem, exercise_number=st.session_state.exercise_number)

    # Add reset button in sidebar
    if st.button('Reset Chat'):
        reset_history(exercise_number=st.session_state.exercise_number)
        initialize_conversation(model_version=model_version, selected_problem=st.session_state.selected_problem, exercise_number=st.session_state.exercise_number)
        st.rerun()

# Main chat container
chat_container = st.container()
user_input = None

# Handle chat input and display
with chat_container:
    if st.session_state.exercise_number == "Exercise 1":
        st.image(filepaths[selected_problem-1], caption="Please describe what you see in this image.", width=600)

    audio = audiorecorder("Click to record", "Click to stop recording", key=st.session_state.audio_key)

    if len(audio) > 0:
        filename = "userRecording" + str(len(st.session_state.chat_history)) + ".wav"
        audio.export(filename, format="wav")
        user_input = whisper_api(filename)

    for message in st.session_state.chat_history:
        if message["role"] == "Patient":
            st.markdown(f"> **User**: {message['content']}")
        elif message["role"] == "therapist":
            st.markdown(f"> **Therapist**: {message['content']}")
        elif message["role"] == "Initial Prompt":
            st.markdown(f"<h1 style='text-align: center; color: black;'>{image_and_word_descriptions[exercise_number][selected_problem-1]}</h1>", unsafe_allow_html=True)
        elif message["role"] == "context":
            with st.expander("Click to see the recordings"):
                st.audio(message['content'])
                if st.session_state.response_filename:
                    autoplay_audio(st.session_state.response_filename)
    
# Handle chat input and display
if user_input:
    st.session_state.chat_history.append({"role": "Patient", "content": user_input})
    model_response = model_query(user_input)
    st.session_state.response_filename = get_response_audio(model_response, str(len(st.session_state.chat_history)))
    st.session_state.chat_history.append({"role": "therapist", "content": model_response})
    #context = None # implement context as audio recording
    if len(audio) > 0:
        st.session_state.chat_history.append({"role": "context", "content": filename})
    reset_audiorecorder()

    # Use st.rerun() to update the display immediately after sending the message
    st.rerun()