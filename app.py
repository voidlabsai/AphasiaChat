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

# CSS for chat messages
st.markdown("""
<style>
.chat-message {
    padding: 10px;
    border-radius: 20px;
    margin: 5px 0;
    width: fit-content;
    max-width: 80%;
}

.patient-message {
    background-color: #39ff5a;  /* Green background for patient */
    margin-left: auto;  /* Right aligns the message */
    text-align: left;
}

.therapist-message {
    background-color: #d8d8d8;  /* Gray background for therapist */
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS for sticky footer (not working yet)
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;  # You can change the background color
    text-align: center;
    padding: 10px 0;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)


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

# Footer with audio recorder (using an emoji as a microphone icon)
footer_container = st.empty()  # Use an empty container to insert elements
with footer_container.container():
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    audio = audiorecorder("🎤", "Click to stop recording", key=st.session_state.audio_key)
    st.markdown('</div>', unsafe_allow_html=True)


# Handle chat input and display
with chat_container:
    if st.session_state.exercise_number == "Exercise 1":
        st.image(filepaths[selected_problem-1], use_column_width="always")

    for message in st.session_state.chat_history:
        if message["role"] == "Patient":
            st.markdown(f'<div class="chat-message patient-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "therapist":
            st.markdown(f'<div class="chat-message therapist-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "Initial Prompt":
            st.markdown(f"<h1 style='text-align: center; color: black;'>{image_and_word_descriptions[exercise_number][selected_problem-1]}</h1>", unsafe_allow_html=True)
        elif message["role"] == "context":
            with st.expander("Click to see the recordings"):
                st.audio(message['content'])
                if st.session_state.response_filename:
                    autoplay_audio(st.session_state.response_filename)

    if len(audio) > 0:
        filename = "userRecording" + str(len(st.session_state.chat_history)) + ".wav"
        audio.export(filename, format="wav")
        user_input = whisper_api(filename)

# Handle chat input and display
if user_input:
    model_response = model_query(user_input)
    st.session_state.response_filename = get_response_audio(model_response, str(len(st.session_state.chat_history)))

    st.session_state.chat_history.append({"role": "Patient", "content": user_input})
    st.session_state.chat_history.append({"role": "therapist", "content": model_response})
    st.session_state.chat_history.append({"role": "context", "content": filename})

    reset_audiorecorder()
    # Use st.rerun() to update the display immediately after sending the message
    st.rerun()