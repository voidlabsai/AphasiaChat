import base64
import requests
import streamlit as st
from openai import OpenAI
from constants import openai_api_key

CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh" # Dorothy

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": st.secrets["XI_API_KEY"]
}

def get_response_audio(text, conversation_number):
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    with open('output' + conversation_number + '.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    return 'output' + conversation_number + '.mp3'

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


def whisper_api(audio_path):
    client = OpenAI(api_key=openai_api_key)

    audio_file= open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text",
    prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Ba... ball. Re... red. Cat... sleep. House... outside. So... sofa. Ap... apples. Bike... ride."
    )
    return transcript