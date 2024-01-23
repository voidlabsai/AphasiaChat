import streamlit as st
import whisper
import openai
import numpy as np
from pathlib import Path
import os
from openai import OpenAI

# openai.api_key = None  # Add your API key here for local testing
openai.api_key = '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'

def transcribe_audio(audio_path):
    if isinstance(audio_path, str):
        audio_path = Path(audio_path)
        audio_path = os.path.abspath(audio_path)
    print(audio_path)
    # Check if the file exists
    exists = os.path.exists(audio_path)
    print(f"File Exists: {exists}")

    # Check read permission
    readable = os.access(audio_path, os.R_OK)
    print(f"File Readable: {readable}")

    # Check write permission
    writable = os.access(audio_path, os.W_OK)
    print(f"File Writable: {writable}")
    model = whisper.load_model("small")  # Use the latest model version
    audio = whisper.load_audio(audio_path)
    result = model.transcribe(audio)
    print('Transcribed audio: ' + result['text'])
    return result['text']

def whisper_api(audio_path):
    client = OpenAI(api_key='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')

    audio_file= open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text"
    )
    return transcript

def text_to_embedding(text):
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def compare_embeddings(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def main():
    st.title("Describe the picture below in an audio file")

    # Audio input
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if audio_file is not None:
        # Save the audio file and process it
        with open(audio_file.name, "wb") as f:
            f.write(audio_file.getbuffer())
        audio_path = Path(audio_file.name)

        if st.button("Process Audio"):
            transcribed_text = transcribe_audio(audio_path)
            st.write("Transcribed Text:", transcribed_text)

            # Comparison with the expected label
            expected_label = "Someone baking a cake"
            transcribed_embedding = text_to_embedding(transcribed_text)
            label_embedding = text_to_embedding(expected_label)
            similarity = compare_embeddings(transcribed_embedding, label_embedding)
            threshold = 0.75

            st.write("Expected label:", expected_label)
            st.write("Similarity:", similarity)
            if similarity > threshold:
                st.success("Semantically similar.")
            else:
                st.error("Not semantically similar.")

    # Display image
    st.image("woman-baking-cake.jpg")

if __name__ == "__main__":
    main()
