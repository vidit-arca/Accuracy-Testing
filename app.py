import streamlit as st
from transformers import pipeline
from jiwer import wer
from rouge_score import rouge_scorer
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import scipy.io.wavfile as wav

def real_time_audio_capture():
    recognizer = sr.Recognizer()
    
    # Audio recording parameters
    sample_rate = 44100  # Standard CD-quality audio
    duration = 30  # Record for 5 seconds

    st.write("Listening... Speak into the microphone.")
    
    # Record audio
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Save as WAV file
    wav.write("temp_audio.wav", sample_rate, audio_data)

    # Recognize speech from the recorded file
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)

    try:
        transcript = recognizer.recognize_google(audio)
        st.write("Transcribed Text:")
        st.write(transcript)
        return transcript
    except sr.UnknownValueError:
        st.write("Sorry, could not understand the audio.")
        return ""
    except sr.RequestError as e:
        st.write(f"Error connecting to Google Speech Recognition service: {e}")
        return ""

def generate_conversation():
    try:
        generator = pipeline("text-generation", model="distilgpt2")

        prompt = (
            "Doctor: What brings you in today?\n"
            "Patient: I have been experiencing back pain and fatigue.\n"
            "Doctor: How long have you been dealing with these issues?\n"
            "Patient: It's been a few weeks now.\n"
            "Doctor: Have you had any similar issues in the past?\n"
            "Patient: No, this is the first time.\n"
            "Doctor: Any significant medical history we should know about?\n"
            "Patient: I have a history of hypertension.\n"
            "Doctor: Have you had any surgeries in the past?\n"
            "Patient: Yes, I had an appendectomy five years ago.\n"
        )

        response = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.7, top_p=0.9)
        return response[0]['generated_text']
    except Exception as e:
        st.write(f"Error during conversation generation: {e}")
        return "Error generating conversation."

def calculate_transcription_accuracy(reference_transcript, system_transcript):
    try:
        error_rate = wer(reference_transcript, system_transcript)
        accuracy = max(0, 1 - error_rate)
        return accuracy * 100
    except Exception as e:
        st.write(f"Error calculating transcription accuracy: {e}")
        return 0.0

def evaluate_summary(reference_summary, generated_summary):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        return scorer.score(reference_summary, generated_summary)
    except Exception as e:
        st.write(f"Error during summary evaluation: {e}")
        return {}
    
def process_system(with_microphone=False):
    conversation = ""  # Initialize conversation to avoid UnboundLocalError

    if with_microphone:
        st.write("Real-Time Audio Workflow")
        system_transcript = real_time_audio_capture()
        if not system_transcript:
            return {"error": "No audio transcription was captured from the microphone."}

        # Use the same transcript as reference (since there's no predefined reference)
        reference_transcript = system_transcript  

    else:
        st.write("Synthetic Audio Workflow")
        conversation = generate_conversation()
        if conversation.startswith("Error"):
            return {"error": "Conversation generation failed."}
        
        system_transcript = conversation  # Assign conversation to system_transcript
        reference_transcript = conversation  # Use generated text as reference

    reference_summary = "Patient exhibits early signs of diabetes."
    generated_summary = "The patient shows symptoms of fatigue, back pain, and difficulty sleeping."

    transcription_accuracy = calculate_transcription_accuracy(reference_transcript, system_transcript)
    summary_scores = evaluate_summary(reference_summary, generated_summary)

    return {
        "reference_transcript": reference_transcript,  # Now properly assigned
        "system_transcript": system_transcript,
        "reference_summary": reference_summary,
        "generated_summary": generated_summary,
        "transcription_accuracy": transcription_accuracy,
        "summary_scores": summary_scores
    }

def main():
    st.title("Speech-to-Text Accuracy and ROUGE Evaluation")

    with_microphone = st.radio("Use real-time microphone input?", ('Yes', 'No')) == 'Yes'

    if st.button("Run Workflow"):
        results = process_system(with_microphone)

        if "error" in results:
            st.error(results["error"])
        else:
            st.subheader("Transcripts")
            st.write("**Reference Transcript:**", results["reference_transcript"])
            st.write("**System Transcript:**", results["system_transcript"])

            st.subheader("Transcription Accuracy")
            st.write(f"Accuracy: {results['transcription_accuracy']:.2f}%")

            st.subheader("Summaries")
            st.write("**Reference Summary:**", results["reference_summary"])
            st.write("**Generated Summary:**", results["generated_summary"])

            st.subheader("ROUGE Scores")
            for key, score in results["summary_scores"].items():
                st.write(f"{key} - Precision: {score.precision:.2%}, Recall: {score.recall:.2%}, F1: {score.fmeasure:.2%}")

if __name__ == "__main__":
    main()