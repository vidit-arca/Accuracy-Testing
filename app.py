import streamlit as st
from transformers import pipeline
from jiwer import wer
from rouge_score import rouge_scorer
import numpy as np
import sounddevice as sd
import speech_recognition as sr

# Define functions
def real_time_audio_capture():
    """
    Capture audio in real-time using the microphone via sounddevice
    and transcribe it using speech recognition.
    """
    recognizer = sr.Recognizer()

    try:
        st.write("Listening... Speak into the microphone (Press Ctrl+C to stop).")
        audio_data = []

        # Callback function to collect microphone data
        def callback(indata, frames, time, status):
            if status:
                st.write(f"Status: {status}")
            audio_data.extend(indata[:, 0])  # Capture mono-channel data

        st.write("Recording... Press Ctrl+C to stop.")
        with sd.InputStream(callback=callback, samplerate=16000, channels=1):
            while True:
                pass

    except KeyboardInterrupt:
        st.write("Stopped recording.")
        audio_bytes = (np.array(audio_data) * 32768).astype(np.int16).tobytes()
        audio = sr.AudioData(audio_bytes, 16000, 2)

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
    except Exception as e:
        st.write(f"An error occurred during real-time audio capture: {e}")
        return ""

def generate_conversation():
    """
    Generate a synthetic medical conversation using a text-generation model.
    """
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
            "Doctor: Is there any family history of chronic illnesses?\n"
            "Patient: Yes, my father has diabetes.\n"
            "Doctor: Do you have any history of addiction?\n"
            "Patient: No, I don't.\n"
            "Doctor: How about your diet?\n"
            "Patient: I try to eat a balanced diet, but I sometimes skip meals due to a busy schedule.\n"
            "Doctor: How often do you engage in physical activity?\n"
            "Patient: I try to exercise at least three times a week.\n"
            "Doctor: How has your stress level been recently?\n"
            "Patient: It's been quite high due to work pressure.\n"
            "Doctor: How well are you sleeping?\n"
            "Patient: I struggle with sleep and often wake up feeling tired.\n"
            "Doctor: Are you currently on any medication?\n"
            "Patient: Yes, I'm taking medication for hypertension.\n"
            "Doctor: "
        )

        response = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.7, top_p=0.9)
        conversation = response[0]['generated_text']
        return conversation
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
        scores = scorer.score(reference_summary, generated_summary)
        return scores
    except Exception as e:
        st.write(f"Error during summary evaluation: {e}")
        return {}

def process_system(with_microphone=False):
    if with_microphone:
        st.write("Real-Time Audio Workflow")
        system_transcript = real_time_audio_capture()
        if not system_transcript:
            return {"error": "No audio transcription was captured from the microphone."}
    else:
        st.write("Synthetic Audio Workflow")
        conversation = generate_conversation()
        if conversation.startswith("Error"):
            return {"error": "Conversation generation failed."}
        system_transcript = conversation

    reference_summary = "Patient exhibits early signs of diabetes."
    generated_summary = "The patient shows symptoms of fatigue, back pain, and difficulty sleeping."

    transcription_accuracy = calculate_transcription_accuracy(conversation, system_transcript)
    summary_scores = evaluate_summary(reference_summary, generated_summary)

    return {
        "reference_transcript": conversation,
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
            st.write("**Reference Transcript:**")
            st.write(results["reference_transcript"])
            st.write("**System Transcript:**")
            st.write(results["system_transcript"])

            st.subheader("Transcription Accuracy")
            st.write(f"Accuracy: {results['transcription_accuracy']:.2f}%")

            st.subheader("Summaries")
            st.write("**Reference Summary:**")
            st.write(results["reference_summary"])
            st.write("**Generated Summary:**")
            st.write(results["generated_summary"])

            st.subheader("ROUGE Scores")
            for key, score in results["summary_scores"].items():
                st.write(f"{key} - Precision: {score.precision:.2%}, Recall: {score.recall:.2%}, F1: {score.fmeasure:.2%}")

if __name__ == "__main__":
    main()
