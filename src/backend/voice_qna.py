import boto3
import json
import os
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify
import speech_recognition as sr
import tempfile

app = Flask(__name__)

# Initialize AWS clients
s3 = boto3.client('s3')
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
)

BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

def query_bedrock(question, context=""):
    prompt = f"""Human: Given the following context (if any), please answer the question. If there's no context, treat it as a general conversation.

Context: {context}

Question: {question}
