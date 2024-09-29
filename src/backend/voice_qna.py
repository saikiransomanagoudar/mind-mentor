import boto3
import json
import os
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify, send_file
import speech_recognition as sr
import tempfile
from boto3 import Session

app = Flask(__name__)

# Initialize AWS clients
session = Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN'),
    region_name='us-west-2'
)

s3 = session.client('s3')
bedrock_runtime = session.client('bedrock-runtime')
polly = session.client('polly')

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
    # Ensure that the triple-quoted string is properly closed
    prompt = f"""Human: Given the following context (if any), please answer the question. If there's no context, treat it as a general conversation.

Context: {context}

Question: {question}

Assistant:"""  # The triple-quoted string ends here.

    body = {
        "inputText": prompt,
        "maxTokens": 500,
        "temperature": 0.7,
        "top_p": 1
    }

    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId="amazon.titan-tg1-large",  # Ensure this model ID is available
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        return response_body.get('outputText', 'No response')
    except ClientError as e:
        return f"Error querying Bedrock: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=5050)
