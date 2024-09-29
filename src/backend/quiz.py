from flask import Flask, request, jsonify
import boto3
import json
import os
from botocore.exceptions import ClientError
import random

app = Flask(__name__)

# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
)

def generate_quiz(document_content, num_questions=5):
    prompt = f"""
    Human: Given the following document content, generate a multiple-choice quiz with {num_questions} questions. 
    For each question, provide 4 options (A, B, C, D) with one correct answer.
    Format the output as a JSON array of objects, where each object represents a question and has the following structure:
    {{
        "question": "The question text",
        "options": {{
            "A": "Option A text",
            "B": "Option B text",
            "C": "Option C text",
            "D": "Option D text"
        }},
        "correct_answer": "The correct option letter (A, B, C, or D)"
    }}

    Document content:
    {document_content}

    Assistant: Here's a multiple-choice quiz based on the provided document content:

    [
        {{
            "question": "What is the capital of France?",
            "options": {{
                "A": "London",
                "B": "Berlin",
                "C": "Paris",
                "D": "Madrid"
            }},
            "correct_answer": "C"
        }},
        {{
            "question": "Who painted the Mona Lisa?",
            "options": {{
                "A": "Vincent van Gogh",
                "B": "Leonardo da Vinci",
                "C": "Pablo Picasso",
                "D": "Michelangelo"
            }},
            "correct_answer": "B"
        }}
    ]

    Human: Great! Now, based on the document I provided earlier, please generate a quiz with {num_questions} questions following the same JSON format.

    Assistant: Certainly! I'll generate a quiz with {num_questions} questions based on the document content you provided. Here's the quiz in the requested JSON format:

    """

    try:
        response = bedrock.invoke_model(
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 2000,
                "temperature": 0.7,
                "top_p": 0.9,
            }),
            modelId="anthropic.claude-v2",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        quiz_json = json.loads(response_body['completion'])
        return quiz_json
    except ClientError as e:
        print(f"Error calling Bedrock: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        content = file.read().decode('utf-8')
        return jsonify({"message": "File uploaded successfully", "content": content}), 200

@app.route('/generate_quiz', methods=['POST'])
def create_quiz():
    data = request.json
    document_content = data.get('document_content')
    num_questions = data.get('num_questions', 5)
    
    if not document_content:
        return jsonify({"error": "No document content provided"}), 400
    
    quiz = generate_quiz(document_content, num_questions)
    if quiz:
        return jsonify({"quiz": quiz}), 200
    else:
        return jsonify({"error": "Failed to generate quiz"}), 500

@app.route('/take_quiz', methods=['POST'])
def take_quiz():
    data = request.json
    quiz = data.get('quiz')
    user_answers = data.get('answers')
    
    if not quiz or not user_answers:
        return jsonify({"error": "Quiz or answers not provided"}), 400
    
    score = 0
    total_questions = len(quiz)
    
    for i, question in enumerate(quiz):
        if user_answers.get(str(i)) == question['correct_answer']:
            score += 1
    
    return jsonify({
        "score": score,
        "total_questions": total_questions,
        "percentage": (score / total_questions) * 100
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
