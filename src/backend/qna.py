import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)
CORS(app)

# Configure the Bedrock client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-west-2',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')
)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['flashcards_db']
flashcards_collection = db['flashcards']

def save_flashcards(topic, flashcards):
    for card in flashcards:
        flashcards_collection.insert_one({
            'topic': topic,
            'question': card['question'],
            'answer': card['answer']
        })

def generate_flashcards(prompt):
    """Invoke the Bedrock model to generate flashcards based on the prompt."""
    body = {
        "prompt": f"Generate 5 flashcards in Q&A format about the following topic: {prompt}. Format the output as a JSON array of objects, each with 'question' and 'answer' keys.",
        "max_tokens_to_sample": 1000,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    response = bedrock_runtime.invoke_model(
        body=json.dumps(body),
        modelId="anthropic.claude-v2",
        accept="application/json",
        contentType="application/json"
    )

    # Parse the response from Bedrock
    response_body = json.loads(response['body'].read())
    flashcards = json.loads(response_body.get('completion', '[]'))
    
    # Save flashcards to the database
    save_flashcards(prompt, flashcards)
    
    return flashcards

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint to generate Q&A flashcards."""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']
    try:
        flashcards = generate_flashcards(prompt)
        return jsonify({'flashcards': flashcards}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/flashcards', methods=['GET'])
def get_flashcards():
    """API endpoint to retrieve saved flashcards."""
    topic = request.args.get('topic')
    
    flashcards = list(flashcards_collection.find(query))
    
    # Convert ObjectId to string for JSON serialization
    for card in flashcards:
        card['_id'] = str(card['_id'])
    
    return jsonify({'flashcards': flashcards}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
