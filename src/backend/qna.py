import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS here
import boto3

app = Flask(__name__)
CORS(app)  # Enable CORS for the app, allowing all domains on all routes

# Configure the Bedrock client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-west-2',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')
)

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
    return json.loads(response_body.get('completion', '[]'))

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

if __name__ == '__main__':
    app.run(debug=True, port=5044)
