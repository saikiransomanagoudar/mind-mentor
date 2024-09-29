import os
import json
import boto3
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import docx

app = Flask(__name__)

# Configure the Bedrock client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-west-2',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == '.txt':
        with open(file_path, 'r') as file:
            return file.read()
    elif extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in pdf_reader.pages])
    elif extension == '.docx':
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])

def generate_summary(text):
    prompt = f"Please summarize the following text in a concise manner:\n\n{text}\n\nSummary:"
    
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
        "top_p": 1,
    })

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="anthropic.claude-v2",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    return response_body.get('completion', '').strip()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            text = read_file(file_path)
            summary = generate_summary(text)
            os.remove(file_path)  # Remove the file after processing
            return jsonify({'summary': summary}), 200
        except Exception as e:
            os.remove(file_path)  # Remove the file if an error occurs
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
