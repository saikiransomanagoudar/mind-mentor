from flask import Flask, request, jsonify
import boto3
import json
import os
import uuid
from botocore.exceptions import ClientError
from PIL import Image
import io
import base64
import subprocess

app = Flask(__name__)

# Initialize AWS clients
s3 = boto3.client('s3')
textract = boto3.client('textract')
bedrock_runtime = boto3.client('bedrock-runtime')
polly = boto3.client('polly')

BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')

def extract_text_from_document(document_key):
    try:
        response = textract.detect_document_text(
            Document={'S3Object': {'Bucket': BUCKET_NAME, 'Name': document_key}}
        )
        return ' '.join([item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])
    except ClientError as e:
        print(f"Error extracting text: {e}")
        return None

def generate_script(document_text):
    prompt = f"Create a video tutorial script based on the following document content:\n\n{document_text}\n\nScript:"
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 2000,
                "temperature": 0.7,
                "top_p": 0.95,
            })
        )
        
        return json.loads(response['body'].read())['completion']
    except ClientError as e:
        print(f"Error generating script: {e}")
        return None

def text_to_speech(script):
    try:
        response = polly.synthesize_speech(
            Text=script,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        
        return response['AudioStream'].read()
    except ClientError as e:
        print(f"Error converting text to speech: {e}")
        return None

def generate_image(prompt):
    try:
        response = bedrock_runtime.invoke_model(
            modelId="stability.stable-diffusion-xl",
            body=json.dumps({
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 10,
                "steps": 50,
                "seed": 42
            })
        )
        
        image_data = json.loads(response['body'].read())['artifacts'][0]['base64']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return image
    except ClientError as e:
        print(f"Error generating image: {e}")
        return None

def create_video(audio_file, image_files, output_file):
    image_inputs = ' '.join([f"-loop 1 -t 5 -i {img}" for img in image_files])
    filter_complex = ''.join([f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fade=t=in:st={i*5}:d=1,fade=t=out:st={i*5+4}:d=1[v{i}];" for i in range(len(image_files))])
    filter_complex += ''.join([f"[v{i}]" for i in range(len(image_files))]) + f"concat=n={len(image_files)}:v=1:a=0,format=yuv420p[v]"
    
    cmd = f"ffmpeg -y {image_inputs} -i {audio_file} -filter_complex \"{filter_complex}\" -map \"[v]\" -map {len(image_files)}:a -c:v libx264 -c:a aac -b:a 192k -shortest {output_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        
        try:
            s3.upload_file(file_path, BUCKET_NAME, filename)
            os.remove(file_path)
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return jsonify({'error': 'Error uploading file'}), 500

@app.route('/generate_tutorial', methods=['POST'])
def generate_tutorial():
    data = request.json
    document_key = data.get('document_key')
    
    if not document_key:
        return jsonify({'error': 'No document key provided'}), 400
    
    # Extract text from the document
    document_text = extract_text_from_document(document_key)
    if not document_text:
        return jsonify({'error': 'Failed to extract text from document'}), 500
    
    # Generate script
    script = generate_script(document_text)
    if not script:
        return jsonify({'error': 'Failed to generate script'}), 500
    
    # Convert script to speech
    audio_data = text_to_speech(script)
    if not audio_data:
        return jsonify({'error': 'Failed to convert text to speech'}), 500
    
    # Save audio file
    audio_file = f'/tmp/{uuid.uuid4()}.mp3'
    with open(audio_file, 'wb') as f:
        f.write(audio_data)
    
    # Generate images
    image_prompts = [
        "Educational video background with abstract shapes",
        "Illustration of a person explaining concepts",
        "Visual representation of key points from the tutorial"
    ]
    image_files = []
    for i, prompt in enumerate(image_prompts):
        image = generate_image(prompt)
        if image:
            image_file = f'/tmp/temp_image_{i}.png'
            image.save(image_file)
            image_files.append(image_file)
    
    if not image_files:
        return jsonify({'error': 'Failed to generate images'}), 500
    
    # Create video
    output_file = f'/tmp/{uuid.uuid4()}.mp4'
    if not create_video(audio_file, image_files, output_file):
        return jsonify({'error': 'Failed to create video'}), 500
    
    # Upload video to S3
    video_key = f'tutorials/{os.path.basename(output_file)}'
    try:
        s3.upload_file(output_file, BUCKET_NAME, video_key)
    except ClientError as e:
        print(f"Error uploading video: {e}")
        return jsonify({'error': 'Failed to upload video'}), 500
    
    # Clean up temporary files
    os.remove(audio_file)
    for image_file in image_files:
        os.remove(image_file)
    os.remove(output_file)
    
    return jsonify({'message': 'Video tutorial generated successfully', 'video_key': video_key}), 200

if __name__ == '__main__':
    app.run(debug=True)
