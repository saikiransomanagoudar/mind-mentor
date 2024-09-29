from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    button_clicked = data.get('button')

    if button_clicked == 'mind_map':
        script = 'mind_map.py'
    elif button_clicked == 'qna':
        script = 'qna.py'
    elif button_clicked == 'voice_qna':
        script = 'voice_qna.py'
    elif button_clicked == 'quiz':
        script = 'quiz.py'
    else:
        return jsonify({'error': 'Invalid button'}), 400

    try:
        result = subprocess.run(['python', script], capture_output=True, text=True, check=True)
        return jsonify({'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e), 'output': e.output}), 500

if __name__ == '__main__':
    app.run(debug=True)
