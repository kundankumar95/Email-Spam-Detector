
from flask import Flask, request
from flask_cors import CORS  # Import CORS
from main import detect_spam

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/email', methods=['GET'])
def email_spam_detector():
    message = request.args.get('message')

    if message is None:
        return "No message provided", 400

    result = detect_spam(message)
    return f"{result}"

if __name__ == '__main__':
    app.run(debug=True)
