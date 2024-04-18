from flask import Flask, request, jsonify
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask_cors import CORS

app = Flask(__name__)

# Configure CORS options
cors = CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # You can specify domains, e.g., "http://localhost:3000"
        "methods": ["GET", "POST", "DELETE", "PUT"],  # Allowed methods
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

def load_data(directory):
    questions = []
    answers = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".jsonl"):
                with open(os.path.join(directory, filename), 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        questions.append(data['question'])
                        answers.append(data['answer'])
        return questions, answers
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

class QuestionSearcher:
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.vectorizer = TfidfVectorizer()
        try:
            self.question_vectors = self.vectorizer.fit_transform(self.questions)
        except Exception as e:
            print(f"Error fitting vectorizer: {e}")
            self.question_vectors = None

    def find_most_similar(self, query):
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.question_vectors)
            max_index = similarities.argmax()
            return self.answers[max_index]
        except Exception as e:
            print(f"Error finding similar question: {e}")
            return "An error occurred while finding a similar question."

# Load data and prepare searcher
questions, answers = load_data('bank')
searcher = QuestionSearcher(questions, answers)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        question = data['question']
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        answer = searcher.find_most_similar(question)
        return jsonify({'question': question, 'answer': answer})
    except KeyError:
        return jsonify({'error': 'Invalid data format, question key missing'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
