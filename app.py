from flask import Flask, render_template, request, jsonify
from collections import defaultdict
import math
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

webpages = {}
index = defaultdict(set)
tf = defaultdict(lambda: defaultdict(int))
df = defaultdict(int)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Function to build index
def build_index():
    global index, tf, df
    index.clear()
    tf.clear()
    df.clear()
    for page, content in webpages.items():
        words = preprocess(content)
        unique_words = set(words)
        for word in words:
            tf[page][word] += 1
        for word in unique_words:
            df[word] += 1
            index[word].add(page)

# Compute TF-IDF
def compute_tfidf(query, page):
    words = preprocess(query)
    score = 0
    for word in words:
        if word in tf[page]:
            tf_val = tf[page][word]
            idf_val = math.log((len(webpages) + 1) / (1 + df[word])) + 1
            score += tf_val * idf_val
    return score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_page', methods=['POST'])
def add_page():
    data = request.get_json()
    title = data['title']
    content = data['content']
    webpages[title] = content
    build_index()
    return jsonify({'message': 'Page added successfully'})

@app.route('/get_pages', methods=['GET'])
def get_pages():
    page_data = []
    for page, content in webpages.items():
        word_counts = {word: tf[page][word] for word in tf[page]}
        page_data.append({'title': page, 'word_counts': word_counts})
    return jsonify(page_data)

@app.route('/clear_pages', methods=['GET'])
def clear_pages():
    global webpages, index, tf, df
    webpages.clear()
    index.clear()
    tf.clear()
    df.clear()
    return jsonify({'message': 'All pages deleted on refresh'})

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    
    relevant_pages = set()
    for word in preprocess(query):
        relevant_pages.update(index[word])
    
    results = [(page, compute_tfidf(query, page)) for page in relevant_pages]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return jsonify({'steps': ['Preprocessing query', 'Finding relevant pages', 'Calculating TF-IDF'], 'results': results})

if __name__ == '__main__':
    app.run(debug=True)
