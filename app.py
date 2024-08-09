from flask import Flask, request, render_template
import os
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import nltk

stopwords
nltk.download('stopwords')

app = Flask(__name__)

N = 0
corpusroot = './US_Inaugural_Addresses/'
docs = {}

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as doc:
            docs[filename] = doc.read()
            N += 1

# Tokenize and stem the documents
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
doc_tokens = {}
for filename, doc in docs.items():
    tokens = [stemmer.stem(token.lower()) for token in tokenizer.tokenize(doc) if token.lower() not in stop_words]
    doc_tokens[filename] = tokens

idf = {}
for token in set([token for tokens in doc_tokens.values() for token in tokens]):
    df = sum([1 for tokens in doc_tokens.values() if token in tokens])
    idf[token] = math.log10(N / df)

# Function to calculate weights
def calculate_weights(flag):
    doc_weights = {}
    for filename, tokens in doc_tokens.items():
        tf = {}
        for token in tokens:
            if token in tf:
                tf[token] += 1
            else:
                tf[token] = 1

        doc_weights[filename] = {}
        for token, freq in tf.items():
            weight = (1 + math.log10(freq))
            if flag:
                doc_weights[filename][token] = weight * idf[token]
            else:
                doc_weights[filename][token] = weight 
    return doc_weights

# Function to process the query
def query(qstring):
    tokens = [stemmer.stem(token.lower()) for token in tokenizer.tokenize(qstring) if token.lower() not in stop_words]
    
    query_weights = {}
    for token in tokens:
        if token not in query_weights:
            temp = tokens.count(token)
            query_weights[token] = (1 + math.log10(temp)) if temp > 0 else 1

    values = {filename: 0 for filename in docs}
    
    query_tf_idf = {}
    for token, weight in query_weights.items():
        query_idf = idf.get(token, 0)
        query_tf_idf[token] = weight * query_idf

    query_mag = sum([weight * weight for weight in query_tf_idf.values()])
    query_mag = math.sqrt(query_mag)

    if query_mag == 0:
        return None, None, "Query magnitude is zero. Please enter a different query."

    for token in query_tf_idf:
        query_tf_idf[token] = query_tf_idf[token] / query_mag
    
    doc_mag = {}
    doc_weights = calculate_weights(False)
    for doc in doc_weights:
        x = sum([weight * weight for weight in doc_weights[doc].values()])
        doc_mag[doc] = math.sqrt(x)

    for doc, vals in doc_weights.items():
        mag_doc = doc_mag[doc]
        for t, v in vals.items():
            doc_weights[doc][t] = v / mag_doc

    for filename in doc_weights:
        for token in query_tf_idf:
            if token in doc_weights[filename]:
                values[filename] += doc_weights[filename][token] * query_tf_idf[token]

    sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)

    if not sorted_values:
        return None, None, "No results found for your query."

    top_match = sorted_values[0][0]
    top_score = sorted_values[0][1]

    inference = (
        f"The document '{top_match}' is most relevant to your query. "
        f"This relevance score is based on the similarity between your search terms "
        f"and the content of the document, taking into account how often your search terms appear in the text "
        f"and their significance across all documents."
    )

    return top_match, top_score, inference

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query_string = request.form['query']
    if not query_string:
        return render_template('index.html', error="Invalid query. Please enter a valid search term.")
    
    result_doc, score, inference = query(query_string)
    
    if result_doc:
        excerpt = docs[result_doc]
        return render_template('index.html', query=query_string, result={'file_name': result_doc, 'score': score, 'excerpt': excerpt}, inference=inference)
    else:
        return render_template('index.html', error=f"'{query_string}' does not exist in the documents. Please try a different query.")

if __name__ == '__main__':
    app.run(debug=True)
