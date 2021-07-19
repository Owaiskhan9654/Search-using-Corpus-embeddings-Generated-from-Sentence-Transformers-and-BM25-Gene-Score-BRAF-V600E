from flask import Flask, render_template, request
from ranking import rank
import nltk
#nltk.download('punkt')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search/results', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["input"]
    articlelink,score,query,Title1 = rank(search_term)
    return render_template('results.html', res=articlelink ,score=score,title=Title1,query=query)

if __name__ == '__main__':
    app.run(debug=True)
