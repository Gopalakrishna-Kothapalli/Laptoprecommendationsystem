from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
from utils import category_a_laptops, category_b_laptops, category_c_laptops
encoder_decoder = joblib.load('encoder_dictionary.joblib')
lr_model = joblib.load('LogisticRegression.joblib')
dt_model = joblib.load('DecitionTree.joblib')
rf_model = joblib.load('RandomForestClassifier.joblib')

laptop_mapper = {
    0: category_c_laptops,
    1: category_b_laptops,
    2: category_a_laptops,
    3: category_a_laptops
}


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/suggest')
def suggest():
    return render_template('suggest.html')

@app.route('/student_suggest', methods=['GET','POST'])
def student_suggest():
    if request.method == 'POST':
        data_list = []
        data = request.form
        for list_type in data.keys():
            data_list.append(int(data.getlist(list_type)[0]))
        data_list = np.array(data_list)
        predict = lr_model.predict([data_list])
        laptop_suggestions = laptop_mapper[predict[0]]
        return render_template('results.html', laptops=laptop_suggestions)
    return render_template('student_suggest.html')


@app.route('/teacher_suggest', methods=["GET", "POST"])
def teacher_suggest():
    if request.method == 'POST':
        data_list = []
        data = request.form
        for list_type in data.keys():
            data_list.append(int(data.getlist(list_type)[0]))
        data_list = np.array(data_list)
        predict = lr_model.predict([data_list])
        laptop_suggestions = laptop_mapper[predict[0]]
        return render_template('results.html', laptops=laptop_suggestions)
    return render_template('teacher_suggest.html')

@app.route('/it_suggest', methods=["GET", "POST"])
def it_suggest():
    if request.method == 'POST':
        data_list = []
        data = request.form
        for list_type in data.keys():
            data_list.append(int(data.getlist(list_type)[0]))
        data_list = np.array(data_list)
        predict = lr_model.predict([data_list])
        laptop_suggestions = laptop_mapper[predict[0]]
        return render_template('results.html', laptops=laptop_suggestions)
    return render_template('it_suggest.html')

@app.route('/streamlit')
def recommend_streamlit():
    return render_template("streamlit_recommend.html")

if __name__ == '__main__':
    app.run(debug=True)
