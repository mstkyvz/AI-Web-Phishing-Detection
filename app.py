from flask import Flask, request, render_template
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')
from feature import FeatureExtraction 


file = open("model.pkl", "rb")
gbc = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    result_class = None
    url = None
    if request.method == 'POST':
        url = request.form['url']
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        if y_pred == 1:
            result = "It is {0:.2f}% safe to go".format(y_pro_non_phishing * 100)
            result_class = "safe"
        else:
            result = "It is {0:.2f}% phishing".format(y_pro_phishing * 100)
            result_class = "phishing"

    return render_template('index.html', result=result, url=url, result_class=result_class)
if __name__ == '__main__':
    app.run(debug=True)
