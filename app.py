from flask import Flask, render_template, request
import pandas as pd
from OEP import validation

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def hello_world():
    product = []  

    if request.method == "POST":
        # Retrieve checkbox values from the form
        selected_skin_types = ''
        if request.form.get('combi'):
            selected_skin_types += '1,'
        else:
            selected_skin_types += '0,'
        if request.form.get('dry'):
            selected_skin_types += '1,'
        else:
            selected_skin_types += '0,'
        if request.form.get('normal'):
            selected_skin_types += '1,'
        else:
            selected_skin_types += '0,'
        if request.form.get('oily'):
            selected_skin_types += '1,'
        else:
            selected_skin_types += '0,'
        if request.form.get('sensitive'):
            selected_skin_types += '1'
        else:
            selected_skin_types += '0'

        result = validation(selected_skin_types)
        
        # If result is a DataFrame, convert it to a dictionary, else set product to result
        if isinstance(result, pd.DataFrame):
            product = result.to_dict(orient='records')
        else:
            product = []

    # Render the form on a GET request
    return render_template('index.html', product = product)


@app.route("/index")
def idx():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)
