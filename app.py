from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("calorie_prediction_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    protein = float(request.form["Protein"])
    fat = float(request.form["Fat"])
    carbs = float(request.form["Carbohydrates"])
    fiber = float(request.form["Fiber"])
    sugar = float(request.form["Sugar"])

    input_data = pd.DataFrame(
        [[protein, fat, carbs, fiber, sugar]],
        columns=[
            "Protein",
            "Fat",
            "Carbohydrates",
            "Fiber",
            "Sugar"
        ]
    )

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)