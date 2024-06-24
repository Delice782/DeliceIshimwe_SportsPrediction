from flask import Flask, request, render_template_string, jsonify
import random

app = Flask(__name__)

def predict_player_rating(features):
    rating = list(range(80, 96))
    confidence_range = [round(i * 0.01, 2) for i in range(80, 101)]
    
    predicted_rating = random.choice(rating)
    confidence_score = random.choice(confidence_range)
    return predicted_rating, confidence_score

index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enter Player features to predict Player rating</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 600px;
            margin: 0 auto;
            text-align: center;
        }
        h2 {
            margin-bottom: 20px;
            color: #333;
        }
        form {
            display: flex;
            flex-direction: column;
        }
        label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        input[type="text"] {
            padding: 8px;
            width: 60%;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="submit"] {
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            width: 100px;
            height: 50px;
            align-self: center;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        #result {
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
        }
    </style>
    <script>
        async function predictRating(event) {
            event.preventDefault();
            const pace = document.querySelector('input[name="pace"]').value;
            const shooting = document.querySelector('input[name="shooting"]').value;
            const passing = document.querySelector('input[name="passing"]').value;
            const dribbling = document.querySelector('input[name="dribbling"]').value;
            const defending = document.querySelector('input[name="defending"]').value;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    pace: pace,
                    shooting: shooting,
                    passing: passing,
                    dribbling: dribbling,
                    defending: defending
                })
            });

            const result = await response.json();
            document.getElementById('result').innerHTML = `Rating: ${result.prediction}<br>Confidence Score: ${result.confidence}`;
        }
    </script>
</head>
<body>
    <div class="container">
        <h2>Enter Player features to predict Player rating:</h2>
        <form onsubmit="predictRating(event)">
            <label>Pace: <input type="text" name="pace" required></label><br>
            <label>Shooting: <input type="text" name="shooting" required></label><br>
            <label>Passing: <input type="text" name="passing" required></label><br>
            <label>Dribbling: <input type="text" name="dribbling" required></label><br>
            <label>Defending: <input type="text" name="defending" required></label><br>
            <input type="submit" value="Predict">
        </form>
        <div id="result"></div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pace = int(data['pace'])
    shooting = int(data['shooting'])
    passing = int(data['passing'])
    dribbling = int(data['dribbling'])
    defending = int(data['defending'])

    predicted_rating, confidence_score = predict_player_rating({
        'pace': pace,
        'shooting': shooting,
        'passing': passing,
        'dribbling': dribbling,
        'defending': defending
    })

    return jsonify({
        'prediction': predicted_rating,
        'confidence': confidence_score
    })

if __name__ == "__main__":
    app.run(debug=True)
