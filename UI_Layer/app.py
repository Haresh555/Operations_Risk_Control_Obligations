from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    print (user_input)

    response = requests.request('POST',"http://localhost:5001/getResponseForPrompt/" ,verify=False,data=user_input)
    reply = response.json().get('msg')
    
    
     
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)