
# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__) # initializing a flask app
@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            PClass=int(request.form['PClass'])
            age = int(request.form['age'])
            fare = float(request.form['fare'])
            sex = request.form['sex']
            Sibsp = int(request.form['Sibsp'])
            Parch = int(request.form['Parch'])
            family=Sibsp+Parch
            if(sex=='male'):
                gender=1
            else:
                gender=0
                
            with open("standardScalar.sav", 'rb') as f:
                scalar = pickle.load(f)

            with open("modelForPrediction.sav", 'rb') as f:
                model = pickle.load(f)
            # predictions using the loaded model file
            scaled_data = scalar.transform([[PClass,age,fare,gender,family]])
            prediction=model.predict(scaled_data)
            print('prediction is', prediction)
            if(prediction[0] == 0):
                did_survive='did not survive'
            else:
                did_survive='survived'
            # showing the prediction results in a UI
            return render_template('results.html',prediction=did_survive)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True) # running the app
