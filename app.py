#Importing necessary libraries and modules.
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler

#Creating a flask app
app = Flask(__name__)
#Unpickling the pkl file containing our trained model.
model = pickle.load(open('nmodel.pkl', 'rb'))
#Creating a route for homepage
@app.route('/')
def home():
    #returns the homepage html
    return render_template('index.html')
#Creating another route for the prediction
@app.route('/predict',methods=['POST'])
def predict():
    # Get all features values from the form
    input_features = [float(x) for x in request.form.values()]
    # Forming array of the input features
    features_values = [np.array(input_features)]
    #Since we peformed feature selection on the dataset we will use the 23 feature names.
    #Creating list of feature names in the same order.
    features_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    #Creating a dataframe with the input feature values and respective feature names.   
    df = pd.DataFrame(features_values, columns=features_names)
    #sc=StandardScaler()
    #df=sc.fit_transform(df)
    #Prediction of given input stored in output variable.
    output = model.predict(df)
    #Since Malignant is signified by 0 and Benign is signified by 1 in the dataset
    # We are checking the int output and storing the result as benign or malignant as a string to display on the page.    
    if output == [1]:
        return render_template('benign.html')
    elif output==[0]:
        return render_template('malignant.html')
        
    
    
if __name__ == "__main__":
    #Run the app
    app.run(debug=False)
