import joblib
def predict(data):
    model = joblib.load('Tripmodel.joblib')
    
    return model.predict(data)