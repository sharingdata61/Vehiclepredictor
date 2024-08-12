import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, time
import joblib


cnxn_str = ("Driver={ODBC Driver 17 for SQL Server};"
            "Server=10.20.30.75,1433;"
            "Database=BI_TEST;"
            "UID=HC_Gaurav4;"
            "PWD=Gaurav4;"
            )

engine = create_engine('mssql+pyodbc:///?odbc_connect=' + cnxn_str)


query = """
    SELECT Operated_Day, Route_Name,
           (Actual_Trip_Start) AS Actual_Trip_START,
           (Actual_Trip_End) AS Actual_Trip_END
    FROM VehicleTrip_Detail_Process_Data
	WHERE Route_Name NOT LIKE '%STL%'
"""

df = pd.read_sql(query, engine)

# def combine_time_with_date(time_val):
#     return datetime.combine(datetime.today(), time_val)

# df['Actual_Trip_START'] = df['Actual_Trip_START'].apply(combine_time_with_date)
# df['Actual_Trip_END'] = df['Actual_Trip_END'].apply(combine_time_with_date)

X = df[['Operated_Day', 'Route_Name', 'Actual_Trip_START']].copy()
y = df['Actual_Trip_END'].copy()

label_encoder_day = LabelEncoder()
label_encoder_route = LabelEncoder()

X['Operated_Day'] = label_encoder_day.fit_transform(X['Operated_Day'])
X['Route_Name'] = label_encoder_route.fit_transform(X['Route_Name'])

def time_to_float(t):
   return t.hour + t.minute / 60 + t.second / 3600

X['Actual_Trip_START'] = X['Actual_Trip_START'].apply(time_to_float)
y = y.apply(time_to_float)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train model
model = RandomForestRegressor(n_estimators=10)
model.fit(X_train, y_train)

# Evaluate model accuracy
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy (R^2 score): {accuracy}')

# Make predictions on X_test
y_pred = model.predict(X_test)

# Convert predicted float values back to time format
def float_to_time(f):
    hours = int(f)
    minutes = int((f - hours) * 60)
    seconds = int((f - hours - minutes / 60) * 3600)
    return time(hours, minutes, seconds)

y_pred_time = [float_to_time(ts) for ts in y_pred]
y_test_time = [float_to_time(ts) for ts in y_test]


# Print predictions vs actual values
# print("Predictions vs Actual values:")
# for pred, actual in zip(y_pred_time, y_test_time):
#     print(f"Predicted: {pred}, Actual: {actual}")

joblib.dump(model, 'Tripmodel.joblib')
joblib.dump(label_encoder_day,'LabelEncoder.joblib')
joblib.dump(label_encoder_route,'LabelEncoder1.joblib')