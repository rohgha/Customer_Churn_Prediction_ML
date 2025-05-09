import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template, jsonify
import pickle
import sqlite3
from datetime import datetime
import os # For checking template file existence, though not strictly necessary for render_template

app = Flask("__name__")

# --- Database Setup ---
DATABASE = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            churn_prediction_label TEXT NOT NULL,
            churn_prediction_value INTEGER NOT NULL,
            churn_confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
# --- End Database Setup ---

# Load data and models globally
try:
    df_1 = pd.read_csv("churn_dataset.csv")
    # Clean TotalCharges in df_1 once after loading
    if 'TotalCharges' in df_1.columns:
        df_1['TotalCharges'] = df_1['TotalCharges'].replace(' ', pd.NA) # Treat spaces as NA first
        df_1['TotalCharges'] = pd.to_numeric(df_1['TotalCharges'], errors='coerce').fillna(0.0)
    # Ensure SeniorCitizen is integer and handled
    if 'SeniorCitizen' in df_1.columns:
        df_1['SeniorCitizen'] = pd.to_numeric(df_1['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    # Ensure tenure is int
    if 'tenure' in df_1.columns:
        df_1['tenure'] = pd.to_numeric(df_1['tenure'], errors='coerce').fillna(0).astype(int)

    model = pickle.load(open("modela.sav", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error loading essential files: {e}. Make sure 'churn_dataset.csv', 'modela.sav', and 'model_columns.pkl' are present.")
    df_1 = pd.DataFrame() # Empty dataframe to prevent further errors
    model = None
    model_columns = []


@app.route("/")
def loadPage():
    return render_template('home.html', query="", customer_id_for_save="")

@app.route("/", methods=['POST'])
def predict():
    if model is None or not model_columns:
        return render_template('home.html', output1="Error: Model not loaded.", output2="Please check server logs.", **request.form)

    customer_id_to_save = request.form.get('customer_id_for_save', 'UNKNOWN_CUSTOMER')
    input_data = [
        request.form.get('query1', ''),  # SeniorCitizen
        request.form.get('query2', ''),  # MonthlyCharges
        request.form.get('query3', ''),  # MonthlyCharges
        request.form.get('query4', ''),  # MonthlyCharges
        request.form.get('query5', ''),  # MonthlyCharges
        request.form.get('query6', ''),  # MonthlyCharges
        request.form.get('query7', ''),  # MonthlyCharges
        request.form.get('query8', ''),  # MonthlyCharges
        request.form.get('query9', ''),  # MonthlyCharges
        request.form.get('query10', ''),  # MonthlyCharges
        request.form.get('query11', ''),  # MonthlyCharges
        request.form.get('query12', ''),  # MonthlyCharges
        request.form.get('query13', ''),  # MonthlyCharges
        request.form.get('query14', ''),  # MonthlyCharges
        request.form.get('query15', ''),  # MonthlyCharges
        request.form.get('query16', ''),  # MonthlyCharges
        request.form.get('query17', ''),  # MonthlyCharges
        request.form.get('query18', ''), # PaymentMethod
        request.form.get('query19', '')  # tenure
    ]
    o1 = ""
    o2 = ""
    prediction_val = None
    probability_val = None

    try:
        new_df_columns = [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'tenure'
        ]
        new_df = pd.DataFrame([input_data], columns=new_df_columns)

        new_df['SeniorCitizen'] = pd.to_numeric(new_df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
        new_df['MonthlyCharges'] = pd.to_numeric(new_df['MonthlyCharges'], errors='coerce').fillna(0.0).astype(float)
        new_df['TotalCharges'] = new_df['TotalCharges'].replace(' ', pd.NA)
        new_df['TotalCharges'] = pd.to_numeric(new_df['TotalCharges'], errors='coerce').fillna(0.0).astype(float)
        new_df['tenure'] = pd.to_numeric(new_df['tenure'], errors='coerce').fillna(0).astype(int)

        # Ensure df_1_for_concat has only the expected feature columns for dummification.
        # This requires df_1 to have these columns.
        # An empty DataFrame is created if not all columns are present in df_1.
        if all(col in df_1.columns for col in new_df_columns):
            df_1_subset_for_concat = df_1[new_df_columns].copy()
        else:
            # If df_1 is missing some columns, create an empty DataFrame with those columns.
            # This might lead to issues if dummification expects certain values.
            print("Warning: df_1 is missing some columns required for concatenation. Using an empty frame for these.")
            df_1_subset_for_concat = pd.DataFrame(columns=new_df_columns)


        df_combined = pd.concat([df_1_subset_for_concat, new_df], ignore_index=True)
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_combined['tenure_group'] = pd.cut(df_combined.tenure, range(1, 80, 12), right=False, labels=labels)
        df_combined.drop(columns=['tenure'], inplace=True)

        dummy_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]
        df_dummies = pd.get_dummies(df_combined[dummy_cols])
        final_df = df_dummies.reindex(columns=model_columns, fill_value=0)

        prediction_input = final_df.tail(1)
        prediction_val = model.predict(prediction_input)
        probability_val = model.predict_proba(prediction_input)[:, 1]

        o1 = "This customer is likely to be churned!!" if prediction_val[0] == 1 else "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability_val[0] * 100)

        if prediction_val is not None and probability_val is not None:
            db_conn = None
            print("DEBUG: Attempting to save prediction to database...")
            try:
                db_conn = sqlite3.connect(DATABASE)
                cursor = db_conn.cursor()
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pred_label = "Churned" if prediction_val[0] == 1 else "Continue"
                cursor.execute('''
                    INSERT INTO predictions (prediction_date, customer_id, churn_prediction_label, churn_prediction_value, churn_confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (current_date_time, customer_id_to_save, pred_label, int(prediction_val[0]), float(probability_val[0])))
                db_conn.commit()
                print(f"Prediction saved for customer {customer_id_to_save}")
            except sqlite3.Error as e:
                print(f"Database error: {e}")
                o1 += " (DB_Save_Error)"
            finally:
                if db_conn:
                    db_conn.close()

    except Exception as e:
        o1 = "Error occurred during prediction or data processing."
        o2 = f"Details: {str(e)}"
        print(f"Prediction/Processing Error: {e}")

    form_data_to_repopulate = {key: value for key, value in request.form.items()}
    if 'customer_id_for_save' not in form_data_to_repopulate:
        form_data_to_repopulate['customer_id_for_save'] = customer_id_to_save
        
    return render_template('home.html', output1=o1, output2=o2, **form_data_to_repopulate)


@app.route("/get_customer_data", methods=["POST"])
def get_customer_data():
    if df_1.empty:
         return jsonify({"status": "error", "message": "Customer dataset not loaded."})

    customer_id_lookup = request.json['customer_id']
    match = df_1[df_1['customerID'] == customer_id_lookup]

    if match.empty:
        return jsonify({"status": "error", "message": "Customer not found"})

    row = match.iloc[0]
    # TotalCharges is already cleaned to float or 0.0 in global df_1 loading
    total_charges = row["TotalCharges"]

    customer_form_data = {
        "1": str(row.get("SeniorCitizen", 0)), # .get for safety
        "2": str(row.get("MonthlyCharges", 0.0)),
        "3": str(total_charges),
        "4": str(row.get("gender", "")),
        "5": str(row.get("Partner", "")),
        "6": str(row.get("Dependents", "")),
        "7": str(row.get("PhoneService", "")),
        "8": str(row.get("MultipleLines", "")),
        "9": str(row.get("InternetService", "")),
        "10": str(row.get("OnlineSecurity", "")),
        "11": str(row.get("OnlineBackup", "")),
        "12": str(row.get("DeviceProtection", "")),
        "13": str(row.get("TechSupport", "")),
        "14": str(row.get("StreamingTV", "")),
        "15": str(row.get("StreamingMovies", "")),
        "16": str(row.get("Contract", "")),
        "17": str(row.get("PaperlessBilling", "")),
        "18": str(row.get("PaymentMethod", "")),
        "19": str(row.get("tenure", 0)),
        "customer_id_for_save": str(customer_id_lookup)
    }
    return jsonify({"status": "ok", "customer": customer_form_data})

# --- NEW HISTORY PAGE ROUTE ---
@app.route("/history")
def history():
    if not os.path.exists('templates/history.html'):
        return "Error: History template not found.", 404
    
    logs_for_template = []
    db_conn = None
    try:
        db_conn = sqlite3.connect(DATABASE)
        db_conn.row_factory = sqlite3.Row # Access columns by name
        cursor = db_conn.cursor()
        
        # Fetch predictions from the database
        cursor.execute("""
            SELECT customer_id, prediction_date, churn_prediction_value 
            FROM predictions 
            ORDER BY prediction_date DESC
        """)
        prediction_logs_from_db = cursor.fetchall()

        if df_1.empty:
            print("Warning: Customer data (df_1) is not loaded. History will only show limited prediction data.")
        
        for pred_log in prediction_logs_from_db:
            customer_id = pred_log['customer_id']
            record = {
                'customerID': customer_id,
                'prediction': pred_log['churn_prediction_value'],
                'prediction_time': pred_log['prediction_date']
            }

            if not df_1.empty and 'customerID' in df_1.columns:
                customer_details_series = df_1[df_1['customerID'] == customer_id]
                if not customer_details_series.empty:
                    details = customer_details_series.iloc[0]
                    # These keys must match your history.html template
                    record.update({
                        'gender': details.get('gender', 'N/A'),
                        'SeniorCitizen': details.get('SeniorCitizen', 0), # Already int
                        'Partner': details.get('Partner', 'N/A'),
                        'Dependents': details.get('Dependents', 'N/A'),
                        'tenure': details.get('tenure', 0), # Already int
                        'PhoneService': details.get('PhoneService', 'N/A'),
                        'MultipleLines': details.get('MultipleLines', 'N/A'),
                        'InternetService': details.get('InternetService', 'N/A'),
                        'OnlineSecurity': details.get('OnlineSecurity', 'N/A'),
                        'OnlineBackup': details.get('OnlineBackup', 'N/A'),
                        'DeviceProtection': details.get('DeviceProtection', 'N/A'),
                        'TechSupport': details.get('TechSupport', 'N/A'),
                        'StreamingTV': details.get('StreamingTV', 'N/A'),
                        'StreamingMovies': details.get('StreamingMovies', 'N/A'),
                        'Contract': details.get('Contract', 'N/A'),
                        'PaperlessBilling': details.get('PaperlessBilling', 'N/A'),
                        'PaymentMethod': details.get('PaymentMethod', 'N/A'),
                        'MonthlyCharges': details.get('MonthlyCharges', 0.0), # Already float
                        'TotalCharges': details.get('TotalCharges', 0.0)    # Already float
                    })
                else: # Customer ID from prediction DB not found in df_1
                    print(f"Warning: Customer ID {customer_id} from predictions DB not found in df_1.")
                    for key in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                'MonthlyCharges', 'TotalCharges']:
                        record[key] = 'N/A' # Fill missing details
            else: # df_1 is empty or missing customerID column
                 for key in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                'MonthlyCharges', 'TotalCharges']:
                        record[key] = 'Data Unavailable'


            logs_for_template.append(record)
            
    except sqlite3.Error as e:
        print(f"Database error while fetching history: {e}")
        # Optionally, pass an error message to the template
    except Exception as e:
        print(f"An error occurred while preparing history: {e}")
    finally:
        if db_conn:
            db_conn.close()
            
    return render_template('history.html', logs=logs_for_template)
# --- END HISTORY PAGE ROUTE ---


if __name__ == "__main__":
    init_db() # Initialize the database (create table if not exists)
    app.run(debug=True)