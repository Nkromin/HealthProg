from itertools import count
from flask import Flask, render_template ,url_for ,request,Response, jsonify
import numpy as np
import database
import prediction
import json
import io
import random
import visualization
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import modelbuild
from datetime import datetime
try:
    from flasgger import Swagger
    _HAS_FLASGGER = True
except Exception:
    # Flasgger not available in the analyzer environment; proceed without Swagger UI
    Swagger = None
    _HAS_FLASGGER = False


app = Flask ( __name__ )

# Initialize Flasgger Swagger UI (only if available)
swagger_template = {
    "info": {
        "title": "Cardio Monitor API",
        "description": "API for uploading blood reports and heart readings from wearables and getting simple health analysis.",
        "version": "1.0.0"
    }
}
if _HAS_FLASGGER:
    Swagger(app, template=swagger_template)

# In-memory data store for uploaded readings
DATA_STORE = {}


def analyze_blood(blood):
    """Simple blood analysis logic (heuristic checks)."""
    issues = []
    # Example fields: hemoglobin, wbc, rbc, cholesterol, glucose
    hgb = blood.get("hemoglobin")
    if isinstance(hgb, (int, float)):
        if hgb < 12:
            issues.append("Low hemoglobin (possible anemia).")
        elif hgb > 17:
            issues.append("High hemoglobin.")
    chol = blood.get("cholesterol")
    if isinstance(chol, (int, float)) and chol > 200:
        issues.append("High cholesterol (risk for cardiovascular disease).")
    glucose = blood.get("glucose")
    if isinstance(glucose, (int, float)):
        if glucose >= 126:
            issues.append("High fasting glucose (possible diabetes).")
        elif glucose >= 100:
            issues.append("Elevated glucose (pre-diabetes).")
    wbc = blood.get("wbc")
    if isinstance(wbc, (int, float)) and wbc > 11000:
        issues.append("Elevated WBC (possible infection).")
    return issues


def analyze_heart(heart):
    """Simple heart reading analysis logic (blood pressure and heart rate)."""
    issues = []
    # Example fields: systolic, diastolic, heart_rate
    sys = heart.get("systolic")
    dia = heart.get("diastolic")
    hr = heart.get("heart_rate")
    if isinstance(sys, (int, float)) and isinstance(dia, (int, float)):
        if sys >= 140 or dia >= 90:
            issues.append("High blood pressure (hypertension).")
        elif sys < 90 or dia < 60:
            issues.append("Low blood pressure (hypotension).")
    if isinstance(hr, (int, float)):
        if hr > 100:
            issues.append("High heart rate (tachycardia).")
        elif hr < 50:
            issues.append("Low heart rate (bradycardia).")
    return issues


def save_reading(user_id, kind, payload):
    now = datetime.utcnow().isoformat() + "Z"
    analysis = analyze_blood(payload) if kind == "blood" else analyze_heart(payload)
    entry = {"data": payload, "analysis": analysis, "ts": now}
    DATA_STORE.setdefault(user_id, {})[kind] = entry
    return entry


def create_figure1(data1):
    fig = plt.subplots(figsize =(12, 8))
    barWidth = 0.25
    normal = data1[0]
    user = data1[1]
    br1 = np.arange(len(normal))
    br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]
    plt.bar(br1, normal, color ='g', width = barWidth,edgecolor ='grey', label ='Normal Value')
    plt.bar(br2, user, color ='r', width = barWidth,edgecolor ='grey', label ="Yours Value")
    # plt.bar(br3, CSE, color ='b', width = barWidth, edgecolor ='grey', label ='CSE')
    plt.xlabel('Health status defining attributes', fontweight ='bold', fontsize = 15)
    plt.ylabel('respective values', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(normal))],['cp','chol','fbs','exang','oldpeak','slope','ca','thal'])
    plt.legend()
    plt.savefig('static/plotng.png') 

def create_figure2(data2):
    fig = plt.subplots(figsize =(12, 8))
    barWidth = 0.25
    normal = data2[0]
    user = data2[1]
    br1 = np.arange(len(normal))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, normal, color ='g', width = barWidth,edgecolor ='grey', label ='Normal Value')
    plt.bar(br2, user, color ='r', width = barWidth,edgecolor ='grey', label ="Yours Value")
    plt.xlabel('Health status defining attributes', fontweight ='bold', fontsize = 15)
    plt.ylabel('respective values', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(normal))],['trestbps','chol','thalach'])
    plt.legend()
    plt.savefig('static/plotng2.png') 

@app.route('/api/upload/blood', methods=['POST'])
def upload_blood():
    """
    Upload a blood report JSON and receive a quick analysis.
    ---
    tags:
      - Health API
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            user_id:
              type: string
            blood:
              type: object
              properties:
                hemoglobin:
                  type: number
                cholesterol:
                  type: number
                glucose:
                  type: number
                wbc:
                  type: number
    responses:
      200:
        description: Stored blood reading and analysis
      400:
        description: Invalid request
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    user_id = payload.get("user_id", "anonymous")
    blood = payload.get("blood")
    if not isinstance(blood, dict):
        return jsonify({"error": "Missing or invalid 'blood' object"}), 400
    entry = save_reading(user_id, "blood", blood)
    return jsonify({"status": "ok", "stored": entry}), 200


@app.route('/api/upload/heart', methods=['POST'])
def upload_heart():
    """
    Upload heart readings (from a watch/ring) and receive a quick analysis.
    ---
    tags:
      - Health API
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            user_id:
              type: string
            heart:
              type: object
              properties:
                systolic:
                  type: number
                diastolic:
                  type: number
                heart_rate:
                  type: number
    responses:
      200:
        description: Stored heart reading and analysis
      400:
        description: Invalid request
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    user_id = payload.get("user_id", "anonymous")
    heart = payload.get("heart")
    if not isinstance(heart, dict):
        return jsonify({"error": "Missing or invalid 'heart' object"}), 400
    entry = save_reading(user_id, "heart", heart)
    return jsonify({"status": "ok", "stored": entry}), 200


@app.route('/api/report/<user_id>', methods=['GET'])
def get_report(user_id):
    """
    Get latest combined report (blood + heart) for a user.
    ---
    tags:
      - Health API
    parameters:
      - name: user_id
        in: path
        type: string
        required: true
        description: User identifier
    responses:
      200:
        description: Combined report with analyses
      404:
        description: No data for user
    """
    user = DATA_STORE.get(user_id)
    if not user:
        return jsonify({"error": "No data for user"}), 404
    combined = {
        "blood": user.get("blood"),
        "heart": user.get("heart"),
        "summary": (user.get("blood", {}).get("analysis", []) + user.get("heart", {}).get("analysis", []))
    }
    return jsonify(combined), 200

@app.route('/')
def home():
    global counter2
    counter2+=1
    return render_template('home.html',all_count=counter2)


global counter
counter=0
global counter2
counter2=0

@app.route('/predict',methods=['POST'])
def predict():
    global data1
    global data2
    global counter
    global counter2
    if request.method  == 'POST':
        nameofpatient= request.form ['name']
        age= request.form ['age']
        sex=request.form ['sex']
        cp= request.form ['cp']
        trestbps= request.form ['trestbps']
        chol= request.form ['chol']
        fbs= request.form ['fbs']
        restecg=request.form ['restecg']
        thalach=request.form ['thalach']
        exang=request.form ['exang']
        oldpeak=request.form ['oldpeak']
        slope=request.form ['slope']
        ca=request.form ['ca']
        thal=request.form ['thal']
        counter+=1
        if(counter<=50):
            result=prediction.preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal )
        else:
            #modelbuild.bulidmodel()
            result=prediction.preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal )
            counter=0
        #database.crudOperation(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal,result)
        data1,data2=visualization.visualizationpreprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal,result)
        create_figure1(data1)
        create_figure2(data2)
        return render_template ('result.html',prediction = result, nameofpatient=nameofpatient, model_counter=counter, total_counter=counter2)

@app.route('/about')
def about():
    return render_template('disease.html')


@app.errorhandler(500)
def internal_error(error):

    return render_template('error.html')


@app.errorhandler(404)
def not_found(error):
    return "404 error",404

if __name__ == '__main__':
    app.run( debug = True)