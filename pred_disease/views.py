from django.http import HttpResponse
from django.shortcuts import render

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import pickle

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


def index(request):
    return render(request, "index.html")


def predict(request):
    return render(request, "predict.html")


def model_ML(data_input):
    with open(
        os.path.join(BASE_DIR, "pred_disease/model_ML/predict_disease_DT.sav"), "rb"
    ) as r:
        r_model_ML = pickle.load(r)
        r.close()
    return r_model_ML.predict(data_input)


def output(request):
    data = pd.read_csv(os.path.join(BASE_DIR, "pred_disease/dataset/dataset.csv"))
    X = data.drop(columns=["Unnamed: 0", "Diagnosis"]).values
    Y = data["Diagnosis"].values
    data = data.drop(columns=["Unnamed: 0", "Diagnosis"])
    if request.method == "POST":
        updated_request = request.POST.copy()
        for col in data:
            if updated_request.get(col) == None:
                updated_request.update({col: 0})
        myDict = dict(updated_request)
        newData = pd.DataFrame(myDict)
        newData = newData.drop(["csrfmiddlewaretoken"], axis=1)
        for col in data:
            newData[col] = newData[col].astype(data[col].dtype)
        predictDisease = model_ML(newData)
        html_class = []
        tingkat_kesakitan = []
        penyakit = ["jantung", "stroke", "cancer", "diabetes"]
        for i in range(0, len(penyakit)):
            if predictDisease[0] == i + 1:
                html_class.append("text-danger")
                tingkat_kesakitan.append("TINGGI")
            else:
                html_class.append("text-success")
                tingkat_kesakitan.append("RENDAH")
        output = {
            "output": [
                {
                    "penyakit": penyakit[0],
                    "tingkat_kesakitan": tingkat_kesakitan[0],
                    "html_class": html_class[0],
                },
                {
                    "penyakit": penyakit[1],
                    "tingkat_kesakitan": tingkat_kesakitan[1],
                    "html_class": html_class[1],
                },
                {
                    "penyakit": penyakit[2],
                    "tingkat_kesakitan": tingkat_kesakitan[2],
                    "html_class": html_class[2],
                },
                {
                    "penyakit": penyakit[3],
                    "tingkat_kesakitan": tingkat_kesakitan[3],
                    "html_class": html_class[3],
                },
            ],
        }
        return render(request, "output.html", output)


def decisionTree(request):
    return render(request, "decisionTree.html")
