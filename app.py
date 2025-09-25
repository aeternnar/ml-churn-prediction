from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
import joblib
import io
import base64

pipe_xgb = joblib.load("./models/pipe_xgb.joblib")
necessary_columns = ['State', 'Account length', 'Area code',
                     'International plan', 'Number vmail messages',
                     'Total day minutes', 'Total day calls', 'Total eve minutes',
                     'Total eve calls', 'Total night minutes', 'Total night calls',
                     'Total intl minutes', 'Total intl calls',
                     'Customer service calls']

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    X = pd.read_csv(io.BytesIO(contents))

    missing = set(necessary_columns) - set(X.columns)
    if missing:
        return HTMLResponse(f"<h3 style='color:red'>Missing columns: {missing}</h3>")

    X_clear = X[necessary_columns]
    pred = pipe_xgb.predict(X_clear)
    pred_proba = pipe_xgb.predict_proba(X_clear)[:, 1]

    X["Churn prediction"] = pred
    X["Churn probability"] = pred_proba.round(3)

    output = io.StringIO()
    X.to_csv(output, index=False)
    csv_bytes = output.getvalue().encode()
    b64_csv = base64.b64encode(csv_bytes).decode()

    table_html = X.to_html(index=False, classes="table table-striped table-hover table-bordered")

    return templates.TemplateResponse("results.html", {
        "request": request,
        "table": table_html,
        "csv_b64": b64_csv
    })
