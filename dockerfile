FROM python:3.11.6-slim

WORKDIR / C:\Users\gaurav.kumar4\Desktop\FINAL Vehicle Predictor

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
