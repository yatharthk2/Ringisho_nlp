git clone https://github.com/yatharthk2/Ringisho_nlp.git
cd Ringisho_nlp
python3 -m venv fastapi
source fastapi/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload