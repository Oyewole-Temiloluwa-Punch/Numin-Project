@echo off
echo Installing requirements...
pip install -r requirements.txt

echo Starting Streamlit application...
python -m streamlit run app.py

pause
