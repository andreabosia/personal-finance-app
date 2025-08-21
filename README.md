# personal-finance-app
Web App that allows to track personal expenses and budgeting. Developed leveraging streamline for UI, FastAPI  for data processing, Docker for containerisation, airflow for automation. 



start venv:
    source .app_venv/bin/activate

pip install packages:

    to safely bind packages to the python being used in the venv use python -m pip package_name instead of just pip install package_name

run streamlit app:
   streamlit run frontend/streamlit_app.py
   python -m streamlit run frontend/streamlit_app.py

stop streamlit app from terminal:
    control + c

remove from cache files that were tracked before adding them to gitignore:
    git rm --cached file_name

run backend with FastAPI:
    uvicorn backend.api_main:app 

Swagger UI for API doc:
    http://127.0.0.1:8000/docs 

start server with fast API:
    for production use -> fastapi run backend/api_main.py
    for dev use -> python -m fastapi dev backend/api_main.py
    preffered way (to understand) -> uvicorn backend.api_main:app --reload
