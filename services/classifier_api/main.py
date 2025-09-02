from fastapi import FastAPI
# from services.classifier_api.schemas import CSVJob, JoinJob
# from services.classifier_api.bootstrap import classify_csv_idempotent, join_results_to_csv
from backend.classification.main import orchestrate

app = FastAPI(title="Classifier API")

@app.post("/classify_transactions")
def classify_transactions():
    orchestrate()

# @app.post("/classify/csv")
# def classify_csv(job: CSVJob):
#     return classify_csv_idempotent(
#         csv_path=job.csv_path,
#         models=job.models,
#         id_col=job.id_col,
#         merchant_col=job.merchant_col,
#     )

# @app.post("/classify/join")
# def classify_join(job: JoinJob):
#     return join_results_to_csv(
#         csv_path=job.csv_path,
#         id_col=job.id_col,
#         out_path=job.out_path,
#     )