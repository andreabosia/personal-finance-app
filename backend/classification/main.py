import yaml
import pandas as pd
from backend.classification.models.embedding import EmbeddingAnchorConfig, EmbeddingAnchorClassifier
from backend.classification.models.utils import ClassificationRequest
"""
This scrpt reads a config yaml file and initialize the model accordingly.
"""

def load_model_from_yaml(yaml_path):
    classifiers_list = []
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    for key, config in config_dict.items():
        if key.lower() == 'embeddingconfig':
            model_config = EmbeddingAnchorConfig(**config)
            model = EmbeddingAnchorClassifier(model_config)
            classifiers_list.append(model)
        # Add other model types here as elif blocks
        else:
            raise ValueError(f"Unsupported model type in YAML: {key}")
    return classifiers_list

def orchestrate(model_config_path="/Users/andreabosia/Projects/personal-finance-app/backend/classification/artifacts/model_config.yaml",
                transactions_df_path="/Users/andreabosia/Projects/personal-finance-app/data/trusted/transactions.csv",
                merchant_col="descrizione"):
    classifiers_list = load_model_from_yaml(model_config_path)
    df = pd.read_csv(transactions_df_path)
    for classifier in classifiers_list:
        preds = classifier.predict(df[merchant_col])
        df[f'pred_{classifier.name}'] = preds
    df.to_csv("/Users/andreabosia/Projects/personal-finance-app/data/trusted/transactions_classified.csv", index=False)
    return df
if __name__ == "__main__":
    df = orchestrate()
    #print(df.head())

# Example usage: python -m backend.classification.main