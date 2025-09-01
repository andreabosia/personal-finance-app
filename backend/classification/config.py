from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    RESULTS_DB: str = "data/trusted/results.db"
    CANDIDATE_LABELS: List[str] = ["groceries","restaurants&bar","transport","utilities&subscriptons","shopping","salary", "investments", "bank transfer"]
    ANCHORS: Dict[str, List[str]] = {
        "groceries": ["Esselunga","Unes","Carrefour","Coop","Migros","Denner","Aldi","Lidl","Supermarket"],
        "restaurants&bar": ["ristorante","trattoria","bistro","pizzeria","bar","café", "Starbucks", "McDonald's", "Burger King", "KFC", "casafiorichiari", "kebab", "pizza", "pasticceria", "All Antico Vinaio", "Kanji", "Fra Diavolo", "Glovo"],
        "transport": ["swiss","volotea","easyjet","ryanair","flixbus","ATM","SBB","CFF","FFS","Tram","Taxi","Uber","ZVV","SBB.ch", "Trenord", "Italo", "Trenitalia"],
        "utilities&subscriptons": ["gym","palestra","netflix","rent","affitto","gas","bill", "wellhub", "apple music", "icloud", "Openai *Chatgpt"],
        "shopping": ["clothing","zara","levi","rinascnete","shop","amazon"],
        "salary": ["Ord: Prometeia S P a Ben:"],
        "investments": ["Compravendita Titoli"],
        "bank transfer": ["Revolut**"]
    }

    class Config:
        env_file = ".env"

settings = Settings()