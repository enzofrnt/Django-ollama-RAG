from django.conf import settings
from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Initialise et retourne une fonction d'embedding basée sur le modèle spécifié dans les paramètres.

    L'embedding est utilisé pour convertir du texte en représentations numériques
    afin de permettre des comparaisons et des recherches basées sur la similarité.

    :return: Une instance d'OllamaEmbeddings.
    """
    # Charger le modèle spécifié dans les paramètres
    model_name = settings.EMBEDDING_MODEL_NAME

    # Initialiser la fonction d'embedding avec le modèle donné
    embeddings = OllamaEmbeddings(model=model_name)

    return embeddings
