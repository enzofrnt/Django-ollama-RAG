from django.conf import settings
from httpx import ConnectError
from langchain_ollama import OllamaEmbeddings


def embed_query(text: str):
    """
    Génère un embedding pour le texte donné.

    :param text: Texte à encoder.
    :return: Embedding du texte.
    """
    model_name = settings.EMBEDDING_MODEL_NAME

    # Initialiser la fonction d'embedding avec le modèle donné
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url=settings.OLLAMA_API_URL,
    )

    try:
        embedding = embeddings.embed_query(text)
    except ConnectError:
        raise ConnectError("❌ Erreur de connexion impossible d'accéder à Ollama.")

    return embedding
