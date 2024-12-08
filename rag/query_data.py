from django.conf import settings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pgvector.django import CosineDistance

from .embedding_function import embed_query
from .models import Chunk


def get_similar_chunks(query_embedding, top_k=5):
    """
    Trouve les chunks les plus similaires à un embedding donné en utilisant la distance cosinus.

    :param query_embedding: Embedding de la requête utilisateur (liste de flottants).
    :param top_k: Nombre de résultats les plus proches à retourner.
    :return: Liste des chunks et leurs distances.
    """
    # Effectuer la recherche avec CosineDistance
    similar_chunks = Chunk.objects.annotate(
        distance=CosineDistance("embedding", query_embedding)
    ).order_by(
        "distance"
    )[  # Distance cosinus croissante (plus proche = meilleur)
        :top_k
    ]  # Limite à top_k résultats

    return similar_chunks


def query_rag(query_text: str):
    """
    Interroge une base PostgreSQL pour récupérer des chunks similaires,
    puis utilise un modèle de langage pour répondre.

    :param query_text: Question utilisateur.
    :return: Générateur de réponse et liste des sources.
    """
    # Générer l'embedding pour la requête
    query_embedding = embed_query(query_text)

    # Rechercher les chunks similaires
    similar_chunks = get_similar_chunks(query_embedding)

    if not similar_chunks:
        return iter(["Désolé, aucun document pertinent trouvé."]), []

    # Générer le contexte à partir des chunks
    context_text = "\n\n---\n\n".join([chunk.content for chunk in similar_chunks])

    # Charger le modèle de langage
    prompt_template = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = OllamaLLM(model=settings.LANGUAGE_MODEL_NAME)

    # Streamer la réponse et collecter les sources
    response_generator = model.stream(prompt)
    sources = [
        f"{chunk.document.file.name}: Page {chunk.page}, Chunk {chunk.chunk_index}"
        for chunk in similar_chunks
    ]

    return response_generator, sources
