import argparse
import subprocess

from django.conf import settings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from .get_embedding_function import get_embedding_function

from pgvector.django import CosineDistance

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
        similarity=CosineDistance("embedding", query_embedding)
    ).order_by(
        "similarity"
    )[  # Distance cosinus croissante (plus proche = meilleur)
        :top_k
    ]  # Limite à top_k résultats

    return similar_chunks


from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from .get_embedding_function import get_embedding_function
from .models import Chunk


def query_rag_with_postgres(query_text: str):
    """
    Interroge une base PostgreSQL pour récupérer des chunks similaires,
    puis utilise un modèle de langage pour répondre.

    :param query_text: Question utilisateur.
    :return: Générateur de réponse et liste des sources.
    """
    # Générer l'embedding pour la requête
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query_text)

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
