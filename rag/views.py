import os
import re

from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django_eventstream import send_event
from langchain_community.document_loaders import PyPDFLoader

from .get_embedding_function import get_embedding_function
from .populate_database import (
    add_to_django,
    load_documents,
    populate_database,
    split_documents,
)
from .query_data import query_rag_with_postgres


@csrf_exempt
def chat(request):
    """
    Vue permettant de gérer le système de chat basé sur un modèle RAG (Retrieval-Augmented Generation).
    Envoie les messages en temps réel via des événements serveur (Server-Sent Events).
    """
    if request.method == "POST":
        query_text = request.POST.get("query")  # Récupère la requête utilisateur
        # response_generator, sources = query_rag(query_text)  # Interroge le modèle RAG
        response_generator, sources = query_rag_with_postgres(
            query_text
        )  # Interroge le modèle RAG

        formatted_sources_text = clean_ids(
            sources
        )  # Nettoie les identifiants des sources

        # Définir un canal d'événements pour la session
        channel_name = f"chat"

        # Envoie les réponses en morceaux via des événements serveur
        for chunk in response_generator:
            send_event(channel_name, "message", {"text": chunk})

        # Retourne les sources en réponse pour terminer
        return JsonResponse({"sources": formatted_sources_text})
    return render(request, "rag/chat.html")  # Charge la page HTML pour le chat


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Document
from .populate_database import add_to_django, load_documents_from_files, split_documents


@csrf_exempt
def add_file(request):
    if request.method == "POST" and request.FILES:
        uploaded_files = request.FILES.getlist("files")
        for uploaded_file in uploaded_files:
            # Créez une instance du modèle Document pour chaque fichier
            document = Document.objects.create(file=uploaded_file)
            document.save()
            print(f"✅ Fichier '{document.file.name}' sauvegardé.")

            # Charger et traiter le document
            loader = PyPDFLoader(document.file.path)
            pages = loader.load()
            chunks = split_documents(pages)
            add_to_django(chunks, document)

        return JsonResponse({"status": "Fichiers ajoutés avec succès"})
    return JsonResponse({"error": "Aucun fichier envoyé"}, status=400)


@csrf_exempt
@require_GET
def list_documents(request):
    """
    Vue permettant de lister tous les documents présents dans la base de données.
    """
    documents = Document.objects.all()
    document_names = [doc.file.name for doc in documents]
    return JsonResponse({"documents": document_names})


from django.http import JsonResponse

from .models import Chunk


@csrf_exempt
@require_GET
def list_documents_postgres(request):
    documents = Document.objects.all()
    document_list = [{"id": doc.id, "name": str(doc)} for doc in documents]
    return JsonResponse({"documents": document_list})


@csrf_exempt
@require_POST
def delete_document(request):
    doc_id = request.POST.get("doc_id")
    if not doc_id:
        return JsonResponse({"error": "ID du document manquant"}, status=400)

    try:
        document = Document.objects.get(pk=doc_id)
        document.delete()
        print(
            f"✅ Document '{document.file.name}' et ses chunks associés ont été supprimés."
        )
        return JsonResponse({"status": "Document supprimé avec succès"})
    except Document.DoesNotExist:
        return JsonResponse({"error": "Document introuvable"}, status=404)


from .models import Chunk


def delete_file_references_postgres(file_name: str):
    """
    Supprime toutes les références liées à un fichier dans la base PostgreSQL.

    :param file_name: Nom du fichier à supprimer (e.g., "mon_fichier.pdf").
    """
    # Rechercher tous les chunks associés à la source (file_name)
    chunks_to_delete = Chunk.objects.filter(source=file_name)

    # Vérifier si des chunks existent pour ce fichier
    if chunks_to_delete.exists():
        count = chunks_to_delete.count()
        chunks_to_delete.delete()
        print(f"✅ {count} références supprimées pour '{file_name}'.")
    else:
        print(f"🚫 Aucune référence trouvée pour '{file_name}'.")


def delete_file_references(file_name: str):
    """
    Supprime toutes les références liées à un fichier dans la base PostgreSQL
    et supprime le fichier via le modèle Document.
    """
    # Supprimer les chunks associés
    chunks_to_delete = Chunk.objects.filter(source=file_name)
    count_chunks = chunks_to_delete.count()
    chunks_to_delete.delete()

    # Supprimer le document
    document = Document.objects.filter(file__endswith=file_name).first()
    if document:
        document.delete()
        print(f"✅ Document '{file_name}' supprimé.")
    else:
        print(f"🚫 Document '{file_name}' introuvable dans la base.")

    print(f"✅ {count_chunks} références supprimées pour '{file_name}'.")


def delete_file(file_name: str):
    """
    Supprime un fichier spécifique du système de fichiers.
    :param file_name: Nom du fichier à supprimer.
    """
    file_path = os.path.join(settings.DATA_PATH, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Fichier '{file_name}' supprimé avec succès.")
    else:
        print(f"🚫 Fichier '{file_name}' introuvable.")


def clean_ids(documents):
    """
    Nettoie les identifiants des documents pour n'extraire que l'ID de base.
    :param documents: Liste des identifiants bruts.
    :return: Liste nettoyée des identifiants.
    """
    cleaned_id = set()
    for id in documents:
        cleaned_id.add(id.split(":")[0].split("/")[-1])
    return list(cleaned_id)


from django.views.generic import ListView

from .models import Chunk


class ChunkListView(ListView):
    model = Chunk
    template_name = "chunk_list.html"  # Spécifie le template
    context_object_name = "chunks"
