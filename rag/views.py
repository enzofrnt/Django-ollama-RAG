from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django.views.generic import ListView
from django_eventstream import send_event
from httpx import ConnectError
from langchain_community.document_loaders import PyPDFLoader

from .graph import display_cos_sim_in_3D
from .models import Chunk, Document
from .populate_database import add_to_django, split_documents
from .query_data import query_rag


@csrf_exempt
def chat(request):
    """
    Vue permettant de gérer le système de chat basé sur un modèle RAG (Retrieval-Augmented Generation).
    Envoie les messages en temps réel via des événements serveur (Server-Sent Events).
    """
    if request.method == "POST":
        query_text = request.POST.get("query")  # Récupère la requête utilisateur
        response_generator, sources = query_rag(query_text)  # Interroge le modèle RAG

        formatted_sources_text = clean_ids(
            sources
        )  # Nettoie les identifiants des sources

        # Définir un canal d'événements pour la session
        channel_name = f"chat"

        # Envoie les réponses en morceaux via des événements serveur
        try:
            for chunk in response_generator:
                send_event(channel_name, "message", {"text": chunk})
        except ConnectError:
            send_event(
                channel_name,
                "message",
                {"text": "❌ Erreur impossible d'accéder à Ollama."},
            )
            raise ConnectError("❌ Erreur de connexion impossible d'accéder à Ollama.")

        # Retourne les sources en réponse pour terminer
        return JsonResponse({"sources": formatted_sources_text})
    return render(request, "rag/chat.html")


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
        doc_name = str(document)
        document.delete()
        print(f"✅ Document '{doc_name}' et ses chunks associés ont été supprimés.")
        return JsonResponse({"status": "Document supprimé avec succès"})
    except Document.DoesNotExist:
        return JsonResponse({"error": "Document introuvable"}, status=404)


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


def view_request_in_3d(request):
    query = request.GET.get("query", "Requête par défaut si vide")
    k = 5

    graph_html, best_chunks = display_cos_sim_in_3D(query, k)

    return render(
        request,
        "interactive_graph.html",
        {"graph_html": graph_html, "query": query, "chunks": best_chunks},
    )


class ChunkListView(ListView):
    model = Chunk
    template_name = "chunk_list.html"
    context_object_name = "chunks"
