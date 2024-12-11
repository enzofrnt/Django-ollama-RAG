import logging
import mimetypes

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django.views.generic import ListView
from django_eventstream import send_event
from httpx import ConnectError
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from .graph import display_cos_sim_in_3D
from .models import Chunk, Document
from .populate_database import add_to_django, split_documents
from .query_data import query_rag

logger = logging.getLogger(__name__)

mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
)


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
        channel_name = "chat"

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

        # Types de fichiers pris en charge
        supported_types = {
            "application/pdf": PyPDFLoader,  # Pour les PDF
            "text/plain": TextLoader,  # Pour les fichiers .txt
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": UnstructuredWordDocumentLoader,  # Pour les .docx
            "text/markdown": TextLoader,  # Pour les .md
            "text/x-markdown": TextLoader,  # Cas alternatif pour .md
            "text/x-wiki": TextLoader,  # Pour les fichiers Wikitext
        }

        for uploaded_file in uploaded_files:
            # Détecter le type de fichier en fonction de son extension
            file_type, encoding = mimetypes.guess_type(uploaded_file.name)

            # Vérifie si le type MIME est pris en charge
            if file_type not in supported_types:
                return JsonResponse(
                    {
                        "error": f"Type de fichier '{file_type}' non pris en charge pour '{uploaded_file.name}'"
                    },
                    status=400,
                )

            # Créer l'objet Document en base de données
            document = Document.objects.create(file=uploaded_file)
            logger.info(f"✅ Fichier '{document.file.name}' sauvegardé.")
            document.save()

            # Charger et traiter le document en fonction de son type MIME
            try:
                loader_class = supported_types[file_type]
                loader = loader_class(document.file.path)
                pages = loader.load()
            except Exception as e:
                document.delete()
                logger.error(
                    f"❌ Erreur de chargement du fichier '{document.file.name}': {str(e)}"
                )
                return JsonResponse(
                    {
                        "error": f"Erreur de traitement du fichier '{uploaded_file.name}'"
                    },
                    status=400,
                )

            # Diviser le document en chunks
            try:
                chunks = split_documents(pages)
                add_to_django(chunks, document)
            except Exception as e:
                logger.error(
                    f"❌ Erreur de segmentation du fichier '{document.file.name}': {str(e)}"
                )
                document.delete()
                return JsonResponse(
                    {
                        "error": f"Erreur de segmentation du fichier '{uploaded_file.name}'"
                    },
                    status=400,
                )

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
        logger.info(
            f"✅ Document '{doc_name}' et ses chunks associés ont été supprimés."
        )
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

    graph_html_pca, graph_html_tsne, graph_html_umap, best_chunks = (
        display_cos_sim_in_3D(query, k)
    )

    return render(
        request,
        "interactive_graph.html",
        {
            "graph_html_pca": graph_html_pca,
            "graph_html_tsne": graph_html_tsne,
            "graph_html_umap": graph_html_umap,
            "query": query,
            "chunks": best_chunks,
        },
    )


class ChunkListView(ListView):
    model = Chunk
    template_name = "chunk_list.html"
    context_object_name = "chunks"
