from django.urls import include, path
from django.views.generic import RedirectView
from django_eventstream import urls

from . import views
from .views import ChunkListView

urlpatterns = [
    path(
        "", RedirectView.as_view(url="chat/", permanent=True)
    ),  # Redirige vers la page de chat
    path("chat/", views.chat, name="chat"),  # Page de chat
    path("add_file/", views.add_file, name="add_file"),  # Ajouter un fichier
    # path(
    #     "list_documents/", views.list_documents, name="list_documents"
    # ),  # Liste des documents / charger les documents qui ne le sont pas encore
    path(
        "list_documents/", views.list_documents_postgres, name="list_documents"
    ),  # Liste des documents / charger les documents qui ne le sont pas encore
    path(
        "delete_document/", views.delete_document, name="delete_document"
    ),  # Supprimer un document
    path(
        "events/",
        include(urls),
        {"channels": ["chat"]},
    ),  # URL pour les événements, chat en temps réel
    path("chunks/", ChunkListView.as_view(), name="chunk_list"),
]
