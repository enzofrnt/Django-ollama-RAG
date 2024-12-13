from django.urls import include, path
from django.views.generic import RedirectView
from django_eventstream import urls
from django_eventstream.viewsets import EventsViewSet
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from hybridrouter import HybridRouter

from . import views, viewsets

router = HybridRouter()
router.register(r"documents", viewsets.DocumentViewSet)
router.register(r"chunks", viewsets.ChunkViewSet)
router.register(
    "events",
    EventsViewSet,
    basename="events1",
)
router.register(r"chat", views.ChatAPIView, basename="chat")
router.register(r"schema/swagger-ui", SpectacularSwaggerView, basename="swagger-ui")
router.register(r"schema", SpectacularAPIView, basename="schema")

urlpatterns = [
    path(
        "", RedirectView.as_view(url="chat/", permanent=True)
    ),  # Redirige vers la page de chat
    path("chat/", views.chat, name="chat"),  # Page de chat
    path("add_file/", views.add_file, name="add_file"),  # Ajouter un fichier
    path(
        "list_documents/", views.list_documents, name="list_documents"
    ),  # Liste des documents / charger les documents qui ne le sont pas encore
    path(
        "delete_document/", views.delete_document, name="delete_document"
    ),  # Supprimer un document
    path(
        "events/",
        include(urls),
        {"channels": ["chat"]},
    ),  # URL pour les événements, chat en temps réel
    path("chunks/", views.ChunkListView.as_view(), name="chunk_list"),
    path("3d_view/", views.view_request_in_3d, name="test"),
    path("api/", include(router.urls)),
]
