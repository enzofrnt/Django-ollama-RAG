import logging
import mimetypes

from django_filters.rest_framework import DjangoFilterBackend
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from rest_framework import status, viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import Chunk, Document
from .populate_database import add_to_django, split_documents
from .serializers import ChunkSerializer, DocumentSerializer

logger = logging.getLogger(__name__)


# ONly delete, get, post, head and options are allowed
class DocumentViewSet(
    viewsets.mixins.CreateModelMixin,
    viewsets.mixins.DestroyModelMixin,
    viewsets.mixins.ListModelMixin,
    viewsets.mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("file")
        print(uploaded_file)
        if not uploaded_file:
            return Response(
                {"error": "Aucun fichier envoyé"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Types de fichiers pris en charge
        supported_types = {
            "application/pdf": PyPDFLoader,
            "text/plain": TextLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": UnstructuredWordDocumentLoader,
            "text/markdown": TextLoader,
            "text/x-markdown": TextLoader,
            "text/x-wiki": TextLoader,
        }

        # Liste des documents créés, si besoin de retour plus détaillé
        created_docs = []

        # Détecter le type MIME
        file_type, encoding = mimetypes.guess_type(uploaded_file.name)
        if file_type not in supported_types:
            return Response(
                {
                    "error": f"Type de fichier '{file_type}' non pris en charge pour '{uploaded_file.name}'"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validation via le serializer
        serializer = self.get_serializer(data={"file": uploaded_file})
        serializer.is_valid(raise_exception=True)

        # Création du Document
        document = serializer.save()
        logger.info(f"✅ Fichier '{document.file.name}' sauvegardé.")

        # Charger et traiter le document
        try:
            loader_class = supported_types[file_type]
            loader = loader_class(document.file.path)
            pages = loader.load()
        except Exception as e:
            document.delete()
            logger.error(
                f"❌ Erreur de chargement du fichier '{document.file.name}': {str(e)}"
            )
            return Response(
                {"error": f"Erreur de traitement du fichier '{uploaded_file.name}'"},
                status=status.HTTP_400_BAD_REQUEST,
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
            return Response(
                {"error": f"Erreur de segmentation du fichier '{uploaded_file.name}'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        created_docs.append(document)

        return Response(
            DocumentSerializer(created_docs, many=True).data,
            status=status.HTTP_201_CREATED,
        )


class ChunkViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Chunk.objects.all()
    serializer_class = ChunkSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["document"]
