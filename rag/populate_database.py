import argparse
import os
import shutil

from django.conf import settings
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .get_embedding_function import get_embedding_function


def populate_database():
    """
    Charge les documents, les segmente en morceaux et les ajoute à la base de données Chroma.
    """
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_django(chunks)


def reset_database():
    """
    Réinitialise complètement la base de données en supprimant tout son contenu.
    """
    clear_database()


def load_documents():
    """
    Charge les documents PDF depuis un répertoire spécifié dans les paramètres Django.

    :return: Liste de documents chargés.
    """
    document_loader = PyPDFDirectoryLoader(settings.DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """
    Divise les documents en morceaux de taille contrôlée pour l'indexation.

    :param documents: Liste de documents à segmenter.
    :return: Liste de morceaux de texte segmentés.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Taille maximale d'un morceau (en caractères).
        chunk_overlap=80,  # Chevauchement entre les morceaux pour la continuité.
        length_function=len,  # Fonction pour mesurer la longueur des morceaux.
        is_separator_regex=False,  # Indique que le séparateur n'est pas une expression régulière.
    )
    return text_splitter.split_documents(documents)


from django.db.utils import IntegrityError

from .models import Chunk


def add_to_django(chunks: list[Document], document: Document):
    """
    Ajoute les chunks à la base de données en les associant au document fourni.

    :param chunks: Liste des chunks à ajouter.
    :param document: Instance du Document auquel les chunks sont associés.
    """
    embedding_function = get_embedding_function()
    for chunk in chunks:
        # Calculer l'embedding pour le contenu
        embedding = embedding_function.embed_query(chunk.page_content)

        # Extraire les métadonnées
        page = int(chunk.metadata.get("page", 0))
        chunk_index = int(chunk.metadata.get("id", "0").split(":")[-1])

        # Créer et sauvegarder l'objet dans la base
        Chunk.objects.create(
            document=document,
            page=page,
            chunk_index=chunk_index,
            content=chunk.page_content,
            embedding=embedding,
        )


def calculate_chunk_ids(chunks):
    """
    Calcule des identifiants uniques pour chaque morceau basé sur la source, la page et l'index.

    :param chunks: Liste de morceaux de texte.
    :return: Liste de morceaux avec des identifiants ajoutés.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get(
            "source"
        )  # Source du document (par exemple, le nom du fichier PDF).
        page = chunk.metadata.get("page")  # Numéro de la page.
        current_page_id = f"{source}:{page}"  # ID unique pour la page (source:page).

        # Incrémente l'index si le morceau appartient à la même page que le précédent.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Génère un ID unique pour le morceau.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Ajoute l'ID au metadata du morceau.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Supprime complètement le contenu de la base de données Chroma en effaçant le dossier correspondant.

    :return: None
    """
    if os.path.exists(settings.CHROMA_PATH):
        shutil.rmtree(settings.CHROMA_PATH)


from langchain_community.document_loaders import PyPDFLoader

from .models import Document


def load_documents_from_files():
    """
    Charge les documents à partir des fichiers enregistrés via le modèle Document.
    """
    documents = []
    # Récupérer tous les documents
    for doc in Document.objects.all():
        file_path = doc.file.path
        # Charger le document
        loader = PyPDFLoader(file_path)
        doc_pages = loader.load()
        # Ajouter la source dans les métadonnées
        for page in doc_pages:
            page.metadata["source"] = doc.file.name
        documents.extend(doc_pages)
    return documents
