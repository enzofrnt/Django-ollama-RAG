from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embedding_function import embed_query
from .models import Chunk


def split_documents(documents: list[Document]):
    """
    Divise les documents en morceaux de taille contrôlée pour l'indexation.

    :param documents: Liste de documents à segmenter.
    :return: Liste de morceaux de texte segmentés.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Taille maximale d'un morceau (en caractères).
        chunk_overlap=200,  # Chevauchement entre les morceaux pour la continuité.
        length_function=len,  # Fonction pour mesurer la longueur des morceaux.
        is_separator_regex=False,  # Indique que le séparateur n'est pas une expression régulière.
    )
    return text_splitter.split_documents(documents)


def add_to_django(chunks: list[Document], document: Document):
    """
    Ajoute les chunks à la base de données en les associant au document fourni.

    :param chunks: Liste des chunks à ajouter.
    :param document: Instance du Document auquel les chunks sont associés.
    """
    for chunk in chunks:
        # Calculer l'embedding pour le contenu du chunk
        embedding = embed_query(chunk.page_content)

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
