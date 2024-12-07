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
        chunk_size=800,  # Taille maximale d'un morceau (en caractères).
        chunk_overlap=80,  # Chevauchement entre les morceaux pour la continuité.
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


import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from .query_data import get_similar_chunks


def display_acp_in_2d(query_text: str):
    """
    use plot to display a graph of every chunk in the database usging ACP
    it display the query in red in the graph
    """
    chunks = Chunk.objects.all()
    pca = PCA(n_components=2)
    embeded_query = embed_query(query_text)

    # get the embedding of every chunk
    embeddings = [chunk.embedding for chunk in chunks]

    reduced_embeddings_pca = pca.fit_transform(embeddings)
    reduced_query_pca = pca.transform([embeded_query])

    # plot the reduced embeddings
    plt.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1])
    plt.scatter(reduced_query_pca[:, 0], reduced_query_pca[:, 1], color="red")
    plt.show()


def display_acp_in_3d(query_text: str):
    """
    Affiche un nuage de points 3D des embeddings de tous les chunks en base.
    Les chunks similaires à la requête sont coloriés selon un dégradé de couleurs
    (du plus proche au moins proche). La requête est en rouge.
    Une légende et une barre de couleur sont également affichées.
    """
    chunks = list(Chunk.objects.all())
    print(len(chunks))
    embeded_query = embed_query(query_text)
    similar_chunks = get_similar_chunks(embeded_query, 410)

    # On récupère les IDs des chunks similaires
    similar_chunk_ids = {chunk.id for chunk in similar_chunks}

    # On sépare les chunks non similaires
    non_similar_chunks = [
        chunk for chunk in chunks if chunk.id not in similar_chunk_ids
    ]

    # On extrait les embeddings
    embeddings_non_similar = [chunk.embedding for chunk in non_similar_chunks]
    embeddings_similar = [chunk.embedding for chunk in similar_chunks]

    # On récupère les valeurs de similarité (distance cosinus) des chunks similaires
    # On part du principe que similar_chunks contient l'attribut "similarity"
    similarities = [chunk.similarity for chunk in similar_chunks]

    # PCA en 3D
    pca = PCA(n_components=3)
    # Nous devons fit_transform sur les embeddings non similaires, puis transformer les similaires et la requête
    reduced_embeddings_pca = pca.fit_transform(embeddings_non_similar)
    reduced_similar_chunks_pca = pca.transform(embeddings_similar)
    reduced_query_pca = pca.transform([embeded_query])

    # Création de la figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter pour les chunks non similaires (en bleu)
    ax.scatter(
        reduced_embeddings_pca[:, 0],
        reduced_embeddings_pca[:, 1],
        reduced_embeddings_pca[:, 2],
        c="blue",
        label="Autres chunks",
    )

    # Scatter pour la requête (en rouge)
    ax.scatter(
        reduced_query_pca[:, 0],
        reduced_query_pca[:, 1],
        reduced_query_pca[:, 2],
        color="red",
        label="Requête",
    )

    # On normalise les similarités pour la colormap
    norm = mpl.colors.Normalize(vmin=min(similarities), vmax=max(similarities))
    cmap = plt.cm.viridis  # ou une autre colormap de votre choix

    # Scatter pour les chunks similaires, avec un dégradé de couleurs
    sc = ax.scatter(
        reduced_similar_chunks_pca[:, 0],
        reduced_similar_chunks_pca[:, 1],
        reduced_similar_chunks_pca[:, 2],
        c=similarities,
        cmap=cmap,
        label="Chunks similaires",
    )

    # Ajout d'une colorbar pour montrer la correspondance des couleurs avec la distance
    cb = fig.colorbar(sc, ax=ax, label="Distance cosinus")

    # Affichage de la légende
    ax.legend()

    plt.show()


def display_acp_in_3d_better(query_text: str):
    """
    Affiche un nuage de points 3D des embeddings de tous les chunks en base.
    Les chunks similaires à la requête (top 200) sont coloriés selon un dégradé,
    la requête est en rouge, et les autres chunks en bleu.
    Une légende et une barre de couleur sont également affichées.
    Le PCA est entraîné sur l'ensemble des données avant de projeter les sous-ensembles.
    """
    # Récupérer tous les chunks et leurs embeddings
    chunks = list(Chunk.objects.all())
    all_embeddings = [chunk.embedding for chunk in chunks]

    # Embedding de la requête
    query_embedding = embed_query(query_text)

    # Récupérer les chunks similaires (par exemple top 200)
    similar_chunks = get_similar_chunks(query_embedding, 430)
    similar_chunk_ids = {chunk.id for chunk in similar_chunks}
    first_chunk = similar_chunks[0]
    print("first chunk")
    print(first_chunk.similarity)
    print(first_chunk.content)
    second_chunk = similar_chunks[1]
    print("second chunk")
    print(second_chunk.similarity)
    print(second_chunk.content)

    print("last chunk")
    last_chunk = similar_chunks[429]
    print(last_chunk.similarity)
    print(last_chunk.content)

    # Séparer les chunks non similaires
    non_similar_chunks = [c for c in chunks if c.id not in similar_chunk_ids]

    # Embeddings similaires et non similaires
    embeddings_similar = [c.embedding for c in similar_chunks]
    embeddings_non_similar = [c.embedding for c in non_similar_chunks]

    # On ajuste le PCA sur l'ensemble complet des données
    pca = PCA(n_components=3)
    pca.fit(all_embeddings)

    # On projette maintenant chaque sous-ensemble avecloins  le même PCA
    reduced_similar = pca.transform(embeddings_similar)
    reduced_non_similar = pca.transform(embeddings_non_similar)
    reduced_query = pca.transform([query_embedding])

    # On récupère les similarités pour les chunks similaires
    similarities = [c.similarity for c in similar_chunks]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Tracé des chunks non similaires (en bleu)
    ax.scatter(
        reduced_non_similar[:, 0],
        reduced_non_similar[:, 1],
        reduced_non_similar[:, 2],
        c="blue",
        label="Autres chunks",
    )

    # Tracé de la requête (en rouge)
    ax.scatter(
        reduced_query[:, 0],
        reduced_query[:, 1],
        reduced_query[:, 2],
        color="red",
        label="Requête",
    )

    # On normalise les similarités pour la colormap
    norm = mpl.colors.Normalize(vmin=min(similarities), vmax=max(similarities))
    cmap = plt.cm.viridis

    # Tracé des chunks similaires (dégradé en fonction de la similarité)
    sc = ax.scatter(
        reduced_similar[:, 0],
        reduced_similar[:, 1],
        reduced_similar[:, 2],
        c=similarities,
        cmap=cmap,
        norm=norm,
        label="Chunks similaires",
    )

    # Ajout d'une colorbar pour montrer la distribution des similarités
    cb = fig.colorbar(sc, ax=ax, label="Distance cosinus")

    # Affichage de la légende
    ax.legend()

    plt.show()
