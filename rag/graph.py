import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from .embedding_function import embed_query
from .models import Chunk
from .query_data import get_similar_chunks


def display_cos_sim_in_3D(
    query_text: str,
    k: int = 5,
):
    """
    Calcule la projection PCA 3D des embeddings, met en avant la requête,
    les chunks similaires et non similaires. Retourne une version HTML
    d'un graphique interactif Plotly.

    :param query_text: Texte de la requête utilisateur.
    :param k: Nombre de chunks similaires à récupérer.
    """
    # Récupérer tous les chunks et leurs embeddings
    chunks = list(Chunk.objects.all())
    all_embeddings = [chunk.embedding for chunk in chunks]
    len_all_embeddings = len(all_embeddings)

    # Embedding de la requête
    query_embedding = embed_query(query_text)
    all_embeddings.append(query_embedding)

    # Récupérer les chunks similaires
    similar_chunks = get_similar_chunks(query_embedding, len_all_embeddings - 1)
    similar_chunk_ids = {chunk.id for chunk in similar_chunks}
    best_chunks = similar_chunks[:k]

    # Séparer les chunks non similaires
    non_similar_chunks = [c for c in chunks if c.id not in similar_chunk_ids]

    # Embeddings similaires et non similaires
    embeddings_similar = [c.embedding for c in similar_chunks]
    embeddings_non_similar = [c.embedding for c in non_similar_chunks]

    # PCA sur l'ensemble
    pca = PCA(n_components=3)
    pca.fit(all_embeddings)

    reduced_similar = pca.transform(embeddings_similar)
    reduced_non_similar = pca.transform(embeddings_non_similar)
    reduced_query = pca.transform([query_embedding])

    # Similarités
    similarities = [c.similarity for c in similar_chunks]

    # Création de la figure Plotly
    fig = go.Figure()

    # Chunks non similaires (en bleu)
    fig.add_trace(
        go.Scatter3d(
            x=reduced_non_similar[:, 0],
            y=reduced_non_similar[:, 1],
            z=reduced_non_similar[:, 2],
            mode="markers",
            marker=dict(color="blue", size=4),
            name="Autres chunks",
        )
    )

    # La requête (en rouge)
    fig.add_trace(
        go.Scatter3d(
            x=reduced_query[:, 0],
            y=reduced_query[:, 1],
            z=reduced_query[:, 2],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Requête",
        )
    )

    # Chunks similaires (dégradé selon la similarité)
    fig.add_trace(
        go.Scatter3d(
            x=reduced_similar[:, 0],
            y=reduced_similar[:, 1],
            z=reduced_similar[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=similarities,
                colorscale="Viridis",
                cmin=min(similarities),
                cmax=max(similarities),
                colorbar=dict(title="Sim cosinus - Higher is better"),
            ),
            name="Chunks similaires",
        )
    )

    fig.update_layout(
        title="Projection PCA 3D - Interactif",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(x=0, y=1),
    )
    # Retourne le code HTML du graphique
    return fig.to_html(full_html=False), best_chunks
