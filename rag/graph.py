import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .embedding_function import embed_query
from .models import Chunk
from .query_data import get_similar_chunks


def generate_3d_figure(
    query_point, similar_points, non_similar_points, similarities, title
):
    """
    Génère une figure 3D Plotly pour un ensemble de points, avec la requête, les similaires et non-similaires.
    """
    fig = go.Figure()

    # Chunks non similaires (en bleu)
    fig.add_trace(
        go.Scatter3d(
            x=non_similar_points[:, 0],
            y=non_similar_points[:, 1],
            z=non_similar_points[:, 2],
            mode="markers",
            marker=dict(color="blue", size=4),
            name="Autres chunks",
        )
    )

    # La requête (en rouge)
    fig.add_trace(
        go.Scatter3d(
            x=query_point[:, 0],
            y=query_point[:, 1],
            z=query_point[:, 2],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Requête",
        )
    )

    # Chunks similaires (dégradé selon la similarité)
    fig.add_trace(
        go.Scatter3d(
            x=similar_points[:, 0],
            y=similar_points[:, 1],
            z=similar_points[:, 2],
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
        title=title,
        scene=dict(xaxis_title="Dim1", yaxis_title="Dim2", zaxis_title="Dim3"),
        legend=dict(x=0, y=1),
    )

    return fig


def display_cos_sim_in_3D(query_text: str, k: int = 5):
    """
    Calcule les projections PCA, t-SNE, UMAP 3D des embeddings, met en avant la requête,
    les chunks similaires et non similaires. Retourne les HTML des 3 graphiques interactifs Plotly,
    ainsi que la liste des meilleurs chunks.

    :param query_text: Texte de la requête utilisateur.
    :param k: Nombre de chunks similaires à récupérer.
    """
    # Récupérer tous les chunks et leurs embeddings
    chunks = list(Chunk.objects.all())
    all_embeddings = [chunk.embedding for chunk in chunks]
    len_all_embeddings = len(all_embeddings)

    if len_all_embeddings == 0:
        return (
            "<h1 style='color:red'>Pas de fichier ajouté à l'app pour le moment</h1>",
            "<h1 style='color:red'>Pas de fichier</h1>",
            "<h1 style='color:red'>Pas de fichier</h1>",
            [],
        )

    # Embedding de la requête
    query_embedding = embed_query(query_text)

    # Conversion en np.array
    all_embeddings_array = np.array(all_embeddings)
    query_embedding_array = np.array([query_embedding])
    all_embeddings_with_query = np.concatenate(
        [all_embeddings_array, query_embedding_array], axis=0
    )

    # Récupérer les chunks similaires
    similar_chunks = get_similar_chunks(query_embedding, len_all_embeddings - 1)
    similar_chunk_ids = {chunk.id for chunk in similar_chunks}
    best_chunks = similar_chunks[:k]

    # Séparer les chunks non similaires
    non_similar_chunks = [c for c in chunks if c.id not in similar_chunk_ids]

    # Embeddings similaires et non similaires en np.array
    embeddings_similar = np.array([c.embedding for c in similar_chunks])
    embeddings_non_similar = np.array([c.embedding for c in non_similar_chunks])
    similarities = [c.similarity for c in similar_chunks]

    # ---- PROJECTION PCA ----
    pca = PCA(n_components=3)
    pca.fit(all_embeddings_with_query)
    reduced_similar_pca = pca.transform(embeddings_similar)
    reduced_non_similar_pca = pca.transform(embeddings_non_similar)
    reduced_query_pca = pca.transform(query_embedding_array)

    fig_pca = generate_3d_figure(
        reduced_query_pca,
        reduced_similar_pca,
        reduced_non_similar_pca,
        similarities,
        "Projection PCA 3D",
    )
    graph_html_pca = fig_pca.to_html(full_html=False)

    # ---- PROJECTION t-SNE ----
    # On reconstruit l'ensemble dans l'ordre: similaires, non similaires, requête
    all_data_tsne = np.concatenate(
        [embeddings_similar, embeddings_non_similar, query_embedding_array], axis=0
    )
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    all_data_tsne = tsne.fit_transform(all_data_tsne)

    reduced_similar_tsne = all_data_tsne[: len(similar_chunks)]
    reduced_non_similar_tsne = all_data_tsne[
        len(similar_chunks) : len(similar_chunks) + len(non_similar_chunks)
    ]
    reduced_query_tsne = all_data_tsne[-1:].reshape(1, 3)

    fig_tsne = generate_3d_figure(
        reduced_query_tsne,
        reduced_similar_tsne,
        reduced_non_similar_tsne,
        similarities,
        "Projection t-SNE 3D",
    )
    graph_html_tsne = fig_tsne.to_html(full_html=False)

    # ---- PROJECTION UMAP ----
    all_data_umap = np.concatenate(
        [embeddings_similar, embeddings_non_similar, query_embedding_array], axis=0
    )
    umap_model = UMAP(n_components=3, random_state=42)
    all_data_umap = umap_model.fit_transform(all_data_umap)

    reduced_similar_umap = all_data_umap[: len(similar_chunks)]
    reduced_non_similar_umap = all_data_umap[
        len(similar_chunks) : len(similar_chunks) + len(non_similar_chunks)
    ]
    reduced_query_umap = all_data_umap[-1:].reshape(1, 3)

    fig_umap = generate_3d_figure(
        reduced_query_umap,
        reduced_similar_umap,
        reduced_non_similar_umap,
        similarities,
        "Projection UMAP 3D",
    )
    graph_html_umap = fig_umap.to_html(full_html=False)

    return graph_html_pca, graph_html_tsne, graph_html_umap, best_chunks
