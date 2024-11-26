from django.contrib import admin

from .models import Chunk, Embedding


@admin.register(Embedding)
class EmbeddingAdmin(admin.ModelAdmin):
    list_display = ("file_name", "id")  # Colonnes visibles dans la liste
    search_fields = ("file_name", "content")  # Champs pour la barre de recherche
    readonly_fields = ("embedding",)  # Rendre les embeddings non modifiables


# @admin.register(Chunk)
# class ChunkAdmin(admin.ModelAdmin):
#     list_display = ("source", "page", "chunk_index")  # Colonnes visibles dans la liste
#     search_fields = ("source", "content")  # Champs pour la barre de recherche
#     readonly_fields = ("embedding",)  # Rendre les embeddings non modifiables


# from django.contrib import admin

# from .models import Document


# @admin.register(Document)
# class DocumentAdmin(admin.ModelAdmin):
#     list_display = ("file", "uploaded_at")
