from django.db import models
from pgvector.django import VectorField


class Embedding(models.Model):
    file_name = models.CharField(max_length=255)
    content = models.TextField()
    embedding = VectorField(dimensions=768)  # Mise à jour à 768

    def __str__(self):
        return self.file_name


from django.db import models
from pgvector.django import IvfflatIndex, VectorField


class Document(models.Model):
    file = models.FileField(upload_to="documents/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name.split("/")[-1]

    def delete(self, *args, **kwargs):
        # Vérifier si un fichier est associé
        if self.file:
            # Supprimer le fichier du système de fichiers
            self.file.delete(save=False)
        # Appeler la méthode delete() du parent pour supprimer l'instance
        super().delete(*args, **kwargs)


class Chunk(models.Model):
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    page = models.IntegerField()
    chunk_index = models.IntegerField()
    content = models.TextField()
    embedding = VectorField(dimensions=768)

    class Meta:
        indexes = [
            IvfflatIndex(
                name="embedding_cosine_idx",
                fields=["embedding"],
                lists=100,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"{self.document.file.name} - Page {self.page}, Chunk {self.chunk_index}"
