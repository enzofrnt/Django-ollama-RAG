from rest_framework import serializers

from .models import Chunk, Document


class DocumentSerializer(serializers.ModelSerializer):
    file = serializers.FileField(required=True)

    class Meta:
        model = Document
        fields = ["id", "file", "uploaded_at"]


class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chunk
        fields = "__all__"
