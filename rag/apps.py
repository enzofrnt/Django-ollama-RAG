from django.apps import AppConfig


class RagConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag"

    def ready(self):
        from .populate_database import (
            display_acp_in_2d,
            display_acp_in_3d,
            display_acp_in_3d_better,
        )

        display_acp_in_3d_better(
            "Qu'est ce que enzo pense des IA dans l'apprentissage ?"
        )
