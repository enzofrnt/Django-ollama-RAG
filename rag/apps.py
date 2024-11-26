from django.apps import AppConfig


class RagConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag"

    def ready(self):
        """
        Chargement des documents dans la base de données au démarrage de l'application.
        """
        from .populate_database import populate_database

        # populate_database()
