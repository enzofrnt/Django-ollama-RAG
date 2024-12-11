# Le RAG

## Prérequis

- Python 3.10
- Ollama installé et disponible
- Docker

## Local

### Installation

1. Clonez le dépôt

2. Créez un environnement virtuel et activez-le :
    ```bash
    python3.10 -m venv env
    source env/bin/activate
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

4. Assurez-vous qu'Ollama est installé et disponible sur votre système.

5. Lancer le conteneur Docker de la base de données postgres (cette base de données est capable de stocker des vecteurs) :
```bash
docker run -d -e POSTGRES_PASSWORD=password -p 5432:5432 pgvector/pgvector:0.8.0-pg17
```

### Lancer l'application 

1. Appliquez les migrations de la base de données :
    ```bash
    python manage.py migrate
    ```

2. Lancez le serveur de développement :
    ```bash
    python manage.py runserver
    ```

3. Accédez à l'application via votre navigateur à l'adresse `http://127.0.0.1:8000`.

## Docker

1. Clonez le dépôt

2. Construisez l'image Docker :
    ```bash
    docker compose up -d
    ```

3. Assurez-vous qu'Ollama est installé et disponible sur votre système. Ajuster si nécessaire l'url de l'API Ollama dans le fichier `.env`.

4. Accédez à l'application via votre navigateur à l'adresse `http://localhost:8000`.

