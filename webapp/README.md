# Atelier Haïku — Application Web (`webapp/`)

Interface web conviviale (FastAPI + HTML/CSS/JS sans framework) pour le système multi-agents ADK `haiku_agent`.

## Fonctionnalités
- **Flux SSE temps réel (`POST /api/haiku`)** : création d'une session ADK, progression étape par étape (`{step, label}`), ping SSE toutes les 10 s contre les coupures proxy, puis résultat final (`{haiku, kigo, image_url}`). Seul le texte du *publisher* est affiché.
- **Validation humaine HITL (`POST /api/approve`)** : approbation ou refus de publication après l'interruption `request_human_publication_approval`.
- **Illustration Ukiyo-e (`GET /api/image/{session_id}`)** : sert l'artefact `haiku_illustration.png` généré pour la session.
- **Interface `webapp/static/index.html`** : saisie de la ville, choix de la saison (`auto` par défaut), barre de progression par étapes, carte résultat (haïku centré en 3 lignes, kigo surligné, badge « 5-7-5 validé », estampe ukiyo-e), boutons Approuver/Refuser, Copier et Télécharger l'image.

## Lancement

Depuis la racine du projet :

```bash
uvicorn webapp.server:app --port 8080
```

Ouvrez ensuite [http://localhost:8080](http://localhost:8080) dans votre navigateur.

## Tests unitaires (hors-ligne, sans appel Vertex AI)

```bash
pytest webapp/tests/test_webapp.py
```
