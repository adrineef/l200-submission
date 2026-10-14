# Haiku Agent & Application Web

Système multi-agents ADK de composition de haïkus 5-7-5 avec illustration ukiyo-e et validation humaine (HITL).

## Lancer l'application web

```bash
uvicorn webapp.server:app --port 8080
```

Rendez-vous ensuite sur [http://localhost:8080](http://localhost:8080).

## Lancer les tests

```bash
pytest webapp/tests/test_webapp.py
```
