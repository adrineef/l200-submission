# Règles du Projet — Haiku Agent (ADK Production)

## Authentification
- Pas de clé API : utiliser ADC (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`).
- Les secrets passent par Secret Manager, jamais par le code ni par le dépôt.

## Grille d'évaluation (95 pts — 19 critères × 5 pts)
1. **Docstrings** : Docstrings complètes (Args, Returns, Raises) sur chaque outil, agent et fonction.
2. **Noms d'outils** : Nommage explicite `verbe_nom` en `snake_case`.
3. **Schémas JSON** : Modèles Pydantic (`BaseModel`, `Field`) stricts pour les entrées/sorties structurées (`output_schema`).
4. **Erreurs guidées** : Retours d'erreurs structurés (`status`, `error_code`, `message`, `remediation`) guidant l'auto-correction du LLM.
5. **Constitution** : Charte immuable `<constitution>` dans les instructions système de l'agent.
6. **Compaction** : Configuration de compaction d'événements (`EventsCompactionConfig`) pour résumer l'historique de session.
7. **État persistant** : Gestion de l'état via `VertexAiSessionService`, `ToolContext.state` et `output_key`.
8. **Mémoire asynchrone** : Persistance asynchrone long-terme via `VertexAiMemoryBankService` / `InMemoryMemoryService` (`add_session_to_memory`).
9. **Patterns multi-agents** : Orchestration combinant `SequentialAgent`, `LoopAgent`, `ParallelAgent` et `LlmAgent`.
10. **Routage des modèles** : Sélection dynamique et hiérarchisée entre `gemini-2.5-flash` (tâches rapides/critique) et `gemini-2.5-pro` (composition complexe).
11. **Garde-fous** : Callbacks `before_model_callback`, `after_model_callback`, `before_tool_callback` et `SafetySetting` contre l'injection de prompt et les contenus interdits.
12. **Human-in-the-Loop (HITL)** : Validation humaine via `LongRunningFunctionTool` / `request_input` avant publication critique.
13. **Logs JSON** : Journalisation structurée au format JSON compatible Cloud Logging (`severity`, `trace_id`, `span_id`, `event_type`).
14. **Intention vs Résultat** : Traçabilité explicite comparant l'intention planifiée (`intent`) et le résultat observé (`actual_result` / `outcome`).
15. **Tracing OpenTelemetry** : Instrumentation distribuée avec `TracerProvider` et export vers Cloud Trace (`CloudTraceSpanExporter`).
16. **Masquage des PII** : Anonymisation des données personnelles (emails, téléphones, NIR/SSN, CB) via Cloud DLP API et filtres regex avant appel LLM et dans les logs.
17. **Evals automatisées** : Suite de tests et d'évaluations automatisées (`pytest`, jeux d'évaluation ADK dans `haiku_agent/tests/evals/`).
18. **Terraform** : Infrastructure-as-Code dans `deployment/terraform/` (APIs, Service Accounts, IAM moindre privilège, Secret Manager, Vertex AI / Cloud Run).
19. **Gestion des secrets** : Récupération dynamique depuis Google Cloud Secret Manager sans aucun secret en clair dans le code ou Git.
