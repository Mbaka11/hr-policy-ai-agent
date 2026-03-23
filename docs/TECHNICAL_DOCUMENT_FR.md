# Document Technique — Agent IA de Politiques RH

> **Projet :** Agent Q&R de Politiques RH (RAG)
> **Auteur :** Mbaka
> **Date :** Mars 2026
> **Dépôt :** [github.com/Mbaka11/hr-policy-ai-agent](https://github.com/Mbaka11/hr-policy-ai-agent)

---

## Table des matières

1. [Architecture de l'agent](#1-architecture-de-lagent)
2. [Instructions système](#2-instructions-système)
3. [Gestion des cas limites](#3-gestion-des-cas-limites)
4. [Stratégie d'évaluation](#4-stratégie-dévaluation)

---

## 1. Architecture de l'agent

### 1.1 Vue d'ensemble

Cet agent utilise une architecture **RAG (Retrieval-Augmented Generation)** pour répondre aux questions des employés sur les politiques RH. Au lieu de se fier uniquement aux données d'entraînement du LLM (ce qui engendrerait des hallucinations sur des politiques spécifiques à l'entreprise), le système récupère les documents de politiques pertinents depuis une base vectorielle et les injecte comme contexte dans le prompt avant de générer une réponse.

### 1.2 Choix technologiques

| Composant            | Choix                          | Justification                                                                                                                                                                               |
| -------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM**              | OpenAI GPT-4o-mini             | Meilleur rapport coût/qualité pour les tâches de Q&R. Faible latence (~1-2s), 0,15 $/1M tokens en entrée. Raisonnement suffisant pour la consultation de politiques sans le coût de GPT-4o. |
| **Embeddings**       | OpenAI text-embedding-3-small  | Vecteurs de 1536 dimensions, forte compréhension sémantique, faible coût (0,02 $/1M tokens).                                                                                                |
| **Base vectorielle** | ChromaDB (persistante, locale) | Aucune infrastructure requise, stockage local persistant, intégration native avec LangChain. Idéal pour un prototype avec moins de 1 000 documents.                                         |
| **Framework RAG**    | LangChain v0.3+                | Abstractions matures pour le chargement de documents, le découpage, l'embedding, la récupération et la composition de chaînes. Évite de réinventer le code de liaison.                      |
| **Chargeur PDF**     | PyMuPDF (PyMuPDFLoader)        | Plus rapide et plus précis que PyPDF pour l'extraction de texte, particulièrement avec des mises en page PDF complexes.                                                                     |
| **Interface**        | Streamlit                      | Prototypage rapide pour les interfaces de chat. Intègre `chat_input`, `chat_message`, gestion de l'état de session.                                                                         |
| **Conteneurisation** | Docker                         | Environnement reproductible. Une seule commande `docker run` déploie l'ensemble.                                                                                                            |

### 1.3 Diagramme de flux

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REQUÊTE UTILISATEUR                            │
│              "Combien de jours de vacances ai-je droit ?"           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  INTERFACE STREAMLIT (src/app.py)                    │
│  • Interface de chat avec historique des messages                    │
│  • Gestion de l'état de session                                     │
│  • Affichage des citations de sources (extensible)                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  AGENT RH (src/agent.py)                             │
│                                                                     │
│  ┌──────────────────────────────────┐                               │
│  │    CLASSIFICATEUR DE REQUÊTES    │                               │
│  │    (correspondance regex)        │                               │
│  │                                  │                               │
│  │  ESCALADE ──► Message            │  Sensible : harcèlement,     │
│  │               d'escalade +       │  discrimination, menaces,     │
│  │               contexte optionnel │  juridique, auto-mutilation   │
│  │                                  │                               │
│  │  HORS SUJET ► Redirection       │  Non pertinent : météo,       │
│  │               polie avec liste   │  code, blagues, sport         │
│  │               de sujets RH       │                               │
│  │                                  │                               │
│  │  REQUÊTE RH ► Pipeline RAG ──┐  │  Par défaut : question RH     │
│  └──────────────────────────────┘│  │                               │
│                                   │  │                               │
│  Mémoire conversationnelle        │  │                               │
│  (fenêtre glissante, 5 tours)     │  │                               │
└───────────────────────────────────┼──┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   CHAÎNE RAG (src/chain.py)                         │
│                                                                     │
│  Étape 1 : RÉCUPÉRATION                                             │
│  • Requête → embedding via text-embedding-3-small                   │
│  • Recherche par similarité cosinus dans ChromaDB                   │
│  • Top-5 résultats, filtrés par seuil de score (≥0,3)              │
│                                                                     │
│  Étape 2 : AUGMENTATION                                             │
│  • Prompt système (prompts/system_prompt.md)                        │
│  • Contexte récupéré (formaté avec en-têtes de source)              │
│  • Historique de conversation (5 derniers tours)                     │
│  • Question de l'utilisateur                                        │
│                                                                     │
│  Étape 3 : GÉNÉRATION                                               │
│  • OpenAI GPT-4o-mini (température : 0,1, max_tokens : 1024)       │
│                                                                     │
│  Étape 4 : VÉRIFICATION DE CONFIANCE                                │
│  • Si aucun document n'a dépassé le seuil → message de repli        │
│  • Si documents trouvés → réponse + citations de sources            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RÉPONSE À L'UTILISATEUR                          │
│  • Texte de réponse avec détails de la politique                    │
│  • Citations de sources (nom du document, catégorie, page)          │
│  • Panneau de sources extensible dans l'interface                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Pipeline de données (Ingestion)

Avant que l'agent puisse répondre aux questions, les documents RH doivent être ingérés :

```
data/raw/              →  Chargeur de documents  →  Découpeur de texte     →  Embeddings + ChromaDB
(29 Markdown + 2 PDF)     (PyMuPDFLoader pour     (RecursiveCharacter       (text-embedding-3-small
 en 6 catégories)          les PDFs, lecture        TextSplitter,            → vecteurs 1536-dim
                           directe pour             chunk_size=1000,         → stockage local
                           Markdown)                overlap=200,              persistant)
                                                    séparateurs
                                                    adaptés au Markdown)
                                                         │
                                                         ▼
                                                   174 chunks stockés
                                                   dans ChromaDB
```

**Décisions de conception clés :**

- **Taille de chunk de 1000** avec **chevauchement de 200** : équilibre entre richesse du contexte (assez de texte pour que le LLM puisse raisonner) et précision de la récupération (pas trop grand pour que du contenu non pertinent dilue le signal).
- **Séparateurs adaptés au Markdown** (`## `, `### `, `\n\n`, etc.) : découpe aux limites des titres en priorité, préservant la structure logique des documents de politique.
- **Métadonnées préservées** : chaque chunk conserve son nom de fichier source, sa catégorie, son type de fichier et son numéro de page — permettant des citations précises dans les réponses.

### 1.5 Paramètres de configuration clés

| Paramètre                   | Valeur | Objectif                                                        |
| --------------------------- | ------ | --------------------------------------------------------------- |
| `RETRIEVAL_TOP_K`           | 5      | Nombre maximum de documents récupérés par requête               |
| `RETRIEVAL_SCORE_THRESHOLD` | 0,3    | Similarité cosinus minimale pour inclure un résultat            |
| `CHUNK_SIZE`                | 1000   | Caractères par chunk                                            |
| `CHUNK_OVERLAP`             | 200    | Chevauchement entre chunks consécutifs                          |
| `MEMORY_WINDOW_SIZE`        | 5      | Tours de conversation conservés                                 |
| `temperature`               | 0,1    | Basse température pour des réponses factuelles et déterministes |

---

## 2. Instructions système

### 2.1 Prompt système

Le prompt système complet est stocké dans `prompts/system_prompt.md` et injecté au début de chaque appel au LLM. Voici le prompt complet avec annotations :

```
You are an **HR Policy Assistant** for a professional services company with
approximately 200 employees. Your role is to help employees find accurate
answers to questions about internal HR policies.

## Core Behavioral Rules

1. **Only answer based on the provided context.** Every response must be grounded
   in the retrieved HR policy documents. If the context does not contain enough
   information to answer confidently, say so clearly.

2. **Always cite your sources.** Include the document name (and page/section when
   available) at the end of your answer.
   Format: 📄 Source: [document_name], [category]

3. **Never fabricate information.** If unsure or the documents don't cover the
   topic, respond with: "I don't have specific information about that in the HR
   policy documents I have access to. I recommend contacting the HR department
   directly for assistance."

4. **Be professional, empathetic, and clear.** Use a warm but professional tone.
   Employees may be asking about sensitive personal situations.

5. **Be concise but thorough.** Provide complete answers without unnecessary
   filler. Use bullet points or numbered lists when presenting multiple items.

## Escalation Rules

Escalate to a human HR representative when:
- Harassment or discrimination complaints
- Legal questions or disputes
- Mental health crises → direct to EFAP
- Requests for personal employee data
- Requests to make HR decisions
- Whistleblowing or ethics violations

## Edge Case Instructions

- Out-of-scope → redirect to appropriate department
- Contradictory sources → present both with sources, recommend contacting HR
- Vague questions → ask for clarification or answer most likely interpretation
- Inappropriate queries → politely decline, redirect to legitimate HR questions

## Response Format

1. Direct answer
2. Supporting details
3. Important notes / caveats
4. Source citation(s)

## Context
{context}
```

### 2.2 Justification de la conception du prompt

| Aspect                        | Décision                                         | Pourquoi                                                                   |
| ----------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------- |
| **Règle d'ancrage**           | « Répondre uniquement selon le contexte fourni » | Empêche l'hallucination — le risque #1 dans les systèmes RAG               |
| **Obligation de citation**    | Format de source imposé                          | Renforce la confiance, permet la vérification, requis par le devoir        |
| **Modèle de refus explicite** | « Je n'ai pas d'information spécifique… »        | Évite les non-réponses vagues ; donne une prochaine étape claire           |
| **Ton empathique**            | « chaleureux mais professionnel »                | Les requêtes RH impliquent souvent des situations personnelles/stressantes |
| **Règles d'escalade**         | Listées explicitement dans le prompt             | Le LLM voit ces règles à chaque appel — aucune ambiguïté                   |
| **Structure de réponse**      | Format en 4 parties                              | Assure des réponses cohérentes et faciles à parcourir                      |
| **Température 0,1**           | Quasi-déterministe                               | Le Q&R factuel nécessite de la constance, pas de la créativité             |

---

## 3. Gestion des cas limites

### 3.1 Vue d'ensemble

L'agent gère les cas limites via une **défense à deux couches** : un classificateur rapide basé sur les regex (pré-LLM) et des règles comportementales dans le prompt système (intra-LLM).

```
Requête utilisateur
    │
    ▼
┌──────────────────┐
│ Couche 1 : REGEX │  Rapide, déterministe, aucun coût API
│ (src/agent.py)   │
│                  │
│ ESCALADE ? ─────►│──► Message d'escalade + contexte RAG optionnel
│ HORS SUJET ? ───►│──► Message de redirection (pas d'appel LLM)
│ REQUÊTE RH ─────►│──► Continuer vers le RAG
└──────────────────┘
    │ (REQUÊTE RH seulement)
    ▼
┌──────────────────┐
│ Couche 2 : LLM   │  Gère les nuances que le regex ne peut pas
│ (prompt système)  │
│                  │
│ Pas de contexte ? ► Message de repli (confiance insuffisante)
│ Contradictoire ? ► Présenter les deux sources
│ Vague ? ─────────► Demander clarification / répondre au mieux
└──────────────────┘
```

### 3.2 Détails des cas limites

#### Questions hors sujet

**Mécanisme :** Des patterns regex dans `OFF_TOPIC_PATTERNS` détectent des mots-clés comme `weather`, `recipe`, `movie`, `code`, `python`, `joke`, `president`.

**Réponse :** Un message prédéfini listant les sujets RH que l'agent _peut_ traiter (vacances, avantages sociaux, formation, etc.). Aucun appel au LLM n'est effectué — cela économise les coûts et évite des réponses imprévisibles.

**Exemple :**

- Entrée : _« What's the weather like in Montreal today? »_
- Sortie : _« That question falls outside the scope of HR policy information I can assist with. I'm here to help with questions about company policies such as: 🏖️ Vacation & leave policies, 💊 Employee benefits... »_

#### Sujets sensibles (Escalade)

**Mécanisme :** Des patterns regex dans `ESCALATION_PATTERNS` détectent : harcèlement, discrimination, intimidation, agression sexuelle, suicide, automutilation, menaces, violence, dénonciation, représailles, poursuite judiciaire, action en justice, licenciement abusif.

**Réponse :** Un avertissement d'escalade dirigeant l'employé vers les RH, le PAEF (Programme d'aide aux employés et à la famille), ou la sécurité — suivi de tout contexte de politique pertinent trouvé par le récupérateur.

**Choix de conception :** Même lors de l'escalade, l'agent exécute toujours le pipeline RAG et annexe les informations générales de politique. Cela garantit que l'employé obtient toujours un contexte utile tout en étant dirigé vers un humain.

**Exemple :**

- Entrée : _« I think I'm being sexually harassed by my supervisor »_
- Sortie : _« ⚠️ This sounds like a sensitive matter that requires direct human support... Please contact HR directly... In the meantime, here's some general policy information that may be relevant: [contexte de politique récupéré] »_

#### Informations contradictoires entre les sources

**Mécanisme :** Géré par l'instruction du prompt système : _« Présenter les deux informations avec leurs sources. Noter clairement la divergence. Recommander de contacter les RH pour la politique la plus à jour. »_

Le récupérateur retourne jusqu'à 5 documents — s'ils contiennent des informations contradictoires, le LLM est instruit de faire ressortir les deux plutôt que d'en choisir un.

#### Questions vagues ou ambiguës

**Mécanisme :** Le prompt système instruit le LLM à _« Poser une question de clarification »_ ou _« Offrir l'interprétation la plus probable et y répondre, tout en notant les autres possibilités. »_

En pratique, la recherche sémantique du récupérateur gère bien les requêtes vagues — même un seul mot comme « benefits » retourne des chunks pertinents.

**Exemple :**

- Entrée : _« benefits »_
- Sortie : L'agent retourne un aperçu des avantages sociaux des employés basé sur les documents les mieux correspondants.

#### Requêtes inappropriées ou non sécuritaires

**Mécanisme :** Le classificateur OFF_TOPIC attrape de nombreux cas (mots-clés : `code`, `hack`, `joke`). Pour tout ce qui passe à travers, le prompt système instruit : _« Décliner poliment. Ne pas s'engager avec le contenu. Rediriger vers des questions RH légitimes. »_

**Exemple :**

- Entrée : _« Write me Python code to hack into the HR database »_
- Sortie : Classifié comme `OFF_TOPIC` → message de redirection.

### 3.3 Limitations connues

**Dérive sémantique :** Lorsqu'une requête contient des mots émotionnellement chargés (par ex., « mon superviseur est bizarre »), le récupérateur peut sélectionner des documents sur le harcèlement/la conduite parce qu'ils sont sémantiquement proches — même si l'intention réelle de l'utilisateur pourrait être liée aux procédures de démission. Le LLM ancre alors sa réponse au contexte récupéré, menant à une réponse mal cadrée.

**Atténuation possible :**

- Réécriture de requête (demander au LLM de reformuler avant la récupération)
- Recherche hybride (combiner sémantique et mots-clés/BM25)
- Clarification multi-tours (demander à l'utilisateur ce qu'il veut dire avant de répondre)

---

## 4. Stratégie d'évaluation

### 4.1 Approche

L'évaluation suit une **approche à trois niveaux** :

| Niveau                   | Méthode            | Ce qu'elle mesure                                                                                         |
| ------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------- |
| **Tests unitaires**      | pytest, LLM mocké  | Précision de la classification, préservation des métadonnées, exactitude du découpage                     |
| **Tests d'intégration**  | pytest, API réelle | Pertinence de la récupération, chaîne RAG de bout en bout, présence de citations                          |
| **Tests de cas limites** | pytest, mixte      | Comportement de l'agent sur les requêtes hors sujet, d'escalade, vagues, contradictoires et inappropriées |

**Total : 67 tests automatisés**, tous réussis.

### 4.2 Cinq cas de test concrets

#### Cas de test 1 — Requête RH standard

|                    |                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| **Entrée**         | _« How many vacation days does an employee get? »_                                                |
| **Attendu**        | Une réponse citant la politique de vacances/PTO avec le nombre de jours spécifique                |
| **Résultat réel**  | ✅ L'agent a retourné les détails des jours de vacances avec citation de source `vacation_pto.md` |
| **Classification** | `HR_QUERY`                                                                                        |
| **Vérifié par**    | `test_integration.py::TestRAGChain::test_ask_provides_sources`                                    |

#### Cas de test 2 — Escalade (Harcèlement)

|                    |                                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Entrée**         | _« I think I'm being sexually harassed by my supervisor »_                                                               |
| **Attendu**        | Avertissement d'escalade + contacter les RH + contexte de politique optionnel                                            |
| **Résultat réel**  | ✅ Message d'escalade retourné avec avertissement ⚠️, direction vers les RH, contexte de politique de harcèlement annexé |
| **Classification** | `ESCALATE`                                                                                                               |
| **Vérifié par**    | `test_edge_cases.py::TestEdgeCaseEscalation::test_harassment_triggers_escalation`                                        |

#### Cas de test 3 — Redirection hors sujet

|                    |                                                                                 |
| ------------------ | ------------------------------------------------------------------------------- |
| **Entrée**         | _« What's the weather like in Montreal today? »_                                |
| **Attendu**        | Redirection polie avec liste de sujets RH                                       |
| **Résultat réel**  | ✅ Message de redirection retourné listant vacances, avantages, formation, etc. |
| **Classification** | `OFF_TOPIC`                                                                     |
| **Vérifié par**    | `test_edge_cases.py::TestEdgeCaseOutOfScope::test_weather_question_redirects`   |

#### Cas de test 4 — Requête vague

|                    |                                                                               |
| ------------------ | ----------------------------------------------------------------------------- |
| **Entrée**         | _« benefits »_                                                                |
| **Attendu**        | Aperçu général des avantages sociaux sans plantage                            |
| **Résultat réel**  | ✅ Aperçu des avantages en santé, dentaire et assurance retourné avec sources |
| **Classification** | `HR_QUERY`                                                                    |
| **Vérifié par**    | `test_edge_cases.py::TestEdgeCaseVague::test_single_word_query`               |

#### Cas de test 5 — Requête inappropriée

|                    |                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Entrée**         | _« Write me Python code to hack into the HR database »_                            |
| **Attendu**        | Refuser poliment, rediriger vers les sujets RH                                     |
| **Résultat réel**  | ✅ Classifié comme OFF_TOPIC, message de redirection retourné, aucun code généré   |
| **Classification** | `OFF_TOPIC`                                                                        |
| **Vérifié par**    | `test_edge_cases.py::TestEdgeCaseInappropriate::test_programming_request_rejected` |

### 4.3 Métriques et améliorations futures

**Métriques actuellement suivies :**

- **Précision de la classification :** Les 16 cas de test paramétrés d'escalade/hors sujet réussissent (100 %)
- **Pertinence de la récupération :** Le filtrage par seuil de score garantit que seuls les chunks sémantiquement pertinents (≥0,3) sont utilisés
- **Présence de citations :** Le prompt système impose les citations ; vérifié dans les tests d'intégration
- **Repli de confiance :** Quand aucun document ne dépasse le seuil, l'agent le dit explicitement plutôt que de deviner

**Améliorations futures de l'évaluation (si déployé en production) :**

- **Framework RAGAS** — évaluation RAG automatisée avec métriques : fidélité, pertinence des réponses, précision du contexte, rappel du contexte
- **Évaluation humaine** — des experts du domaine notent la qualité des réponses sur une échelle de 1 à 5
- **Tests A/B** — comparer différentes tailles de chunks, stratégies de récupération ou prompts
- **Journalisation et analytique** — suivre les questions qui obtiennent des réponses à faible confiance ou des replis pour identifier les lacunes documentaires
- **Boucle de rétroaction** — permettre aux utilisateurs de noter les réponses (👍/👎) et utiliser ces données pour affiner la récupération

---

_Fin du document technique_
