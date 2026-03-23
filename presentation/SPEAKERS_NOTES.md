# Speaker Notes — HR Policy AI Agent Presentation

---

## Slide 1 — HR Policy AI Agent (Title)

"Hello, my name is Mbaka. Today I'll be presenting my take-home project for the AI internship at Talsom — an HR Policy AI Agent built using Retrieval-Augmented Generation. This is a prototype that allows employees to ask questions about internal HR policies and get accurate, cited answers in real time. The full source code is available on GitHub."

---

## Slide 2 — Who Am I?

[À compléter avec ta propre introduction — ~30-60 secondes]

---

## Slide 3 — Contexte & Problématique

"Le problème que ce projet résout est simple : dans une entreprise de 200 employés, les équipes RH répondent souvent aux mêmes questions — congés, avantages sociaux, télétravail. Un LLM seul ne peut pas répondre correctement parce qu'il ne connaît pas vos politiques internes spécifiques. La solution est le RAG : on extrait les documents RH, on les indexe dans une base vectorielle, et on injecte les extraits pertinents dans le prompt avant de générer la réponse. L'agent est donc toujours ancré dans la réalité documentaire de l'entreprise."

---

## Slide 4 — Architecture Générale

"L'architecture suit trois chemins distincts. Avant même d'appeler l'API OpenAI, le classificateur regex analyse la requête. Si elle contient des mots-clés sensibles comme 'harcèlement' ou 'poursuite judiciaire', le système escalade immédiatement vers un humain. Si la question est hors sujet — météo, blague, code informatique — le système redirige poliment. Seulement pour les vraies questions RH, on déclenche le pipeline RAG complet. Ce routage en amont évite des appels API inutiles et garantit un comportement sécuritaire."

---

## Slide 5 — Choix Technologiques

"Pour chaque composant, j'ai choisi la solution la mieux adaptée à un prototype de qualité production. GPT-4o-mini coûte 15 fois moins cher que GPT-4o tout en étant largement suffisant pour de la consultation documentaire. ChromaDB permet de démarrer sans aucune infrastructure — la base vectorielle est simplement un dossier sur le disque. LangChain m'a permis d'assembler le pipeline sans réinventer la roue."

---

## Slide 6 — Pipeline d'Ingestion des Documents

"L'ingestion se fait en une seule commande. Le système charge 31 fichiers Markdown et 2 PDFs couvrant 6 catégories RH, les découpe en 174 morceaux tout en préservant les métadonnées — nom de fichier, catégorie, numéro de page. Ces métadonnées sont cruciales parce qu'elles permettent à l'agent de citer ses sources précisément dans chaque réponse. Le découpage est 'Markdown-aware' : on coupe d'abord aux titres de section, ce qui préserve la cohérence logique des documents."

---

## Slide 7 — Démonstration — 3 Types de Requêtes

"Je vais maintenant vous montrer trois comportements distincts de l'agent. D'abord une question RH normale — l'agent récupère les bons documents, génère une réponse structurée et cite sa source. Ensuite une situation d'escalade — notez que le système n'essaie pas de 'répondre' à une question de harcèlement, il redirige immédiatement vers un humain. C'est un choix de design délibéré : certaines situations ne doivent pas être gérées par un chatbot. Enfin, la question hors sujet — l'agent reconnaît qu'il n'est pas là pour ça et redirige."

---

## Slide 8 — Prompt Système & Gestion des Cas Limites

"Le prompt système est le 'contrat comportemental' de l'agent. Il définit 5 règles strictes que le LLM doit respecter à chaque réponse. La plus importante : ne jamais inventer. Si les documents ne couvrent pas la question, l'agent le dit explicitement plutôt que d'halluciner. Pour les cas limites, la majorité sont gérés proprement. La seule limitation connue est la dérive sémantique — si vous demandez 'mon superviseur se comporte bizarrement', le retriever peut ramener des documents sur le harcèlement parce que les termes sont sémantiquement proches, même si votre intention était de parler de démission."

---

## Slide 9 — Tests & Stratégie d'Évaluation

"L'évaluation suit trois niveaux. Les tests unitaires ne font pas d'appels API — ils utilisent des mocks pour valider la logique de classification et de découpage. Les tests d'intégration, eux, font de vrais appels OpenAI pour valider le pipeline de bout en bout. 67 tests en tout, tous passants. Pour aller plus loin en production, j'utiliserais le framework RAGAS qui permet de mesurer des métriques RAG spécifiques : fidélité de la réponse au contexte, pertinence des chunks récupérés, précision et rappel."

---

## Slide 10 — Limitations Identifiées & Pistes d'Amélioration

"Soyons honnêtes sur les limitations. La plus importante est la dérive sémantique — j'en ai parlé. Pour un déploiement réel, je remplacerais ChromaDB par une solution scalable comme Pinecone ou pgvector intégré à PostgreSQL. Et surtout, j'ajouterais une boucle de feedback : si les employés peuvent noter les réponses, ces données deviennent du gold pour améliorer le retrieval."

---

## Slide 11 — Conclusion

"Pour conclure : en deux jours, j'ai construit un agent RAG complet pour les politiques RH — de l'ingestion des documents jusqu'à l'interface Streamlit, en passant par 67 tests automatisés et une documentation technique bilingue. Le code est entièrement disponible sur GitHub. Ce projet m'a permis de démontrer ma capacité à concevoir et livrer un système IA de qualité production, et j'aimerais beaucoup appliquer ces compétences chez Talsom. Merci pour votre attention — je suis disponible pour vos questions."
