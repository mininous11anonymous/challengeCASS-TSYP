import os
import json
from dotenv import load_dotenv

# Importations CrewAI nécessaires
# Note: Dans un environnement réel, les fonctions de prévision (Transformer) et 
# d'accès à la BESS (Firestore, API) seraient implémentées comme des outils Python.
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import Tool, SerperDevTool

# Charger les variables d'environnement (API key, etc.)
load_dotenv()

# --- 1. Initialisation du LLM et des Outils ---

try:
    # Utilisation du modèle Gemini comme cerveau des agents
    # La clé API est lue depuis le fichier .env
    gemini_llm = LLM(
        model=os.getenv("MODEL", "gemini/gemini-2.5-flash"),
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3 # Température plus basse pour des décisions plus fiables
    )
except Exception as e:
    print(f"Erreur lors de l'initialisation de LLM : {e}")
    print("Veuillez vérifier que 'GEMINI_API_KEY' est défini dans votre fichier .env.")
    exit()


# Outil de recherche pour simuler l'accès aux données externes (météo, prix)
# Il est crucial pour le 'grounding' de la prise de décision.
# NOTE: Remplacez par votre propre clé Serper si nécessaire, ou utilisez un outil interne
# pour lire des données réelles de l'API Météo et du marché de l'énergie.
data_search_tool = SerperDevTool()

# --- 2. Définition des Agents ---

# Agent 1: Récupère et analyse les données contextuelles
data_analyst_agent = Agent(
    role='Analyste de Données d\'Énergie',
    goal='Collecter et structurer les données historiques, météorologiques, et tarifaires pour la prévision de consommation.',
    backstory=(
        'Expert en préparation de jeux de données de séries temporelles (30 jours) pour les modèles Transformer. '
        'Assure que l\'Agent Prévisionniste a des entrées propres et complètes (ex: prix ToU, jours fériés, météo).'
    ),
    llm=gemini_llm,
    tools=[data_search_tool],
    verbose=True,
    allow_delegation=False
)

# Agent 2: Exécute le modèle de prévision Transformer
# Dans la réalité, cet agent exécuterait le modèle Python/PyTorch de l'utilisateur.
# Ici, il simule le résultat de cette exécution.
forecaster_agent = Agent(
    role='Prévisionniste de Charge Transformer (Modèle T+24h)',
    goal='Produire une prévision horaire précise de la consommation d\'énergie (kWh) pour les 24 heures suivantes.',
    backstory=(
        'Un spécialiste de l\'exécution du modèle Transformer personnalisé (architecture 30d x 3f) '
        'qui excelle à capturer les cycles hebdomadaires et les anomalies pour un forecast T+24h.'
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

# Agent 3: Prend la décision finale de Charge/Décharge (l'Agent d'Optimisation)
optimization_agent = Agent(
    role='Optimiseur BESS Temps Réel',
    goal='Déterminer l\'action optimale (CHARGE, DÉCHARGE, IDLE) pour le Switcher BESS pour maximiser les économies.',
    backstory=(
        'Le cerveau du système BESS. Il utilise la prévision de consommation, les prix de l\'énergie en temps réel, '
        'et l\'état actuel de la batterie (SoC) pour prendre la décision la plus rentable à chaque instant. '
        'Sa priorité est d\'éviter l\'achat d\'énergie aux heures de pointe coûteuses.'
    ),
    llm=gemini_llm,
    tools=[data_search_tool], # Peut utiliser la recherche pour les prix de dernière minute
    verbose=True,
    allow_delegation=False
)


# --- 3. Définition des Tâches ---

# Tâche 1: Préparer l'input pour le modèle de prévision
# Simule l'étape de Feature Engineering
task_data_preparation = Task(
    description=(
        "Collecter les données nécessaires pour la prévision de demain : "
        "1. Trouver la prévision météo pour la région de Tunis pour demain (ensoleillement et température). "
        "2. Déterminer les heures de pointe (Peak Hours) et d'heures creuses (Off-Peak Hours) pour l'électricité en Tunisie. "
        "3. Structurer ces données de contexte pour l'Agent Prévisionniste."
    ),
    expected_output='Un objet JSON contenant "meteo", "tarification" (avec les heures de pointe) et une note sur les "cycles hebdomadaires" à considérer.',
    agent=data_analyst_agent
)

# Tâche 2: Exécuter la prévision
# Simule l'exécution du modèle Transformer
task_forecasting = Task(
    description=(
        "En utilisant l'input de l'Analyste de Données, simuler la prévision de consommation horaire (kWh) pour les 24 heures de demain (T+1). "
        "La prévision doit être un tableau de 24 valeurs, représentant la demande horaire attendue. "
        "Par exemple: [0.5, 0.4, 0.4, ..., 1.2, 2.5, 3.1, 4.0, 3.5, ...]"
    ),
    expected_output='Une liste Python de 24 chiffres (kWh) représentant la consommation prévue de T+1 (Heure 0 à Heure 23).',
    context=[task_data_preparation], # Dépend de la préparation des données
    agent=forecaster_agent
)

# Tâche 3: Prendre la décision d'optimisation (le résultat final)
task_optimization = Task(
    description=(
        "En utilisant la prévision horaire (Forecast) et sachant que l'état actuel de la BESS (SoC) est de 75%, "
        "déterminez l'action optimale à prendre pour l'heure actuelle (Maintenant: supposons qu'il est 17h00). "
        "La décision doit être prise pour l'heure en cours, en tenant compte des heures de pointe trouvées précédemment. "
        "Produire un résumé de la décision et l'action recommandée : CHARGE, DÉCHARGE ou IDLE."
    ),
    expected_output=(
        "Un court rapport expliquant la logique (ex: 'Pointe tarifaire imminente, SoC suffisant') "
        "suivi de l'action unique recommandée: 'ACTION: DÉCHARGE'."
    ),
    context=[task_forecasting], # Dépend du forecast
    agent=optimization_agent
)

# --- 4. Création et Lancement de l'Équipage (Crew) ---

# Le processus est séquentiel (les tâches s'enchaînent)
bess_crew = Crew(
    agents=[data_analyst_agent, forecaster_agent, optimization_agent],
    tasks=[task_data_preparation, task_forecasting, task_optimization],
    process=Process.sequential,
    verbose=2, # Niveau de verbosité plus élevé pour voir les étapes intermédiaires
)

print("🚀 Lancement du Système Agentique BESS (Prévision et Optimisation)...")
result = bess_crew.kickoff()

print("\n\n################################################")
print("## Résultat de la Tâche d'Optimisation Finale")
print("################################################")
print(result)