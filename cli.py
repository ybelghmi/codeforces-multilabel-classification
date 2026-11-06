"""Utiliser le classifier sauvegardé depuis le terminal.

Les chemins par défaut sont relatifs au projet, quel que soit le dossier courant.
Évaluation : python -m cli evaluer (JSONL annoté).
Prédictions : python -m cli tester exemple ou python -m cli tester ensemble (JSON).
"""

import argparse
import html
import json
import sys
import unicodedata
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from threadpoolctl import threadpool_limits


# Construire les chemins depuis cli.py et fixer l’ordre des huit catégories.
RACINE = Path(__file__).resolve().parent
TAGS = [
    "math", "graphs", "strings", "number theory",
    "trees", "geometry", "games", "probabilities",
]


# Remplacer les erreurs techniques de syntaxe par le rappel des commandes valides.
class AnalyseurCommandes(argparse.ArgumentParser):
    def error(self, message):
        self.exit(
            2,
            "\nErreur : Commande invalide. Utilisez l’une des commandes suivantes :\n"
            "  python -m cli evaluer\n"
            "  python -m cli tester exemple\n"
            "  python -m cli tester ensemble\n\n",
        )


# Recharger le TF-IDF, les SVM et les seuils sauvegardés, sans réentraînement.
def charger_modele(chemin):
    modele = joblib.load(chemin)
    # Vérifier les composants, l’ordre des catégories et les huit seuils.
    requis = {"vectoriseur", "classifieur", "seuils", "tags", "champs_texte"}
    if not isinstance(modele, dict) or not requis.issubset(modele):
        raise ValueError("L’artefact ne contient pas tous les composants du classifier.")
    if [tag.replace("_", " ") for tag in modele["tags"]] != TAGS:
        raise ValueError("Les huit catégories du modèle ne correspondent pas à l’ordre attendu.")
    seuils = np.asarray(modele["seuils"], dtype=float)
    if seuils.shape != (8,) or not np.isfinite(seuils).all():
        raise ValueError("Le modèle doit contenir huit seuils finis.")
    return modele


# Lire le jeu annoté au format JSONL : un exercice par ligne.
def charger_donnees(chemin):
    exercices = []
    with chemin.open(encoding="utf-8") as fichier:
        for numero, ligne in enumerate(fichier, 1):
            if not ligne.strip():
                continue
            try:
                exercice = json.loads(ligne)
            except json.JSONDecodeError as erreur:
                raise ValueError(f"JSON invalide à la ligne {numero} : {erreur.msg}.") from erreur
            if not isinstance(exercice, dict):
                raise ValueError(f"La ligne {numero} doit contenir un objet JSON.")
            exercices.append(exercice)
    if not exercices:
        raise ValueError("Le jeu d’évaluation est vide.")

    # Contrôler les huit cibles binaires et les identifiants avant l’évaluation.
    donnees = pd.DataFrame(exercices)
    colonnes = ["cible_" + tag.replace(" ", "_") for tag in TAGS]
    manquantes = set(colonnes) - set(donnees.columns)
    if manquantes:
        raise ValueError("Cibles manquantes : " + ", ".join(sorted(manquantes)) + ".")
    if not donnees[colonnes].isin([0, 1]).all().all():
        raise ValueError("Les huit cibles doivent toutes être renseignées avec 0 ou 1.")
    if "fichier_source" in donnees and not donnees["fichier_source"].is_unique:
        raise ValueError("Des identifiants fichier_source sont répétés.")
    return exercices, donnees[colonnes].to_numpy(dtype=np.int64)


# Préparer un énoncé et lui attribuer zéro, un ou plusieurs tags cibles.
def predire_exercice(exercice, modele):
    # Reproduire le prétraitement appris : sections titrées, entités HTML et Unicode NFC.
    sections = []
    for champ, titre in modele["champs_texte"].items():
        valeur = exercice.get(champ)
        if isinstance(valeur, str):
            contenu = unicodedata.normalize("NFC", html.unescape(valeur)).strip()
            if contenu:
                sections.append(f"{titre}\n{contenu}")
    texte = "\n\n".join(sections)
    if not texte.strip():
        raise ValueError(
            "L’énoncé est vide ou absent. Ajoutez du texte dans au moins un des champs "
            "prob_desc_description, prob_desc_input_spec, prob_desc_output_spec ou prob_desc_notes."
        )
    # Utiliser le vocabulaire et les SVM existants, sans appeler fit.
    matrice = modele["vectoriseur"].transform([texte])
    scores = modele["classifieur"].decision_function(matrice)[0]
    if np.shape(scores) != (8,) or not np.isfinite(scores).all():
        raise ValueError("Le classifier doit produire huit scores finis.")
    # Chaque tag possède son propre seuil ; aucun tag n’est imposé.
    return (scores >= np.asarray(modele["seuils"])).astype(np.int64)


# Prédire les JSON fournis dans test/exemple ou test/ensemble.
def tester(modele, dossier, exemple_unique):
    if not dossier.is_dir():
        raise ValueError(f"Le dossier {dossier} n’existe pas.")
    # Lire uniquement les JSON directement présents, triés par nom de fichier.
    fichiers = sorted(
        (p for p in dossier.iterdir() if p.is_file() and p.suffix.lower() == ".json"),
        key=lambda p: p.name,
    )
    # Exiger un seul JSON pour exemple et au moins un pour ensemble.
    if exemple_unique and not fichiers:
        raise ValueError(
            "Aucun fichier JSON dans test/exemple. Ajoutez un seul et unique fichier JSON contenant l’exemple à prédire."
        )
    if exemple_unique and len(fichiers) != 1:
        raise ValueError(
            f"test/exemple contient {len(fichiers)} fichiers JSON. "
            "Conservez un seul et unique fichier JSON contenant l’exemple à prédire."
        )
    if not fichiers:
        raise ValueError(
            "Aucun fichier JSON dans test/ensemble. Ajoutez un ou plusieurs fichiers JSON contenant les exemples à prédire."
        )

    # Conserver les mêmes prédictions pour l’affichage et les éventuelles métriques.
    lignes, predictions, cibles, durees = [], [], [], []
    # Limiter les calculs numériques à un thread pour mesurer la latence.
    with threadpool_limits(limits=1):
        for chemin in fichiers:
            try:
                exercice = json.loads(chemin.read_text(encoding="utf-8"))
            except json.JSONDecodeError as erreur:
                raise ValueError(f"JSON invalide dans {chemin.name} : {erreur.msg}.") from erreur
            if not isinstance(exercice, dict):
                raise ValueError(f"{chemin.name} doit contenir un objet JSON représentant un exercice.")
            try:
                # Pour un ensemble, effectuer trois échauffements avant le chronométrage.
                if len(fichiers) > 1 and not predictions:
                    for _ in range(3):
                        predire_exercice(exercice, modele)
                # Mesurer la préparation du texte, le TF-IDF, les SVM et les seuils.
                debut = perf_counter()
                prediction = predire_exercice(exercice, modele)
                durees.append(1000 * (perf_counter() - debut))
            except ValueError as erreur:
                raise ValueError(f"Impossible de prédire les tags de {chemin.name}. {erreur}") from erreur
            predictions.append(prediction)
            # Les annotations servent uniquement à la comparaison, jamais à la prédiction.
            tags_predits = [tag for tag, actif in zip(TAGS, prediction) if actif]
            annotations = exercice.get("tags")
            # Une liste vide signifie aucun tag cible ; un champ absent reste inconnu.
            if isinstance(annotations, list):
                tags_attendus = [tag for tag in TAGS if tag in annotations]
                cibles.append([int(tag in annotations) for tag in TAGS])
                texte_attendu = ", ".join(tags_attendus) if tags_attendus else "Aucun tag cible"
            else:
                texte_attendu = "Non renseignés"
            lignes.append({
                "Fichier": chemin.name,
                "Tags attendus": texte_attendu,
                "Tags prédits": ", ".join(tags_predits) if tags_predits else "Aucun tag cible",
            })
    # Afficher les résultats seulement lorsque tous les fichiers ont été traités.
    if len(fichiers) == 1:
        print("\nPrédiction sur 1 exercice\n")
        print(f"Fichier : {lignes[0]['Fichier']}")
        print(f"Tags attendus : {lignes[0]['Tags attendus']}")
        print(f"Tags prédits : {lignes[0]['Tags prédits']}\n")
    else:
        print(f"\nPrédiction sur {len(fichiers)} exercices")
        # Calculer les métriques uniquement si chaque exercice possède ses annotations.
        if len(cibles) == len(fichiers):
            globales, par_tag, durees, corrects = calculer_metriques(
                np.asarray(cibles), np.stack(predictions), np.asarray(durees)
            )
            afficher_metriques(globales, par_tag, durees, corrects)
        else:
            print("\nMétriques non calculées. Les tags attendus doivent être renseignés "
                  "dans le champ tags de chaque fichier JSON.")
        print("\nRésultats par exercice :")
        print(pd.DataFrame(lignes).to_string(index=False))
        print()


# Évaluer tous les exercices du jeu annoté avec le modèle sauvegardé.
def evaluer(modele, exercices, cibles):
    predictions, durees = [], []
    with threadpool_limits(limits=1):
        # Exclure les trois inférences d’échauffement des durées mesurées.
        for exercice in exercices[:3]:
            predire_exercice(exercice, modele)
        for numero, exercice in enumerate(exercices, 1):
            debut = perf_counter()
            try:
                prediction = predire_exercice(exercice, modele)
            except ValueError as erreur:
                raise ValueError(f"Exercice {numero} : {erreur}") from erreur
            durees.append(1000 * (perf_counter() - debut))
            predictions.append(prediction)
    predictions = np.stack(predictions)
    durees = np.asarray(durees)
    return calculer_metriques(cibles, predictions, durees)


# Partager le calcul des métriques entre evaluer et tester ensemble.
def calculer_metriques(cibles, predictions, durees):
    globales = pd.DataFrame([
        ("F1 macro", f1_score(cibles, predictions, average="macro", zero_division=0)),
        ("F1 micro", f1_score(cibles, predictions, average="micro", zero_division=0)),
        ("Exact Match Ratio", accuracy_score(cibles, predictions)),
        ("Hamming Loss", hamming_loss(cibles, predictions)),
        ("Inférence médiane (ms/exercice)", float(np.median(durees))),
    ], columns=["Métrique", "Valeur"])
    # Détailler les performances et le nombre de positifs réels pour chaque tag.
    precisions, rappels, f1, supports = precision_recall_fscore_support(
        cibles, predictions, average=None, zero_division=0,
    )
    par_tag = pd.DataFrame({
        "Tag": TAGS, "Support": supports.astype(int),
        "Précision": precisions, "Rappel": rappels, "F1": f1,
    })
    # Un exercice est entièrement correct si ses huit décisions sont exactes.
    corrects = int((predictions == cibles).all(axis=1).sum())
    return globales, par_tag, durees, corrects


# Ajouter le titre de l’évaluation et les lignes vides autour des résultats.
def afficher_resultats(globales, par_tag, durees, corrects):
    nombre = len(durees)
    print(f"\nÉvaluation sur {nombre} exercice{'s' if nombre > 1 else ''}")
    afficher_metriques(globales, par_tag, durees, corrects)
    print()


# Afficher les tableaux dans le terminal, sans créer de fichier de résultats.
def afficher_metriques(globales, par_tag, durees, corrects):
    print("\nMétriques globales :")
    print(globales.to_string(index=False, justify="left", formatters={
        "Métrique": "{:<30}".format, "Valeur": "{:.4f}".format,
    }))
    print("\nRésultats par catégorie :")
    print(par_tag.to_string(index=False, formatters={
        nom: "{:.4f}".format for nom in ["Précision", "Rappel", "F1"]
    }))
    print(f"\nExercices entièrement corrects : {corrects}/{len(durees)}")


# Interpréter la commande du terminal et exécuter l’action demandée.
def main(argv=None):
    analyseur = AnalyseurCommandes(
        description="Utiliser le classifier sauvegardé.", add_help=False,
    )
    commandes = analyseur.add_subparsers(dest="commande", required=True)
    # evaluer utilise par défaut le classifier final et data/test.jsonl.
    evaluation = commandes.add_parser(
        "evaluer", help="Afficher les résultats sur un jeu annoté JSONL.",
        description="Évaluer le modèle et ses seuils sauvegardés, sans réentraînement ni export.",
        add_help=False,
    )
    evaluation.add_argument(
        "--donnees", type=Path, default=RACINE / "data" / "test.jsonl",
        help="Jeu annoté JSONL (par défaut : data/test.jsonl).",
    )
    evaluation.add_argument(
        "--modele", type=Path, default=RACINE / "classifier" / "resultats" / "modele.joblib",
        help="Artefact sauvegardé (par défaut : classifier/resultats/modele.joblib).",
    )
    # tester sélectionne le dossier exemple ou ensemble ; l’aide est désactivée.
    prediction = commandes.add_parser(
        "tester", help="Prédire les tags des JSON de test/exemple ou test/ensemble.",
        description="Tester des exercices JSON sans exiger leurs annotations.", add_help=False,
    )
    prediction.add_argument(
        "cible", choices=["exemple", "ensemble"],
        help="exemple : un seul JSON dans test/exemple ; ensemble : tous les JSON de test/ensemble.",
    )
    prediction.add_argument(
        "--modele", type=Path, default=RACINE / "classifier" / "resultats" / "modele.joblib",
        help="Artefact sauvegardé (par défaut : classifier/resultats/modele.joblib).",
    )
    # Valider les arguments, puis charger le modèle une seule fois par commande.
    arguments = analyseur.parse_args(argv)
    try:
        modele = charger_modele(arguments.modele)
        if arguments.commande == "tester":
            tester(modele, RACINE / "test" / arguments.cible, arguments.cible == "exemple")
            return 0
        exercices, cibles = charger_donnees(arguments.donnees)
        globales, par_tag, durees, corrects = evaluer(modele, exercices, cibles)
    # Présenter les erreurs attendues sans traceback, avec un code de sortie de 1.
    except (OSError, ValueError, EOFError, ImportError) as erreur:
        print(f"\nErreur : {erreur}\n", file=sys.stderr)
        return 1
    afficher_resultats(globales, par_tag, durees, corrects)
    return 0


# Lancer le programme avec python -m cli et transmettre son code de sortie au terminal.
if __name__ == "__main__":
    raise SystemExit(main())
