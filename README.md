# Classification multi-tag d’exercices Codeforces

L’objectif de ce projet est d’identifier automatiquement les catégories auxquelles appartient un exercice Codeforces à partir de son contenu. Huit catégories sont étudiées : **math, graphs, strings, number theory, trees, geometry, games et probabilities**. Chaque exercice peut relever de plusieurs de ces catégories, d’une seule ou d’aucune. Il s’agit donc d’un problème de classification multi-étiquette.

## 1. Installation

Télécharger et décompresser l’[archive ZIP complète — modèles inclus](https://github.com/ybelghmi/codeforces-multilabel-classification/releases/download/v1.0.0/codeforces-multilabel-classification.zip), puis ouvrir un terminal à sa racine, dans le dossier contenant `cli.py`.
Installer **Python 3.11.x** ; la version vérifiée est **3.11.11**, indiquée dans `.python-version`.

L’archive de la Release contient tous les poids. Les gros modèles sont stockés avec Git LFS dans le dépôt : l’archive standard « Code → Download ZIP » peut ne contenir que leurs pointeurs. Pour télécharger le projet sans Git LFS, utiliser le lien ci-dessus.

### Créer l’environnement du projet

**macOS / Linux**

```bash
python3.11 -m venv --prompt code-classification .venv
source .venv/bin/activate
python --version
```

**Windows — PowerShell**

```powershell
py -3.11 -m venv --prompt code-classification .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Garder l’environnement activé pour les commandes suivantes. Si l’activation est bloquée sous PowerShell, utiliser `.\.venv\Scripts\python.exe` à la place de `python`. Sous macOS / Linux, l’équivalent sans activation est `.venv/bin/python`.

### Installer les dépendances

**Pour utiliser uniquement le modèle final et les commandes CLI :**

```bash
python -m pip install -r requirements.txt
python -m pip check
```

**Pour exécuter également tous les notebooks d'analyse et d'entraînement :**

```bash
python -m pip install -r requirements-notebooks.txt
python -m pip check
python -m ipykernel install --sys-prefix --name code-classification --display-name "Code Classification (.venv)"
```

L’installation complète permet aussi d’utiliser la CLI. Vérifier que `pip check` affiche bien `No broken requirements found.` Pour les notebooks, choisir **Code Classification (.venv)** dans JupyterLab ou l’interpréteur `.venv` dans VS Code, puis exécuter les cellules des notebooks dans l’ordre, depuis la première. Conserver les versions des dépendances fournies.

## 2. Structure du projet

```text
.
├── code_classification_dataset/         # Exercices JSON bruts
├── data/
│   ├── entrainement.jsonl               # Ensemble d’entraînement
│   ├── validation.jsonl                 # Ensemble de validation
│   ├── test.jsonl                       # Ensemble de test final
│   ├── exclusions.csv                   # Exercices exclus du split
│   └── partitions.csv                   # Affectation des exercices aux ensembles
├── notebooks/                           # Exploration, préparation et évaluations
│   ├── analyse_exploratoire.ipynb       # Exploration des données brutes
│   ├── preparation_donnees.ipynb        # Nettoyage et partitionnement
│   ├── evaluation_modeles.ipynb         # Comparaison de M0 à M11 sur validation
│   ├── comparaison_m2_classifier.ipynb  # Comparaison de M2 et du modèle optimisé
│   └── evaluation_test.ipynb            # Évaluation finale sur le test
├── models/                              # Expériences M0 à M11 et modèles sauvegardés
├── classifier/
│   ├── entrainement.ipynb               # Optimisation du modèle retenu
│   └── resultats/                       # Modèle final, métadonnées et résultats CSV
├── test/
│   ├── exemple/                         # Un seul JSON à prédire
│   └── ensemble/                        # Un ou plusieurs JSON à prédire
├── cli.py                               # Commandes de prédiction et d’évaluation
├── requirements.txt                     # Dépendances de la CLI du projet
├── requirements-notebooks.txt           # Dépendances complètes du projet
├── .python-version                      # Version de Python de référence
├── .gitignore                           # Fichiers locaux exclus de Git
└── README.md                            # Présentation et mode d’emploi
```

## 3. Démarche générale

### 3.1. Explorer les données

Le notebook [analyse_exploratoire.ipynb](notebooks/analyse_exploratoire.ipynb) examine les **4 982 exercices** : annotations, champs disponibles, fréquences des tags, associations et descriptions proches. Cette analyse permet de comprendre le déséquilibre des catégories et le caractère multi-étiquette du problème.

### 3.2. Nettoyer et partitionner

Le notebook [preparation_donnees.ipynb](notebooks/preparation_donnees.ipynb) exclut les 41 exercices sans annotation, déduplique les tags et construit les huit cibles binaires. Les exercices annotés uniquement avec d’autres tags restent des négatifs pour les huit cibles. Le partitionnement stratifié multi-étiquette vise une répartition **70% entraînement / 15% validation / 15% test**. Avec des graines fixes, il produit 3 470 exercices d’entraînement, 739 de validation et 732 de test, sans chevauchement des fichiers source. Les proportions obtenues sont approximatives pour préserver la répartition des tags. 

### 3.3. Entraîner les approches candidates

Les notebooks de [models/](models/) comparent les représentations suivantes :

| Modèles | Représentation | Classifieur |
|---|---|---|
| **M0** | Word TF-IDF de l’énoncé | Naïve Bayes |
| **M1 / M2** | Word TF-IDF de l’énoncé | Régression logistique / SVM linéaire |
| **M3 / M4** | MiniLM figé sur l’énoncé | Régression logistique / SVM linéaire |
| **M5 / M6** | TF-IDF du code | Régression logistique / SVM linéaire |
| **M7 / M8** | UniXcoder figé sur le code | Régression logistique / SVM linéaire |
| **M9 / M10**| Fusion des TF-IDF de l’énoncé et du code | Régression logistique / SVM linéaire |
| **M11** | BERT-mini ajusté sur l’énoncé et le code | Tête de classification multi-étiquette |

Le vocabulaire TF-IDF et les classifieurs sont appris sur l’entraînement. Les hyperparamètres et les seuils sont réglés sur la validation. Chaque tag possède son seuil de décision ; aucun tag n’est imposé.

### 3.4. Comparer et sélectionner

Le notebook [evaluation_modeles.ipynb](notebooks/evaluation_modeles.ipynb) recharge les douze modèles et les évalue sur toute la validation, sans réentraînement. Il inclut aussi les références « aucun tag » et « tous les tags ». Le F1 macro est le critère principal. M10, M9 et M2 surpassent les autres modèles mais obtiennent des scores proches. **M2 est retenu pour sa simplicité** : l’énoncé seul suffit, sans traitement du code source. Il obtient aussi le meilleur Exact Match Ratio parmi ces trois modèles. Ces écarts sur validation ne constituent pas une preuve de meilleure généralisation.

### 3.5. Optimiser le classifier retenu

Le notebook [classifier/entrainement.ipynb](classifier/entrainement.ipynb) reprend l’approche M2 et **approfondit la recherche des paramètres** du TF-IDF, du SVM et des huit seuils sur validation. La configuration retenue utilise les unigrammes et bigrammes, `min_df=4`, un plafond de vocabulaire de `50 000`, et un SVM `C=0.5` sans pondération des classes. Le notebook [comparaison_m2_classifier.ipynb](notebooks/comparaison_m2_classifier.ipynb) compare le classifier optimisé à M2, globalement et par tag.

### 3.6. Évaluer le modèle final

Une fois le classifier et ses seuils figés, le notebook [evaluation_test.ipynb](notebooks/evaluation_test.ipynb) les recharge et prédit les **732 exercices du test**. Il affiche les tags attendus et prédits, puis les métriques globales et par catégorie. Aucun réglage n’est effectué à partir du test.

## 4. Métriques d’évaluation

Le F1 macro est le critère principal pour sélectionner et comparer les modèles. La validation sert aux réglages ; le test sert à l’évaluation finale du classifier figé.

### Métriques globales

| Métrique | Ce qu’elle mesure | Objectif |
|---|---|---|
| **F1 macro** | Moyenne des F1 des huit tags, avec le même poids pour chaque catégorie. | Maximiser ↑ |
| **F1 micro** | F1 calculé en regroupant les vrais positifs, faux positifs et faux négatifs des huit tags. | Maximiser ↑ |
| **Exact Match Ratio** | Proportion d’exercices pour lesquels les huit tags sont exactement corrects. | Maximiser ↑ |
| **Hamming Loss** | Proportion de décisions binaires incorrectes parmi tous les exercices et tags. | Minimiser ↓ |
| **Inférence médiane** | Temps médian pour prédire un exercice, en millisecondes, préparation du texte incluse. | Minimiser ↓ |

### Métriques par catégorie

| Métrique | Ce qu’elle mesure |
|---|---|
| **Support** | Nombre d’exercices qui possèdent réellement ce tag. |
| **Précision** | Parmi les exercices auxquels le modèle attribue ce tag, proportion qui le possèdent réellement. |
| **Rappel** | Parmi les exercices qui possèdent réellement ce tag, proportion retrouvée par le modèle. |
| **F1** | Équilibre entre la précision et le rappel pour ce tag. |

La précision, le rappel et le F1 sont à maximiser. Le support aide à interpréter les scores, notamment pour les catégories rares. Les scores vont de 0 à 1 ; le support est un effectif et le temps d’inférence une durée.

## 5. Résultats finaux

Le classifier figé est évalué sur les 732 exercices du test.

### Métriques globales

| F1 macro ↑ | F1 micro ↑ | Exact Match Ratio ↑ | Hamming Loss ↓ | Inférence médiane ↓ (ms/exercice) |
|---:|---:|---:|---:|---:|
| **0,6672** | **0,6146** | **0,5601** | **0,0675** | **0,5477** |

**410 exercices sur 732 sont entièrement corrects.**

### Résultats par catégorie

| Tag | Précision ↑ | Rappel ↑ | F1 ↑ | Support |
|---|---:|---:|---:|---:|
| math | 0,5647 | 0,6209 | 0,5914 | 211 |
| graphs | 0,5857 | 0,5062 | 0,5430 | 81 |
| strings | 0,6282 | 0,7778 | 0,6950 | 63 |
| number theory | 0,5000 | 0,3962 | 0,4421 | 53 |
| trees | 0,8049 | 0,6735 | 0,7333 | 49 |
| geometry | 0,7083 | 0,6800 | 0,6939 | 25 |
| games | 0,8667 | 0,8125 | 0,8387 | 16 |
| probabilities | 0,9091 | 0,7143 | 0,8000 | 14 |

## 6. Tester depuis le terminal

Ces commandes utilisent directement [`classifier/resultats/modele.joblib`](classifier/resultats/modele.joblib), sans réentraînement ni téléchargement de Transformer. Les lancer depuis la racine du dépôt, avec l’environnement activé.

### Évaluer le test fourni

```bash
python -m cli evaluer
```

Recharge le classifier, **prédit tout l'ensemble de test** (`data/test.jsonl`) et affiche les métriques globales, les résultats par catégorie et le nombre d’exercices entièrement corrects.

### Prédire un exemple

Placer **exactement un fichier JSON** dans `test/exemple`, puis lancer :

```bash
python -m cli tester exemple
```

Exemple d’affichage :

```text
Prédiction sur 1 exercice

Fichier : sample_0.json
Tags attendus : math, strings
Tags prédits : math, strings
```

### Prédire un ensemble d’exemples

Placer **un ou plusieurs fichiers JSON** dans `test/ensemble`, puis lancer :

```bash
python -m cli tester ensemble
```

Avec plusieurs fichiers, les métriques précèdent le tableau des prédictions si chaque exercice possède une liste `tags`. Avec un seul fichier, l’affichage reprend celui d’un exemple individuel.

### Format des fichiers et comportement

Les JSON suivent le format de [code_classification_dataset/](code_classification_dataset/). Au moins un champ doit contenir du texte parmi `prob_desc_description`, `prob_desc_input_spec`, `prob_desc_output_spec` et `prob_desc_notes`. Le code source n’est pas utilisé par le classifier final.

Les `tags` sont facultatifs pour prédire. S’ils sont absents, les tags attendus indiquent `Non renseignés` et les métriques de l’ensemble ne sont pas calculées. Une liste vide correspond à aucun tag cible attendu. Seules les huit catégories cibles sont comparées.
