# Classification d'Images

## Description du Projet
Cette application web interactive permet de classifier des images à l'aide d'un modèle d'apprentissage profond.  
L'utilisateur peut téléverser un dataset au format ZIP pour entraîner le modèle, visualiser les métriques d'entraînement en temps réel, puis tester la classification sur de nouvelles images.

## Fonctionnalités
- **Téléversement de dataset & Entraînement** : Téléversez un fichier ZIP contenant des dossiers d'images pour entraîner un modèle personnalisé.
- **Visualisation des métriques** : Suivez l'évolution de l'exactitude, de la perte, de la précision et du rappel pendant l'entraînement (affichage en temps réel via Streamlit).
- **Test du modèle** : Téléversez une image (PNG, JPG, JPEG) pour la classifier avec le modèle entraîné.
- **Top 3 prédictions** : Affichage des trois classes les plus probables avec leur score de confiance.
- **Rapport de classification** : Affichage détaillé des métriques par classe (précision, rappel, f1-score, support).
- **Matrice de confusion** : Visualisation compacte dans l’interface.

## Architecture du Modèle
Le modèle est une architecture hybride de deep learning, entraînée sur les datasets fournis par l'utilisateur.

**Étapes principales :**
1. **Prétraitement** : Redimensionnement et normalisation des images.
2. **Entraînement** : Le modèle apprend à partir du dataset téléversé.
3. **Prédiction** : Le modèle retourne un vecteur de probabilités pour chaque classe.
4. **Mapping** : Les indices de classes sont convertis en labels lisibles.

### Schéma de l'Architecture
```plaintext
ZIP d'images -> Entraînement -> Modèle (.keras) + Mapping (.pkl)
Image -> Prétraitement -> Modèle hybride -> Vecteur de probabilités -> Classe prédite
```
### Utilisation
1. Installez les dépendances avec `pip install -r requirements.txt`.
2. Lancez l'application avec `streamlit run app.py` en vous assurant que vous êtes dans le répertoire `image_classifier`.
3. Workflow :
   - Onglet "Entraînement" : Téléversez un fichier ZIP pour entraîner le modèle. Les images doivent être organisées en sous-dossiers par classe. Ce qui permet de détecter automatiquement les classes.
   - Visualisez les métriques en temps réel.
   - Après l'entraînement, passez à l'onglet "Validation" pour voir les résultats finaux.
   - Onglet "Test" : Téléversez une image pour la classifier.

### Structure du Projet
   ```
image_classifier/
│
├─ .streamlit/
│   └─ config.toml                # Configuration de Streamlit
├─ models/    # généré avec l'entraînement
│   ├─ user_dataset.zip           # Dernier dataset téléversé
│   ├─ hybrid_final.keras         # Modèle final entraîné
│   ├─ best_hybrid.keras          # Meilleur modèle (checkpoint)
│   ├─ class_indices.pkl          # Mapping des classes
│   └─ accuracy.png, loss.png,... # Graphiques des métriques
├─ src/
│   ├─ app.py                     # Application Streamlit principale
│   ├─ config.py                  # Paramètres globaux
│   ├─ data_loader.py             # Chargement et prétraitement des données
│   ├─ f1score.py                 # Métrique F1 personnalisée
│   ├─ model_builder.py           # Construction du modèle hybride
│   ├─ streamlit_live_metrics_callback.py # Callback pour affichage live
│   ├─ train.py                   # Fonction d'entraînement
│   └─ __init__.py
├─ requirements.txt
└─ README.md
   ```   