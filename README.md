# Classification d'Images

## Description du Projet
Cette application web interactive permet de classifier des images à l'aide d'un modèle d'apprentissage profond.  
L'utilisateur peut téléverser un dataset au format ZIP pour entraîner le modèle, visualiser les métriques d'entraînement en temps réel, puis tester la classification sur de nouvelles images.

## Fonctionnalités
- **Téléversement de dataset & Entraînement** : Téléversez un fichier ZIP contenant des dossiers d'images pour entraîner un modèle personnalisé.
- **Visualisation des métriques** : Suivez l'évolution de la précision, de la perte, de la précision et du rappel pendant l'entraînement.
- **Test du modèle** : Téléversez une image (PNG, JPG, JPEG) pour la classifier avec le modèle entraîné.
- **Top 3 prédictions** : Affichage des trois classes les plus probables avec leur score de confiance.
- **Interface intuitive** : Application Streamlit en mode large, facile à utiliser.

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
Image -> Prétraitement -> Modèle -> Vecteur de probabilités -> Classe prédite
```
### Utilisation
1. Installez les dépendances avec `pip install -r requirements.txt`.
2. Lancez l'application avec `streamlit run app.py`.
3. Workflow :
   - Onglet "Entraînement" : Téléversez un fichier ZIP pour entraîner le modèle.
   - Visualisez les métriques en temps réel.
   - Après l'entraînement, passez à l'onglet "Validation" pour voir les résultats finaux.
   - Onglet "Test" : Téléversez une image pour la classifier.

### Structure du Projet
   ```
   image_classifier/
   │
   ├─ .streamlit/
   │   ├─ config.toml          # Configuration de Streamlit
   ├─ models/
   │   ├─ user_dataset.zip
   │   ├─ hybrid_final.keras         # Créé après l'entraînement
   │   ├─ class_indices.pkl          # Créé après l'entraînement
   │   └─ accuracy.png, loss.png, ...
   ├─ src/
   │   ├─ app.py
   │   ├─ config.py
   │   ├─ data_loader.py
   │   ├─ model_builder.py
   │   ├─ streamlit_live_metrics_callback.py
   │   ├─ train.py
   │   └─ __init__.py
   │
   ├─ requirements.txt
   └─ README.md
   ```   