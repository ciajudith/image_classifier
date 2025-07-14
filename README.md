# Classification d'Images d'Animaux

## Description du Projet
Ce projet est une application web interactive permettant de classifier des images d'animaux en utilisant un modèle d'apprentissage profond. L'utilisateur peut téléverser une image, et l'application prédit la classe de l'animal avec un score de confiance. Les classes possibles sont affichées dynamiquement sur le site.

## Fonctionnalités
- **Téléversement d'images** : L'utilisateur peut téléverser une image au format PNG, JPG ou JPEG.
- **Prédiction de classe** : Le modèle prédit la classe de l'animal dans l'image avec un score de confiance.
- **Top 3 prédictions** : Affichage des trois classes les plus probables avec leurs scores respectifs.
- **Interface intuitive** : Une interface simple et centrée pour une expérience utilisateur optimale.

## Architecture du Modèle
Le modèle utilisé est un modèle hybride d'apprentissage profond, entraîné sur le dataset [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10). Voici les étapes principales :
1. **Prétraitement des images** : Les images sont redimensionnées et normalisées pour correspondre aux dimensions d'entrée du modèle.
2. **Prédiction** : Le modèle génère un vecteur de probabilités pour chaque classe.
3. **Mapping des classes** : Les indices des classes sont traduits en labels compréhensibles grâce à un fichier de mapping.

### Schéma de l'Architecture
```plaintext
Image Input -> Prétraitement -> Modèle Hybride -> Vecteur de Probabilités -> Classe Prédite
```

## Utilisation
1. **Installation des dépendances** :
   Installez les dépendances nécessaires avec la commande suivante :
   ```bash
   pip install -r requirements.txt
   ```

2. **Lancer l'application** :
   Exécutez l'application Streamlit :
   ```bash
   streamlit run src/app.py
   ```

3. **Utilisation de l'interface** :
   - Téléversez une image d'animal.
   - Consultez la classe prédite et les scores de confiance.

## Structure du Projet
```
image_classifier/
│
├─ models/ 
├─ src/
│   ├─ __init__.py       
│   ├─ app.py               
│   ├─ config.py       
│   ├─ data_loader.py        
│   ├─ model_builder.py 
│   ├─ train.py          
│   └─ translate.py           
│
├─ .streamlit/
│   └─ config.toml          # Configuration du thème Streamlit
│
├─ requirements.txt         # Liste des dépendances
└─ README.md                # Documentation du projet
```

## Exemple d'Utilisation
1. Téléversez une image comme celle-ci :  
   ![Exemple d'image]()

2. Résultat attendu :  
   - Classe prédite : **Chat**  
   - Confiance : **95.3%**

## Remarques
- Le modèle est optimisé pour des images d'animaux uniquement.
- Les performances peuvent varier en fonction de la qualité de l'image téléversée.

## Auteurs
Ce projet a été développé pour simplifier la classification d'images d'animaux à l'aide de l'apprentissage profond.