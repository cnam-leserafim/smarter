# Project SMART - Smart Merchandise Automated Recognition Technology
Ce projet permet d'entraîner un modèle YOLO et d'exécuter des inférences en utilisant Picsellia pour la gestion des modèles et des expériences.

## 🎯 Objectif du projet

Le projet SMART (Smart Merchandise Automated Recognition Technology) a pour but de développer une solution en Python capable d'utiliser la vision par ordinateur pour reconnaître automatiquement un ensemble défini de produits.

## 👑 Membres de l'équipe

- Derya AY
- Juliette DEBRESSY
- Paul CHOPINET


## 🔧 Installation
- Installer **PyCharm**
- Installer **Python 3.11** via Windows Store
- Cloner le projet depuis GitHub
- Ouvrir le projet via PyCharm
- Configurer and activer l'environnement virtuel
- Installer les dépendances requises ```pip install -r requirements.txt```
- Exécutez le projet directement via PyCharm ou avec la commande : ```python main.py```

## 🚀 Prerequisites
- Télécharger PyCharm
- Installer Python 3.11
- Cuda
  
## Exécution du Pipeline d'Entraînement
Pour exécuter l'entraînement du modèle YOLO, utilisez la commande suivante :
```
python main.py
```
### Fonctionnalités du script d'entraînement (main.py)
- Importation des images et annotations depuis Picsellia
- Extraction des annotations compressées
- Répartition des données en ensembles d'entraînement, validation et test
- Génération du fichier de configuration config.yaml
- Entraînement du modèle YOLO
- Évaluation du modèle sur l'ensemble de validation
- Export du modèle entraîné vers Picsellia

## Exécution du Pipeline d'Inférence

Le script inference.py permet d'exécuter des inférences en utilisant le modèle YOLO entraîné. Il prend en charge trois modes :
- WEBCAM : Inférence en temps réel depuis une webcam
- IMAGE : Inférence sur une image locale
- VIDEO : Inférence sur une vidéo locale

 ### Commandes pour l'inférence
#### Mode Webcam
```
python inference.py --mode WEBCAM
```

#### Mode Image
```
python inference.py --mode IMAGE --path chemin/vers/image.jpg
```

#### Mode Vidéo
```
python inference.py --mode VIDEO --path chemin/vers/video.mp4
```

### Fonctionnalités du script d'inférence (inference.py)
- Récupération du dernier modèle YOLO enregistré dans Picsellia
- Chargement du modèle YOLO
- Prise en charge de l'inférence sur webcam, image et vidéo
- Affichage des résultats d'inférence

### Remarques 
- Les logs et les résultats sont accessibles via l'interface de Picsellia dans le Groupe-1
