# 🔍 Détecteur d'Images IA avec Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![GUI](https://img.shields.io/badge/GUI-Tkinter-yellow)

Une application complète de détection d'images générées par intelligence artificielle utilisant un modèle hybride combinant Deep Learning et analyse de caractéristiques traditionnelles.

## ✨ Fonctionnalités

### 🎯 Détection Avancée
- **Modèle CNN profond** avec régularisation L2 et Dropout
- **Analyse hybride** combinant deep learning + caractéristiques traditionnelles
- **Détection de texture** avec motifs binaires locaux (LBP)
- **Analyse fréquentielle** par transformée de Fourier (FFT)
- **Détection d'artefacts** de compression

### 🖥️ Interface Utilisateur
- **Interface graphique moderne** avec Tkinter
- **Visualisation interactive** des caractéristiques d'images
- **Onglets multiples** (Analyse, Visualisation, Paramètres)
- **Barres de progression** pour les opérations longues
- **Journalisation complète** des analyses

### 🔧 Outils Professionnels
- **Entraînement personnalisé** avec votre propre dataset
- **Analyse par lots** avec threading et progression
- **Validation croisée** optionnelle (5 folds)
- **Export des résultats** en CSV/Excel
- **Gestion de cache** pour accélérer les analyses

### 📊 Métriques et Visualisation
- **Matrices de confusion** détaillées
- **Graphiques comparatifs** des caractéristiques
- **Suivi de l'overfitting** en temps réel
- **Rapports de classification** complets
- **Visualisation radar** des caractéristiques avancées

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- 4GB RAM minimum (8GB recommandé)
- 2GB espace disque libre

### Installation Automatique
```bash
# Cloner le dépôt
git clone https://github.com/votre-username/ia-image-detector.git
cd ia-image-detector

# Installer les dépendances
pip install -r requirements.txt
```

### Installation Manuellement
```bash
pip install tensorflow pillow numpy opencv-python scikit-learn matplotlib seaborn pandas
```

## 📁 Structure du Projet

```
ia-image-detector/
├── fakeimg.py                  # Application principale
├── requirements.txt            # Dépendances
├── config.json                 # Configuration
├── ia_image_detector.h5        # Modèle pré-entraîné
├── best_model.h5              # Meilleur modèle sauvegardé
├── logs/                      # Journaux d'analyse
│   └── analysis_*.csv
├── dataset/                   # Structure recommandée
│   ├── train/
│   │   ├── real/
│   │   └── ai/
│   └── test/
│       ├── real/
│       └── ai/
└── README.md                  # Ce fichier
```

## 🎮 Utilisation

### Lancement de l'Application
```bash
python fakeimg.py
```

### Guide Rapide

1. **Analyse d'une image unique** :
   - Cliquez sur "📁 Sélectionner une image"
   - Cliquez sur "🔍 Analyser"
   - Consultez les résultats détaillés

2. **Entraînement du modèle** :
   - Cliquez sur "🎓 Entraîner le modèle"
   - Sélectionnez vos dossiers d'images réelles et IA
   - Configurez les paramètres d'entraînement
   - Lancez l'entraînement

3. **Analyse par lots** :
   - Cliquez sur "📂 Analyser un dossier"
   - Sélectionnez un dossier contenant des images
   - Suivez la progression en temps réel
   - Exportez les résultats

### Formats d'Image Supportés
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## 🧠 Architecture Technique

### Modèle Deep Learning
```python
Sequential([
    Augmentation Layer,
    Conv2D(32) + BatchNorm + Dropout(0.3),
    Conv2D(64) + BatchNorm + Dropout(0.3),
    Conv2D(128) + BatchNorm + Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(128) + Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### Caractéristiques Analysées
1. **Texture** : LBP, entropie, contraste
2. **Couleur** : Variance, cohérence LAB
3. **Fréquence** : Analyse FFT
4. **Bords** : Densité, qualité
5. **Artefacts** : Compression, bruit

## 📊 Performance

### Métriques Typiques
| Métrique | Valeur | Description |
|----------|--------|-------------|
| Précision Entraînement | 98-99% | Performance sur données connues |
| Précision Validation | 75-85% | Performance sur nouvelles données |
| Temps d'Analyse | 1-3s/image | Dépend du matériel |
| Taille Modèle | ~15MB | Fichier .h5 compressé |

### Amélioration de la Généralisation
- **Early Stopping** : Arrêt automatique pour éviter l'overfitting
- **Réduction LR** : Ajustement dynamique du learning rate
- **Validation Croisée** : 5 folds pour robustesse
- **Augmentation Données** : Transformations aléatoires

## 🔧 Configuration

### Fichier config.json
```json
{
    "img_size": [128, 128],
    "dropout_rate": 0.3,
    "l2_reg": 0.001,
    "batch_size": 32,
    "epochs": 30,
    "use_early_stopping": true,
    "early_stopping_patience": 10
}
```

### Paramètres Modifiables
- **Taille d'image** : 128x128 par défaut
- **Taux de Dropout** : Contrôle l'overfitting
- **Régularisation L2** : Pénalise les poids importants
- **Batch Size** : Nombre d'images par lot
- **Early Stopping** : Patience avant arrêt

## 📈 Résultats et Visualisation

L'application génère plusieurs types de visualisations :

1. **Graphiques à barres** : Caractéristiques principales
2. **Radar plot** : Caractéristiques avancées
3. **Matrices de confusion** : Performance du modèle
4. **Courbes d'apprentissage** : Suivi de l'overfitting

## 🐛 Dépannage

### Problèmes Courants

1. **Erreur de mémoire** :
```bash
# Réduire la taille du batch
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

2. **Importations manquantes** :
```bash
pip install --upgrade -r requirements.txt
```

3. **Modèle non chargé** :
```bash
# Supprimer et recréer le modèle
rm ia_image_detector.h5
python fakeimg.py
```

### Journaux et Debug
- Les logs sont sauvegardés dans `logs/`
- Chaque analyse génère un fichier CSV horodaté
- Les erreurs sont capturées et affichées dans l'interface

## 📝 Exemples d'Utilisation

### Pour les Développeurs
```python
# Utilisation programmatique
detector = HybridDetector(config_manager)
result = detector.predict_image("image.jpg")
print(f"Résultat: {result['message']}")
```

### Pour la Recherche
- Modifiez `create_improved_model()` pour expérimenter
- Utilisez la validation croisée pour des résultats robustes
- Exportez les données pour analyse externe

### Pour la Production
- Augmentez `img_size` pour plus de précision
- Ajoutez plus de données d'entraînement
- Utilisez `analyze_batch()` pour le traitement en masse

## 🤝 Contribution

### Rapport de Bugs
1. Vérifiez les issues existantes
2. Décrivez le bug avec précision
3. Incluez les messages d'erreur
4. Fournissez des images de test si possible

### Suggestions d'Amélioration
1. Décrivez la fonctionnalité
2. Expliquez son utilité
3. Proposez une implémentation si possible

### Développement
```bash
# Fork le projet
git clone votre-fork
cd ia-image-detector

# Créer une branche
git checkout -b feature/ma-fonctionnalité

# Commiter les changements
git commit -m "Ajout de ma fonctionnalité"

# Pusher
git push origin feature/ma-fonctionnalité

# Créer une Pull Request
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **TensorFlow/Keras** : Framework de deep learning
- **PIL/Pillow** : Traitement d'images
- **OpenCV** : Vision par ordinateur
- **Scikit-learn** : Métriques et validation
- **Tkinter** : Interface graphique



---

**⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile sur GitHub !**

## 🚀 Roadmap

### À Venir
- [ ] Support des modèles pré-entraînés (EfficientNet, ResNet)
- [ ] API REST pour intégration web
- [ ] Dockerisation
- [ ] Interface web avec Streamlit
- [ ] Support GPU avancé
- [ ] Benchmark avec d'autres méthodes

### En Développement
- ✅ Interface graphique complète
- ✅ Analyse hybride deep learning + caractéristiques
- ✅ Entraînement personnalisé
- ✅ Export des résultats
- ✅ Visualisation des données

### Réalisé
- ✅ Modèle CNN de base
- ✅ Détection de caractéristiques traditionnelles
- ✅ Interface utilisateur simple
- ✅ Sauvegarde/chargement des modèles

---

