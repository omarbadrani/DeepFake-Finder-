import os
import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
from datetime import datetime
import cv2
from sklearn.model_selection import train_test_split, KFold
import threading
import time
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Désactiver les warnings inutiles
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.run_functions_eagerly(True)

# Optimiser TensorFlow
tf.config.set_visible_devices([], 'GPU')  # Désactiver GPU si non disponible
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


class ConfigManager:
    """Gestionnaire de configuration"""

    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.default_config = {
            "model_path": "ia_image_detector.h5",
            "img_size": [128, 128],
            "threshold": 0.5,
            "max_cache_size": 50,
            "batch_size": 32,
            "epochs": 30,
            "theme": "dark",
            "language": "fr",
            "auto_save": True,
            "export_format": "csv",
            "dropout_rate": 0.3,
            "l2_reg": 0.001,
            "use_early_stopping": True,
            "early_stopping_patience": 10,
            "use_reduce_lr": True,
            "reduce_lr_patience": 5
        }
        self.config = self.load_config()

    def load_config(self):
        """Charge la configuration depuis un fichier"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return {**self.default_config, **json.load(f)}
            except:
                return self.default_config
        return self.default_config

    def save_config(self):
        """Sauvegarde la configuration"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        if self.config.get('auto_save', True):
            self.save_config()


class ImageCache:
    """Cache pour les images analysées"""

    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, image_path):
        """Récupère une image du cache"""
        if image_path in self.cache:
            # Mettre à jour l'ordre d'accès
            self.access_order.remove(image_path)
            self.access_order.append(image_path)
            return self.cache[image_path]
        return None

    def put(self, image_path, result):
        """Ajoute un résultat au cache"""
        if len(self.cache) >= self.max_size:
            # Supprimer le moins récemment utilisé
            lru = self.access_order.pop(0)
            del self.cache[lru]

        self.cache[image_path] = result
        self.access_order.append(image_path)


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_callback, start_progress, progress_range):
        super().__init__()
        self.progress_callback = progress_callback
        self.start_progress = start_progress
        self.progress_range = progress_range

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback:
            progress = self.start_progress + int(self.progress_range * (epoch + 1) / self.params['epochs'])
            # Conversion explicite en float
            accuracy = float(logs['accuracy'])
            self.progress_callback(f"Époque {epoch + 1}/{self.params['epochs']} - Précision: {accuracy:.2f}", progress)


class ModelDiagnostic:
    """Diagnostic du modèle pour détecter l'overfitting"""

    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def run_diagnostics(self):
        """Exécute des diagnostics complets"""
        print("🔍 DIAGNOSTIC DU MODÈLE")
        print("=" * 50)

        # 1. Évaluer le modèle
        loss, acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        print(f"Précision validation: {acc:.2%}")

        # 2. Vérifier l'overfitting
        y_pred = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)

        # 3. Calculer les métriques
        print("\n📊 RAPPORT DE CLASSIFICATION:")
        print(classification_report(self.y_val, y_pred,
                                    target_names=['Réel', 'IA']))

        print("\n🎯 MATRICE DE CONFUSION:")
        cm = confusion_matrix(self.y_val, y_pred)
        print(cm)

        # 4. Suggestions
        print("\n💡 RECOMMANDATIONS:")
        if acc < 0.7:
            print("- Le modèle a une faible précision")
            print("- Vérifiez la qualité des données")
            print("- Essayez un modèle pré-entraîné")

        if cm[0, 1] > cm[0, 0] or cm[1, 0] > cm[1, 1]:
            print("- Le modèle a des biais de classification")
            print("- Équilibrer les classes d'entraînement")

        return acc, cm


class DeepLearningDetector:
    def __init__(self, config_manager):
        self.config = config_manager
        self.model = None
        self.img_size = tuple(config_manager.get("img_size", [128, 128]))
        self.cache = ImageCache(max_size=config_manager.get("max_cache_size", 50))
        self.load_or_create_model()

    def create_strong_augmentation_layer(self):
        """Augmentation de données plus agressive pour réduire l'overfitting"""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(0.1, 0.1),
        ])

    def create_improved_model(self):
        """Crée un modèle CNN amélioré avec régularisation pour éviter l'overfitting"""
        dropout_rate = self.config.get("dropout_rate", 0.3)
        l2_reg = self.config.get("l2_reg", 0.001)

        model = models.Sequential([
            # Couche d'augmentation de données
            self.create_strong_augmentation_layer(),

            # Préprocessing
            layers.Rescaling(1. / 255, input_shape=(*self.img_size, 3)),

            # Block 1 avec régularisation
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Block 3 simplifié
            layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Classification avec moins de neurones
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        # Optimiseur avec scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    def get_training_callbacks(self):
        """Retourne les callbacks pour éviter l'overfitting"""
        callbacks = []

        if self.config.get("use_early_stopping", True):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get("early_stopping_patience", 10),
                    restore_best_weights=True,
                    verbose=1
                )
            )

        if self.config.get("use_reduce_lr", True):
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.get("reduce_lr_patience", 5),
                    min_lr=1e-6,
                    verbose=1
                )
            )

        # Checkpoint du meilleur modèle
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )

        return callbacks

    def load_or_create_model(self):
        """Charge un modèle existant ou en crée un nouveau"""
        model_path = self.config.get("model_path", "ia_image_detector.h5")

        if os.path.exists(model_path):
            try:
                # Charger le modèle sans l'optimiseur pour éviter les problèmes de compatibilité
                self.model = tf.keras.models.load_model(model_path, compile=False)
                # Recompiler le modèle avec un nouvel optimiseur
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("✅ Modèle de détection IA chargé et recompilé")
            except Exception as e:
                print(f"❌ Erreur lors du chargement: {e}, création d'un nouveau modèle amélioré")
                self.model = self.create_improved_model()
        else:
            print("🔄 Création d'un nouveau modèle amélioré")
            self.model = self.create_improved_model()

    def optimized_preprocess_image(self, image_path):
        """Prétraitement optimisé avec cache"""
        # Vérifier le cache d'abord
        cached = self.cache.get(image_path)
        if cached:
            return cached

        try:
            # Charger avec PIL en mode paresseux
            img = Image.open(image_path)

            # Vérifier les dimensions
            if img.width > 2048 or img.height > 2048:
                # Réduire les images trop grandes
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            # Convertir en RGB si nécessaire
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize avec interpolation optimale
            img = img.resize(self.img_size, Image.Resampling.BILINEAR)

            # Conversion numpy optimisée
            img_array = np.array(img, dtype=np.float32) / 255.0

            result = np.expand_dims(img_array, axis=0)

            # Mettre en cache
            self.cache.put(image_path, result)

            return result
        except Exception as e:
            print(f"Erreur de prétraitement: {e}")
            return None

    def predict(self, image_path):
        """Prédit si une image est générée par IA avec le modèle deep learning"""
        processed_img = self.optimized_preprocess_image(image_path)

        if processed_img is not None and self.model is not None:
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            # Conversion explicite en float pour éviter les problèmes de tenseur
            confidence = float(prediction)

            if confidence > 0.5:
                return {
                    'is_ai_generated': True,
                    'confidence': confidence,
                    'message': f"🤖 DEEP LEARNING: Généré par IA ({confidence * 100:.2f}% de confiance)",
                    'method': 'deep_learning'
                }
            else:
                return {
                    'is_ai_generated': False,
                    'confidence': 1 - confidence,
                    'message': f"📷 DEEP LEARNING: Image réelle ({(1 - confidence) * 100:.2f}% de confiance)",
                    'method': 'deep_learning'
                }

        return None

    def load_all_images(self, real_images_dir, ai_images_dir):
        """Charge toutes les images des dossiers"""
        X, y = [], []

        # Charger les images réelles
        real_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            real_images.extend([f for f in os.listdir(real_images_dir) if f.lower().endswith(ext)])

        # Charger les images IA
        ai_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            ai_images.extend([f for f in os.listdir(ai_images_dir) if f.lower().endswith(ext)])

        print(f"📊 Images réelles trouvées: {len(real_images)}")
        print(f"📊 Images IA trouvées: {len(ai_images)}")

        if len(real_images) == 0 or len(ai_images) == 0:
            raise ValueError(f"Aucune image trouvée. Réelles: {len(real_images)}, IA: {len(ai_images)}")

        # Vérifier l'équilibre des données
        self.check_data_balance(real_images, ai_images)

        # Charger les images réelles
        for filename in real_images:
            img_path = os.path.join(real_images_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(0)  # 0 pour images réelles
            except Exception as e:
                print(f"⚠️ Erreur avec {filename}: {e}")

        # Charger les images IA
        for filename in ai_images:
            img_path = os.path.join(ai_images_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(1)  # 1 pour images IA
            except Exception as e:
                print(f"⚠️ Erreur avec {filename}: {e}")

        # Convertir en arrays numpy
        X = np.array(X)
        y = np.array(y)

        # Mélanger les données
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        return X, y

    def check_data_balance(self, real_images, ai_images):
        """Vérifie l'équilibre des données"""
        real_count = len(real_images)
        ai_count = len(ai_images)

        print(f"📈 Images réelles: {real_count}")
        print(f"📈 Images IA: {ai_count}")
        print(f"📈 Ratio IA/Réelles: {ai_count / real_count:.2f}")

        if abs(real_count - ai_count) > 0.3 * min(real_count, ai_count):
            print("⚠️ Déséquilibre des données détecté!")
            print("Cela peut causer des biais dans le modèle.")

            # Correction par augmentation ciblée
            if real_count < ai_count:
                print(f"💡 Conseil: Augmenter les images réelles de {ai_count - real_count}")
            else:
                print(f"💡 Conseil: Augmenter les images IA de {real_count - ai_count}")

    def train_model(self, real_images_dir, ai_images_dir, epochs=None, progress_callback=None):
        """Entraîne le modèle avec des images réelles et IA"""
        if epochs is None:
            epochs = self.config.get("epochs", 30)

        batch_size = self.config.get("batch_size", 32)

        # Charger les images
        X, y = self.load_all_images(real_images_dir, ai_images_dir)

        if len(X) == 0:
            raise ValueError("Aucune image valide trouvée dans les dossiers spécifiés")

        # Diviser en ensembles d'entraînement et de test
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"📊 Données d'entraînement: {len(X_train)}")
        print(f"📊 Données de validation: {len(X_val)}")

        if progress_callback:
            progress_callback("Début de l'entraînement...", 70)

        # Obtenir les callbacks
        callbacks = self.get_training_callbacks()
        if progress_callback:
            callbacks.append(TrainingCallback(progress_callback, 70, 25))

        # Entraîner le modèle
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

        # Sauvegarder le modèle (sans l'optimiseur pour éviter les problèmes)
        self.model.save("ia_image_detector.h5", include_optimizer=False)
        print("💾 Modèle sauvegardé: ia_image_detector.h5")

        if progress_callback:
            progress_callback("Entraînement terminé!", 95)
            time.sleep(1)
            progress_callback("Sauvegarde du modèle...", 100)

        # Afficher les résultats
        self.analyze_training_results(history, X_val, y_val)

        return history, X_val, y_val

    def analyze_training_results(self, history, X_val, y_val):
        """Analyse les résultats de l'entraînement"""
        train_accuracy = float(history.history['accuracy'][-1])
        val_accuracy = float(history.history['val_accuracy'][-1])

        print(f"✅ Précision sur l'ensemble d'entraînement: {train_accuracy * 100:.2f}%")
        print(f"✅ Précision sur l'ensemble de validation: {val_accuracy * 100:.2f}%")

        # Calculer le gap d'overfitting
        overfit_gap = train_accuracy - val_accuracy
        print(f"📊 Gap d'overfitting: {overfit_gap:.2%}")

        if overfit_gap > 0.2:  # 20% de gap = overfitting sévère
            print("⚠️ OVERFITTING SÉVÈRE DÉTECTÉ!")
            print("💡 Recommandations:")
            print("1. Augmenter le Dropout (0.3-0.5)")
            print("2. Ajouter plus d'augmentation de données")
            print("3. Réduire la complexité du modèle")
            print("4. Collecter plus de données")

        # Diagnostiquer le modèle
        diagnostic = ModelDiagnostic(self.model, X_val, y_val)
        diagnostic.run_diagnostics()

    def train_with_cross_validation(self, real_images_dir, ai_images_dir, k_folds=5):
        """Entraînement avec validation croisée"""
        # Charger toutes les données
        X, y = self.load_all_images(real_images_dir, ai_images_dir)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_no = 1
        accuracies = []

        for train_idx, val_idx in kfold.split(X, y):
            print(f'\n🎯 Entraînement sur fold {fold_no}...')

            # Créer un nouveau modèle pour chaque fold
            model = self.create_improved_model()

            # Données d'entraînement et validation
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Entraîner avec early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[early_stopping]
            )

            # Évaluer
            scores = model.evaluate(X_val, y_val, verbose=0)
            accuracies.append(scores[1] * 100)

            print(f'✅ Fold {fold_no}: {scores[1] * 100:.2f}% de précision')
            fold_no += 1

        print(f'\n📊 Précision moyenne: {np.mean(accuracies):.2f}% (+/- {np.std(accuracies):.2f}%)')

        # Entraîner le modèle final sur toutes les données
        print('\n🎯 Entraînement du modèle final...')
        self.model = self.create_improved_model()

        # Utiliser early stopping pour l'entraînement final
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        return np.mean(accuracies)


class HybridDetector:
    def __init__(self, config_manager):
        self.config = config_manager
        self.dl_detector = DeepLearningDetector(config_manager)
        self.img_size = (256, 256)
        self.cache = ImageCache(max_size=config_manager.get("max_cache_size", 50))

    def calculate_lbp(self, image):
        """Calcule les motifs binaires locaux (LBP) pour l'analyse de texture"""
        try:
            # Implémentation simplifiée de LBP
            radius = 1
            n_points = 8 * radius
            lbp = np.zeros_like(image, dtype=np.uint8)

            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    binary_code = 0
                    for k, (di, dj) in enumerate(
                            [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                        if image[i + di, j + dj] >= center:
                            binary_code |= (1 << k)
                    lbp[i, j] = binary_code

            return lbp
        except:
            return np.zeros_like(image, dtype=np.uint8)

    def calculate_entropy(self, image):
        """Calcule l'entropie d'une image"""
        hist = np.histogram(image, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return entropy

    def calculate_fft_features(self, image):
        """Calcule les caractéristiques FFT d'une image"""
        try:
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            return np.mean(magnitude_spectrum), np.std(magnitude_spectrum)
        except:
            return 0, 0

    def detect_compression_artifacts(self, image_array):
        """Détecte les artefacts de compression"""
        try:
            # Conversion en niveaux de gris
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            # Calcul de la variance de Laplace pour détecter le flou
            laplace_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Les images fortement compressées ont souvent une variance de Laplace faible
            return laplace_var < 100
        except:
            return False

    def calculate_color_consistency(self, a_channel, b_channel):
        """Calcule la cohérence des couleurs dans l'espace LAB"""
        try:
            a_std, b_std = np.std(a_channel), np.std(b_channel)
            return a_std + b_std
        except:
            return 0

    def extract_advanced_features(self, image_path):
        """Extrait des caractéristiques avancées pour l'analyse"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            # Conversion en espace colorimétrique LAB pour une meilleure analyse
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Analyse de texture avec LBP (Local Binary Patterns)
            lbp = self.calculate_lbp(l)

            # Analyse de fréquence avec FFT
            fft_mean, fft_std = self.calculate_fft_features(l)

            # Détection d'artefacts de compression
            compression_artifacts = self.detect_compression_artifacts(img_array)

            features = {
                'contrast': np.std(l),
                'color_variance': [np.std(img_array[:, :, i]) for i in range(3)],
                'edge_density': self.calculate_edge_density(Image.fromarray(img_array)),
                'noise_level': self.calculate_noise_level(Image.fromarray(img_array)),
                'perceptual_quality': self.calculate_perceptual_quality(Image.fromarray(img_array)),
                'lbp_entropy': self.calculate_entropy(lbp),
                'fft_mean': fft_mean,
                'fft_std': fft_std,
                'compression_artifacts': compression_artifacts,
                'color_consistency': self.calculate_color_consistency(a, b)
            }

            return features
        except Exception as e:
            print(f"Erreur lors de l'extraction avancée: {e}")
            return None

    def extract_features(self, image_path):
        """Extrait des caractéristiques pour l'analyse hybride"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize(self.img_size)
            img_array = np.array(img_resized)

            gray = img_resized.convert('L')
            gray_array = np.array(gray)

            features = {
                'contrast': np.std(gray_array),
                'color_variance': [np.std(img_array[:, :, i]) for i in range(3)],
                'edge_density': self.calculate_edge_density(gray),
                'noise_level': self.calculate_noise_level(gray),
                'perceptual_quality': self.calculate_perceptual_quality(img_resized),
            }

            return features
        except Exception as e:
            print(f"Erreur lors de l'extraction: {e}")
            return None

    def calculate_edge_density(self, gray_image):
        """Calcule la densité des bords"""
        try:
            edges = gray_image.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            return np.mean(edges_array > 30)
        except:
            return 0

    def calculate_noise_level(self, gray_image):
        """Calcule le niveau de bruit"""
        try:
            array = np.array(gray_image)
            blurred = gray_image.filter(ImageFilter.GaussianBlur(2))
            blurred_array = np.array(blurred)
            noise = np.std(array - blurred_array)
            return noise
        except:
            return 0

    def calculate_perceptual_quality(self, image):
        """Calcule une mesure de qualité perceptuelle"""
        try:
            gray = image.convert('L')
            array = np.array(gray)
            contrast = np.std(array)
            brightness = np.mean(array)
            return contrast * brightness / 100
        except:
            return 0

    def validate_image_file(self, file_path):
        """Valide qu'un fichier est une image sécuritaire"""
        try:
            # Vérifier l'extension
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in valid_extensions:
                return False, "Extension non supportée"

            # Vérifier la taille du fichier (max 50MB)
            if os.path.getsize(file_path) > 50 * 1024 * 1024:
                return False, "Fichier trop volumineux (>50MB)"

            # Ouvrir l'image avec PIL pour validation
            with Image.open(file_path) as img:
                img.verify()  # Vérifie l'intégrité du fichier
                # Vérifier les dimensions raisonnables
                if img.width * img.height > 10000 * 10000:
                    return False, "Dimensions trop grandes"

            return True, "Fichier valide"
        except Exception as e:
            return False, f"Erreur de validation: {str(e)}"

    def safe_predict(self, image_path):
        """Version robuste de la prédiction"""
        try:
            # Vérifier le cache d'abord
            cached = self.cache.get(image_path)
            if cached:
                return cached

            if not os.path.exists(image_path):
                result = {"error": "Fichier non trouvé", "confidence": 0.0}
                self.cache.put(image_path, result)
                return result

            # Vérifier la taille du fichier
            if os.path.getsize(image_path) == 0:
                result = {"error": "Fichier vide", "confidence": 0.0}
                self.cache.put(image_path, result)
                return result

            # Valider le fichier
            is_valid, message = self.validate_image_file(image_path)
            if not is_valid:
                result = {"error": message, "confidence": 0.0}
                self.cache.put(image_path, result)
                return result

            result = self.predict_image(image_path)
            if result:
                self.cache.put(image_path, result)
            return result
        except Exception as e:
            result = {"error": str(e), "confidence": 0.0}
            self.cache.put(image_path, result)
            return result

    def predict_image(self, image_path):
        """Prédiction hybride combinant deep learning et analyse de caractéristiques"""
        # Prédiction par deep learning
        dl_result = self.dl_detector.predict(image_path)

        # Analyse par caractéristiques avancées
        features = self.extract_advanced_features(image_path)

        if dl_result is None or features is None:
            return None

        # Combinaison des résultats (pondération)
        dl_confidence = dl_result['confidence'] if dl_result['is_ai_generated'] else 1 - dl_result['confidence']

        # Règles basées sur les caractéristiques pour ajuster la confiance
        feature_score = 0
        feature_details = []

        # Images IA ont souvent moins de contraste
        if features['contrast'] < 25:
            feature_score += 0.1
            feature_details.append("Faible contraste (typique des images IA)")
        else:
            feature_score -= 0.1
            feature_details.append("Bon contraste (typique des images réelles)")

        # Images IA ont souvent une variance de couleur anormale
        color_var_avg = np.mean(features['color_variance'])
        if color_var_avg < 15 or color_var_avg > 70:
            feature_score += 0.1
            feature_details.append("Variance de couleur anormale (typique des images IA)")
        else:
            feature_score -= 0.1
            feature_details.append("Variance de couleur normale (typique des images réelles)")

        # Images IA ont souvent une densité de bords particulière
        if features['edge_density'] < 0.01 or features['edge_density'] > 0.1:
            feature_score += 0.1
            feature_details.append("Densité de bords anormale (typique des images IA)")
        else:
            feature_score -= 0.1
            feature_details.append("Densité de bords normale (typique des images réelles)")

        # Utilisation des caractéristiques avancées
        if features.get('compression_artifacts', False):
            feature_score += 0.15
            feature_details.append("Artefacts de compression détectés (typique des images IA)")
        else:
            feature_score -= 0.05
            feature_details.append("Pas d'artefacts de compression significatifs")

        if features.get('lbp_entropy', 0) < 5:
            feature_score += 0.1
            feature_details.append("Faible entropie de texture (typique des images IA)")
        else:
            feature_score -= 0.05
            feature_details.append("Entropie de texture normale (typique des images réelles)")

        # Ajustement de la confiance basé sur les caractéristiques
        adjusted_confidence = min(max(dl_confidence + feature_score * 0.3, 0.05), 0.95)

        # Détermination finale
        if adjusted_confidence > 0.5:
            return {
                'is_ai_generated': True,
                'confidence': adjusted_confidence,
                'message': f"🤖 HYBRIDE: Généré par IA ({adjusted_confidence * 100:.2f}% de confiance)",
                'method': 'hybrid',
                'dl_confidence': dl_confidence,
                'feature_score': feature_score,
                'feature_details': feature_details,
                'advanced_features': features
            }
        else:
            return {
                'is_ai_generated': False,
                'confidence': 1 - adjusted_confidence,
                'message': f"📷 HYBRIDE: Image réelle ({(1 - adjusted_confidence) * 100:.2f}% de confiance)",
                'method': 'hybrid',
                'dl_confidence': dl_confidence,
                'feature_score': feature_score,
                'feature_details': feature_details,
                'advanced_features': features
            }


class BatchAnalyzer:
    """Analyse par lots avec threading"""

    def __init__(self, detector, max_workers=4):
        self.detector = detector
        self.max_workers = max_workers
        self.results = []
        self.progress_callback = None

    def analyze_folder(self, folder_path, progress_callback=None):
        """Analyse un dossier d'images avec threading"""
        self.progress_callback = progress_callback

        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            image_files.extend([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(ext)
            ])

        self.results = []
        total = len(image_files)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.detector.safe_predict, f): f
                for f in image_files
            }

            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=30)
                    if result and 'error' not in result:
                        result['file'] = file_path
                        self.results.append(result)
                    else:
                        self.results.append({
                            'file': file_path,
                            'error': result.get('error', 'Erreur inconnue') if result else 'Erreur inconnue'
                        })
                except Exception as e:
                    self.results.append({
                        'file': file_path,
                        'error': str(e)
                    })

                if progress_callback:
                    progress_callback(i + 1, total)

        return self.results

    def export_results(self, output_path, format='csv'):
        """Exporte les résultats"""
        if format == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Fichier', 'Résultat', 'Confiance', 'Méthode', 'Erreur'])

                for result in self.results:
                    if 'error' in result:
                        writer.writerow([
                            os.path.basename(result['file']),
                            'ERREUR',
                            '0%',
                            'N/A',
                            result['error']
                        ])
                    else:
                        writer.writerow([
                            os.path.basename(result['file']),
                            'IA' if result['is_ai_generated'] else 'Réelle',
                            f"{result['confidence'] * 100:.2f}%",
                            result.get('method', 'N/A'),
                            ''
                        ])


class TrainingInterface:
    def __init__(self, detector):
        self.detector = detector
        self.config = detector.config
        self.setup_training_ui()

    def setup_training_ui(self):
        """Interface pour l'entraînement du modèle"""
        self.train_window = tk.Toplevel()
        self.train_window.title("🎓 Entraînement du Modèle IA")
        self.train_window.geometry("800x700")
        self.train_window.resizable(False, False)

        # Centrer la fenêtre
        self.train_window.transient(self.train_window.master)
        self.train_window.grab_set()

        main_frame = tk.Frame(self.train_window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="🎓 ENTRAÎNEMENT DU MODÈLE DE DÉTECTION IA",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Frame pour les chemins
        paths_frame = tk.LabelFrame(main_frame, text="📁 DOSSIERS D'IMAGES", font=("Arial", 10, "bold"))
        paths_frame.pack(fill=tk.X, pady=10, padx=5)

        tk.Label(paths_frame, text="Dossier d'images réelles:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
        self.real_path_entry = tk.Entry(paths_frame, width=40)
        self.real_path_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(paths_frame, text="Parcourir",
                  command=lambda: self.browse_folder(self.real_path_entry)).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(paths_frame, text="Dossier d'images IA:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        self.ai_path_entry = tk.Entry(paths_frame, width=40)
        self.ai_path_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(paths_frame, text="Parcourir",
                  command=lambda: self.browse_folder(self.ai_path_entry)).grid(row=1, column=2, padx=5, pady=5)

        # Info sur les images trouvées
        self.info_label = tk.Label(paths_frame, text="Sélectionnez les dossiers pour voir le nombre d'images",
                                   fg="gray")
        self.info_label.grid(row=2, column=0, columnspan=3, pady=10)

        # Frame pour les paramètres
        params_frame = tk.LabelFrame(main_frame, text="⚙️ PARAMÈTRES D'ENTRAÎNEMENT", font=("Arial", 10, "bold"))
        params_frame.pack(fill=tk.X, pady=10, padx=5)

        tk.Label(params_frame, text="Nombre d'époques:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
        self.epochs_entry = tk.Entry(params_frame, width=10)
        self.epochs_entry.insert(0, str(self.config.get("epochs", 30)))
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        tk.Label(params_frame, text="Taille du batch:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.batch_entry = tk.Entry(params_frame, width=10)
        self.batch_entry.insert(0, str(self.config.get("batch_size", 32)))
        self.batch_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        tk.Label(params_frame, text="Taux de Dropout:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        self.dropout_entry = tk.Entry(params_frame, width=10)
        self.dropout_entry.insert(0, str(self.config.get("dropout_rate", 0.3)))
        self.dropout_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Options avancées
        self.use_augmentation = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="Utiliser l'augmentation de données",
                       variable=self.use_augmentation).grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=5, padx=10)

        self.use_cross_val = tk.BooleanVar(value=False)
        tk.Checkbutton(params_frame, text="Utiliser la validation croisée (5 folds)",
                       variable=self.use_cross_val).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=5, padx=10)

        # Barre de progression
        progress_frame = tk.LabelFrame(main_frame, text="📊 PROGRESSION", font=("Arial", 10, "bold"))
        progress_frame.pack(fill=tk.X, pady=10, padx=5)

        self.progress_label = tk.Label(progress_frame, text="Prêt à entraîner", fg="blue")
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)

        # Zone de logs
        log_frame = tk.LabelFrame(main_frame, text="📝 LOGS", font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Boutons
        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=20)

        self.train_button = tk.Button(buttons_frame, text="🎯 Commencer l'Entraînement",
                                      command=self.start_training, bg="#4CAF50", fg="white",
                                      font=("Arial", 10, "bold"), padx=20, pady=10)
        self.train_button.pack(side=tk.LEFT, padx=10)

        self.cancel_button = tk.Button(buttons_frame, text="❌ Annuler",
                                       command=self.train_window.destroy, bg="#f44336", fg="white",
                                       padx=20, pady=10)
        self.cancel_button.pack(side=tk.LEFT, padx=10)

        # Lier les événements de modification des entrées
        self.real_path_entry.bind('<KeyRelease>', self.update_image_count)
        self.ai_path_entry.bind('<KeyRelease>', self.update_image_count)

    def browse_folder(self, entry):
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if folder_path:
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)
            self.update_image_count()

    def update_image_count(self, event=None):
        """Met à jour le compte des images dans les dossiers"""
        real_path = self.real_path_entry.get()
        ai_path = self.ai_path_entry.get()

        real_count = 0
        ai_count = 0

        if os.path.exists(real_path):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                real_count += len([f for f in os.listdir(real_path) if f.lower().endswith(ext)])

        if os.path.exists(ai_path):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                ai_count += len([f for f in os.listdir(ai_path) if f.lower().endswith(ext)])

        if real_count > 0 or ai_count > 0:
            self.info_label.config(text=f"✅ Images trouvées: {real_count} réelles, {ai_count} IA", fg="green")
        else:
            self.info_label.config(text="⚠️ Aucune image trouvée (formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp)",
                                   fg="red")

    def update_progress(self, message, value):
        """Met à jour la barre de progression et le message"""
        self.progress_label.config(text=message)
        self.progress_bar['value'] = value
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.train_window.update_idletasks()

    def start_training(self):
        real_path = self.real_path_entry.get()
        ai_path = self.ai_path_entry.get()

        # Vérifier les chemins
        if not real_path or not ai_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner les deux dossiers")
            return

        if not os.path.exists(real_path) or not os.path.exists(ai_path):
            messagebox.showerror("Erreur", "Un ou plusieurs dossiers n'existent pas")
            return

        # Vérifier qu'il y a des images
        real_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            real_images.extend([f for f in os.listdir(real_path) if f.lower().endswith(ext)])

        ai_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            ai_images.extend([f for f in os.listdir(ai_path) if f.lower().endswith(ext)])

        if len(real_images) == 0 or len(ai_images) == 0:
            messagebox.showerror("Erreur", f"Aucune image trouvée. Réelles: {len(real_images)}, IA: {len(ai_images)}")
            return

        try:
            epochs = int(self.epochs_entry.get())
            if epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erreur", "Le nombre d'époques doit être un entier positif")
            return

        try:
            batch_size = int(self.batch_entry.get())
            if batch_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erreur", "La taille du batch doit être un entier positif")
            return

        try:
            dropout_rate = float(self.dropout_entry.get())
            if not 0 <= dropout_rate <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erreur", "Le taux de dropout doit être entre 0 et 1")
            return

        # Mettre à jour la configuration
        self.config.set("epochs", epochs)
        self.config.set("batch_size", batch_size)
        self.config.set("dropout_rate", dropout_rate)

        # Désactiver le bouton pendant l'entraînement
        self.train_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)

        # Lancer l'entraînement dans un thread séparé
        training_thread = threading.Thread(
            target=self.run_training,
            args=(real_path, ai_path, epochs, batch_size)
        )
        training_thread.daemon = True
        training_thread.start()

    def run_training(self, real_path, ai_path, epochs, batch_size):
        """Exécute l'entraînement dans un thread séparé"""
        try:
            self.update_progress("📊 Vérification de l'équilibre des données...", 10)

            if self.use_cross_val.get():
                self.update_progress("🎯 Démarrage de la validation croisée (5 folds)...", 20)
                accuracy = self.detector.dl_detector.train_with_cross_validation(real_path, ai_path)
                self.update_progress(f"✅ Validation croisée terminée - Précision moyenne: {accuracy:.2f}%", 90)
            else:
                self.update_progress("🎯 Démarrage de l'entraînement avec early stopping...", 20)
                history, X_test, y_test = self.detector.dl_detector.train_model(
                    real_path, ai_path, epochs=epochs,
                    progress_callback=self.update_progress
                )

                # Évaluer le modèle
                y_pred = (self.detector.dl_detector.model.predict(X_test, verbose=0) > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Afficher les résultats
                train_acc = float(history.history['accuracy'][-1]) * 100
                val_acc = float(history.history['val_accuracy'][-1]) * 100

                self.update_progress("✅ Entraînement terminé!", 95)

                result_text = f"""
                📊 RÉSULTATS FINAUX:

                Précision entraînement: {train_acc:.2f}%
                Précision validation: {val_acc:.2f}%

                📈 MÉTRIQUES DÉTAILLÉES:
                Exactitude: {accuracy:.2%}
                Précision: {precision:.2%}
                Rappel: {recall:.2%}
                Score F1: {f1:.2%}
                """

                self.log_text.insert(tk.END, result_text)
                self.log_text.see(tk.END)

            # Réactiver le bouton
            self.train_button.config(state=tk.NORMAL)

            # Afficher un message de succès
            messagebox.showinfo("🎉 Entraînement terminé",
                                "Modèle entraîné avec succès!\n\n"
                                "Le modèle a été sauvegardé dans 'ia_image_detector.h5'")

        except Exception as e:
            # Réactiver le bouton en cas d'erreur
            self.train_button.config(state=tk.NORMAL)
            self.update_progress(f"❌ Erreur: {str(e)}", 100)
            messagebox.showerror("Erreur", f"Erreur lors de l'entraînement: {str(e)}")


class IADetectorGUI:
    def __init__(self, root):
        self.root = root
        self.config = ConfigManager()
        self.detector = HybridDetector(self.config)
        self.batch_analyzer = BatchAnalyzer(self.detector)
        self.setup_ui()
        self.setup_logging()

    def setup_logging(self):
        """Configure la journalisation des analyses"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Fichier de log avec horodatage
        self.log_file = os.path.join(log_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        # En-tête CSV
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Filename', 'Prediction', 'Confidence',
                'Contrast', 'ColorVariance', 'EdgeDensity', 'NoiseLevel'
            ])

    def log_analysis(self, image_path, result, features):
        """Journalise les résultats de l'analyse"""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                os.path.basename(image_path),
                'AI' if result['is_ai_generated'] else 'Real',
                result['confidence'],
                features.get('contrast', 0),
                np.mean(features.get('color_variance', [0, 0, 0])),
                features.get('edge_density', 0),
                features.get('noise_level', 0)
            ])

    def setup_ui(self):
        bg_color = '#2c3e50'
        button_color = '#3498db'
        text_color = '#ecf0f1'
        accent_color = '#e74c3c'

        self.root.configure(bg=bg_color)

        # Créer un notebook pour les onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Onglet principal
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="🔍 Analyse")

        # Onglet visualisation
        viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(viz_tab, text="📊 Visualisation")

        # Onglet paramètres
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="⚙️ Paramètres")

        # Configurer les onglets
        self.setup_main_tab(main_tab, bg_color, button_color, text_color, accent_color)
        self.setup_visualization_tab(viz_tab)
        self.setup_settings_tab(settings_tab)

    def setup_main_tab(self, parent, bg_color, button_color, text_color, accent_color):
        """Configure l'onglet principal d'analyse"""
        main_frame = tk.Frame(parent, padx=25, pady=25, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = tk.Frame(main_frame, bg=bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(header_frame,
                               text="🔍 DÉTECTEUR D'IMAGES IA AVEC DEEP LEARNING",
                               font=("Arial", 16, "bold"),
                               bg=bg_color, fg=text_color)
        title_label.pack()

        subtitle_label = tk.Label(header_frame,
                                  text="Détection des images générées par intelligence artificielle",
                                  font=("Arial", 11),
                                  bg=bg_color, fg='#bdc3c7')
        subtitle_label.pack(pady=(5, 0))

        controls_frame = tk.Frame(main_frame, bg=bg_color)
        controls_frame.pack(fill=tk.X, pady=10)

        select_btn = tk.Button(controls_frame, text="📁 Sélectionner une image",
                               command=self.select_image,
                               bg=button_color, fg="white",
                               font=("Arial", 11, "bold"),
                               padx=20, pady=10)
        select_btn.pack(side=tk.LEFT, padx=(0, 15))

        self.analyze_btn = tk.Button(controls_frame, text="🔍 Analyser",
                                     command=self.analyze_image,
                                     state=tk.DISABLED,
                                     bg=accent_color, fg="white",
                                     font=("Arial", 11, "bold"),
                                     padx=20, pady=10)
        self.analyze_btn.pack(side=tk.LEFT)

        train_btn = tk.Button(controls_frame, text="🎓 Entraîner le modèle",
                              command=self.open_training,
                              bg="#9b59b6", fg="white",
                              font=("Arial", 11),
                              padx=20, pady=10)
        train_btn.pack(side=tk.LEFT, padx=(15, 0))

        # Ajouter les boutons de gestion des modèles
        self.setup_model_management(controls_frame)

        image_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=3)
        image_frame.pack(pady=15, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame,
                                    text="Cliquez pour sélectionner une image\n\n📁",
                                    font=("Arial", 12),
                                    bg='#34495e', fg='#95a5a6',
                                    cursor="hand2")
        self.image_label.pack(expand=True)
        self.image_label.bind("<Button-1>", lambda e: self.select_image())

        self.result_label = tk.Label(main_frame, text="",
                                     font=("Arial", 16, "bold"),
                                     bg=bg_color, fg=text_color)
        self.result_label.pack(pady=15)

        details_frame = tk.LabelFrame(main_frame, text="📊 DÉTAILS DE L'ANALYSE",
                                      font=("Arial", 12, "bold"),
                                      bg=bg_color, fg=text_color)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.details_text = scrolledtext.ScrolledText(details_frame,
                                                      height=15,
                                                      font=("Consolas", 10),
                                                      wrap=tk.WORD,
                                                      bg='#ecf0f1',
                                                      fg='#2c3e50')
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_model_management(self, parent):
        """Ajoute une gestion avancée des modèles"""
        management_frame = tk.Frame(parent, bg='#2c3e50')
        management_frame.pack(fill=tk.X, pady=5)

        tk.Button(management_frame, text="🔄 Charger un modèle",
                  command=self.load_model_dialog,
                  bg="#8e44ad", fg="white",
                  font=("Arial", 10),
                  padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(management_frame, text="💾 Exporter le modèle",
                  command=self.export_model,
                  bg="#8e44ad", fg="white",
                  font=("Arial", 10),
                  padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(management_frame, text="📊 Évaluer le modèle",
                  command=self.evaluate_model,
                  bg="#8e44ad", fg="white",
                  font=("Arial", 10),
                  padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        # Ajouter le bouton d'analyse par lots
        self.batch_btn = tk.Button(management_frame, text="📂 Analyser un dossier",
                                   command=self.analyze_batch,
                                   bg="#16a085", fg="white",
                                   font=("Arial", 10),
                                   padx=15, pady=5)
        self.batch_btn.pack(side=tk.LEFT, padx=5)

    def setup_visualization_tab(self, parent):
        """Configure l'onglet de visualisation"""
        # Cadre pour les graphiques
        viz_frame = tk.Frame(parent)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Figure pour les visualisations
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Ajouter un texte d'instructions
        self.viz_label = tk.Label(viz_frame, text="Sélectionnez et analysez une image pour voir les visualisations",
                                  font=("Arial", 12), fg="gray")
        self.viz_label.pack(pady=10)

    def setup_settings_tab(self, parent):
        """Configure l'onglet des paramètres"""
        settings_frame = tk.Frame(parent, padx=20, pady=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(settings_frame, text="⚙️ PARAMÈTRES DE CONFIGURATION",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Frame pour les paramètres du modèle
        model_frame = tk.LabelFrame(settings_frame, text="🧠 Paramètres du Modèle", font=("Arial", 10, "bold"))
        model_frame.pack(fill=tk.X, pady=10, padx=5)

        row = 0
        tk.Label(model_frame, text="Taille d'image:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
        self.img_size_var = tk.StringVar(value=str(self.config.get("img_size", [128, 128])))
        tk.Entry(model_frame, textvariable=self.img_size_var, width=20).grid(row=row, column=1, pady=5, padx=5)
        row += 1

        tk.Label(model_frame, text="Taux de Dropout:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
        self.dropout_var = tk.DoubleVar(value=self.config.get("dropout_rate", 0.3))
        tk.Scale(model_frame, from_=0.1, to=0.5, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.dropout_var, length=200).grid(row=row, column=1, pady=5, padx=5)
        row += 1

        tk.Label(model_frame, text="Régularisation L2:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
        self.l2_reg_var = tk.DoubleVar(value=self.config.get("l2_reg", 0.001))
        tk.Entry(model_frame, textvariable=self.l2_reg_var, width=20).grid(row=row, column=1, pady=5, padx=5)
        row += 1

        # Frame pour les paramètres d'entraînement
        train_frame = tk.LabelFrame(settings_frame, text="🎓 Paramètres d'Entraînement", font=("Arial", 10, "bold"))
        train_frame.pack(fill=tk.X, pady=10, padx=5)

        row = 0
        self.early_stopping_var = tk.BooleanVar(value=self.config.get("use_early_stopping", True))
        tk.Checkbutton(train_frame, text="Utiliser Early Stopping",
                       variable=self.early_stopping_var).grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
        row += 1

        tk.Label(train_frame, text="Patience Early Stopping:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
        self.early_stop_patience_var = tk.IntVar(value=self.config.get("early_stopping_patience", 10))
        tk.Entry(train_frame, textvariable=self.early_stop_patience_var, width=10).grid(row=row, column=1, pady=5,
                                                                                        padx=5)
        row += 1

        self.reduce_lr_var = tk.BooleanVar(value=self.config.get("use_reduce_lr", True))
        tk.Checkbutton(train_frame, text="Réduire le Learning Rate",
                       variable=self.reduce_lr_var).grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
        row += 1

        # Frame pour les boutons
        button_frame = tk.Frame(settings_frame)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="💾 Sauvegarder les paramètres",
                  command=self.save_settings,
                  bg="#4CAF50", fg="white",
                  font=("Arial", 10, "bold"),
                  padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(button_frame, text="🔄 Restaurer les valeurs par défaut",
                  command=self.restore_defaults,
                  bg="#f44336", fg="white",
                  font=("Arial", 10),
                  padx=20, pady=10).pack(side=tk.LEFT, padx=10)

    def save_settings(self):
        """Sauvegarde les paramètres"""
        try:
            # Taille d'image
            img_size = eval(self.img_size_var.get())
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                self.config.set("img_size", list(img_size))

            # Dropout
            self.config.set("dropout_rate", self.dropout_var.get())

            # Régularisation L2
            self.config.set("l2_reg", self.l2_reg_var.get())

            # Early Stopping
            self.config.set("use_early_stopping", self.early_stopping_var.get())
            self.config.set("early_stopping_patience", self.early_stop_patience_var.get())

            # Reduce LR
            self.config.set("use_reduce_lr", self.reduce_lr_var.get())

            messagebox.showinfo("Succès", "Paramètres sauvegardés avec succès!")

            # Recharger le détecteur avec les nouveaux paramètres
            self.detector = HybridDetector(self.config)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {e}")

    def restore_defaults(self):
        """Restore les valeurs par défaut"""
        self.config = ConfigManager()
        self.detector = HybridDetector(self.config)
        messagebox.showinfo("Succès", "Valeurs par défaut restaurées!")

    def update_visualization(self, features):
        """Met à jour la visualisation avec les caractéristiques"""
        self.fig.clear()

        if not features:
            self.viz_label.config(text="Aucune donnée à visualiser")
            self.canvas.draw()
            return

        # Graphique à barres des caractéristiques principales
        ax1 = self.fig.add_subplot(221)
        feature_names = ['Contrast', 'Color Var', 'Edge Density', 'Noise Level']
        feature_values = [
            features.get('contrast', 0),
            np.mean(features.get('color_variance', [0, 0, 0])),
            features.get('edge_density', 0),
            features.get('noise_level', 0)
        ]

        bars = ax1.bar(feature_names, feature_values)
        ax1.set_title('Caractéristiques principales')
        ax1.set_ylabel('Valeur normalisée')

        # Colorier les barres en fonction des valeurs
        for i, bar in enumerate(bars):
            if feature_values[i] > np.mean(feature_values):
                bar.set_color('red')  # Valeurs élevées (potentiellement IA)
            else:
                bar.set_color('green')  # Valeurs normales (potentiellement réelles)

        # Graphique des canaux de couleur
        ax2 = self.fig.add_subplot(222)
        if 'color_variance' in features:
            color_channels = ['Red', 'Green', 'Blue']
            channel_values = features['color_variance']
            ax2.bar(color_channels, channel_values, color=['red', 'green', 'blue'])
            ax2.set_title('Variance des canaux de couleur')
            ax2.set_ylabel('Variance')

        # Graphique radar pour les caractéristiques avancées
        if 'advanced_features' in features:
            ax3 = self.fig.add_subplot(223, polar=True)
            advanced_features = features['advanced_features']
            feature_categories = ['Contrast', 'Texture', 'FFT', 'Compression']
            feature_values = [
                advanced_features.get('contrast', 0),
                advanced_features.get('lbp_entropy', 0),
                advanced_features.get('fft_mean', 0),
                advanced_features.get('compression_artifacts', 0) * 100
            ]

            # Compléter le cercle
            feature_categories += [feature_categories[0]]
            feature_values += [feature_values[0]]

            angles = np.linspace(0, 2 * np.pi, len(feature_categories), endpoint=False).tolist()
            angles += angles[:1]

            ax3.plot(angles, feature_values, 'o-', linewidth=2)
            ax3.fill(angles, feature_values, alpha=0.25)
            ax3.set_thetagrids(np.degrees(angles[:-1]), feature_categories[:-1])
            ax3.set_title('Caractéristiques avancées')

        # Texte informatif
        ax4 = self.fig.add_subplot(224)
        ax4.axis('off')

        info_text = "Interprétation des visualisations:\n\n"
        info_text += "• Barres rouges: valeurs potentiellement anormales\n"
        info_text += "• Barres vertes: valeurs dans la plage normale\n"
        info_text += "• Variance couleur: les images IA ont souvent des patterns anormaux\n"
        info_text += "• Radar: montre la cohérence entre différentes caractéristiques"

        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.fig.tight_layout()
        self.canvas.draw()
        self.viz_label.config(text="Visualisation des caractéristiques de l'image")

    def load_model_dialog(self):
        """Ouvre une boîte de dialogue pour charger un modèle"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un modèle",
            filetypes=[("Fichiers H5", "*.h5"), ("Fichiers Keras", "*.keras"), ("Tous les fichiers", "*.*")]
        )

        if file_path:
            try:
                # Charger le modèle
                self.detector.dl_detector.model = tf.keras.models.load_model(file_path, compile=False)
                self.detector.dl_detector.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                self.config.set("model_path", file_path)
                messagebox.showinfo("Succès", "Modèle chargé avec succès!")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le modèle: {e}")

    def export_model(self):
        """Exporte le modèle actuel"""
        file_path = filedialog.asksaveasfilename(
            title="Exporter le modèle",
            defaultextension=".h5",
            filetypes=[("Fichiers H5", "*.h5"), ("Fichiers Keras", "*.keras")]
        )

        if file_path:
            try:
                self.detector.dl_detector.model.save(file_path, include_optimizer=False)
                messagebox.showinfo("Succès", "Modèle exporté avec succès!")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'exporter le modèle: {e}")

    def evaluate_model(self):
        """Évalue la performance du modèle sur un jeu de test"""
        test_dir = filedialog.askdirectory(title="Sélectionner le dossier de test")
        if not test_dir:
            return

        # Séparer les images réelles et IA
        real_dir = os.path.join(test_dir, "real")
        ai_dir = os.path.join(test_dir, "ai")

        if not os.path.exists(real_dir) or not os.path.exists(ai_dir):
            messagebox.showerror("Erreur", "Le dossier de test doit contenir des sous-dossiers 'real' et 'ai'")
            return

        # Calculer les métriques
        accuracy, precision, recall, f1 = self.calculate_metrics(real_dir, ai_dir)

        # Afficher les résultats
        result_text = f"""
        📊 RÉSULTATS DE L'ÉVALUATION:

        Exactitude (Accuracy): {accuracy:.2%}
        Précision (Precision): {precision:.2%}
        Rappel (Recall): {recall:.2%}
        Score F1: {f1:.2%}
        """

        messagebox.showinfo("Performance du modèle", result_text)

    def calculate_metrics(self, real_dir, ai_dir):
        """Calcule les métriques de performance du modèle"""
        real_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            real_images.extend([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(ext)])

        ai_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            ai_images.extend([os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(ext)])

        y_true = [0] * len(real_images) + [1] * len(ai_images)
        y_pred = []

        # Désactiver le bouton pendant l'évaluation
        self.batch_btn.config(state=tk.DISABLED)

        # Prédire pour les images réelles
        for img_path in real_images:
            try:
                result = self.detector.safe_predict(img_path)
                if result and 'error' not in result:
                    y_pred.append(1 if result['is_ai_generated'] else 0)
                else:
                    y_pred.append(0)  # Par défaut, considérer comme réel en cas d'erreur
            except:
                y_pred.append(0)

        # Prédire pour les images IA
        for img_path in ai_images:
            try:
                result = self.detector.safe_predict(img_path)
                if result and 'error' not in result:
                    y_pred.append(1 if result['is_ai_generated'] else 0)
                else:
                    y_pred.append(1)  # Par défaut, considérer comme IA en cas d'erreur
            except:
                y_pred.append(1)

        # Réactiver le bouton
        self.batch_btn.config(state=tk.NORMAL)

        # Calculer les métriques
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return accuracy, precision, recall, f1

    def analyze_batch(self):
        """Analyse toutes les images d'un dossier"""
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if not folder_path:
            return

        # Créer une nouvelle fenêtre pour les résultats par lots
        batch_window = tk.Toplevel(self.root)
        batch_window.title("📊 Résultats de l'analyse par lots")
        batch_window.geometry("900x700")

        # Cadre principal
        main_frame = tk.Frame(batch_window, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Cadre pour les informations de progression
        progress_frame = tk.LabelFrame(main_frame, text="Progression", font=("Arial", 10, "bold"))
        progress_frame.pack(fill=tk.X, pady=5)

        self.batch_progress_label = tk.Label(progress_frame, text="Analyse en cours...")
        self.batch_progress_label.pack(pady=5)

        self.batch_progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.batch_progress_bar.pack(fill=tk.X, pady=5, padx=10)

        # Tableau pour afficher les résultats
        tree_frame = tk.LabelFrame(main_frame, text="Résultats", font=("Arial", 10, "bold"))
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        columns = ("Fichier", "Résultat", "Confiance", "Méthode", "Erreur")
        self.batch_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)

        for col in columns:
            self.batch_tree.heading(col, text=col)
            self.batch_tree.column(col, width=150)

        # Ajouter une barre de défilement
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.batch_tree.yview)
        self.batch_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Statistiques
        stats_frame = tk.LabelFrame(main_frame, text="Statistiques", font=("Arial", 10, "bold"))
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_label = tk.Label(stats_frame, text="En attente d'analyse...")
        self.stats_label.pack(pady=5)

        # Boutons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.export_btn = tk.Button(button_frame, text="📊 Exporter les résultats",
                                    command=lambda: self.export_batch_results(),
                                    state=tk.DISABLED,
                                    bg="#4CAF50", fg="white",
                                    padx=15, pady=5)
        self.export_btn.pack(side=tk.LEFT, padx=5)

        close_btn = tk.Button(button_frame, text="❌ Fermer",
                              command=batch_window.destroy,
                              bg="#f44336", fg="white",
                              padx=15, pady=5)
        close_btn.pack(side=tk.LEFT, padx=5)

        # Lancer l'analyse dans un thread séparé
        analysis_thread = threading.Thread(target=self.run_batch_analysis, args=(folder_path,))
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_batch_analysis(self, folder_path):
        """Exécute l'analyse par lots dans un thread séparé"""

        def update_progress(current, total):
            self.batch_progress_bar['maximum'] = total
            self.batch_progress_bar['value'] = current
            self.batch_progress_label.config(text=f"Analyse de {current}/{total} images")
            batch_window = self.batch_progress_label.winfo_toplevel()
            batch_window.update_idletasks()

        results = self.batch_analyzer.analyze_folder(folder_path, progress_callback=update_progress)

        # Mettre à jour le tableau
        ai_count = 0
        real_count = 0
        error_count = 0

        for result in results:
            if 'error' in result:
                self.batch_tree.insert("", "end", values=(
                    os.path.basename(result['file']),
                    "ERREUR",
                    "0%",
                    "N/A",
                    result['error']
                ))
                error_count += 1
            else:
                is_ai = result['is_ai_generated']
                confidence = result['confidence']

                self.batch_tree.insert("", "end", values=(
                    os.path.basename(result['file']),
                    "IA" if is_ai else "Réelle",
                    f"{confidence * 100:.2f}%",
                    result.get('method', 'N/A'),
                    ""
                ))

                if is_ai:
                    ai_count += 1
                else:
                    real_count += 1

        # Mettre à jour les statistiques
        total = len(results)
        stats_text = f"""
        📊 STATISTIQUES:

        Total d'images analysées: {total}
        Images IA détectées: {ai_count} ({ai_count / total * 100:.1f}%)
        Images réelles détectées: {real_count} ({real_count / total * 100:.1f}%)
        Erreurs: {error_count} ({error_count / total * 100:.1f}%)
        """

        self.stats_label.config(text=stats_text)
        self.batch_progress_label.config(text=f"✅ Analyse terminée: {total} images traitées")

        # Activer le bouton d'export
        self.export_btn.config(state=tk.NORMAL)

    def export_batch_results(self):
        """Exporte les résultats de l'analyse par lots"""
        file_path = filedialog.asksaveasfilename(
            title="Exporter les résultats",
            defaultextension=".csv",
            filetypes=[("Fichiers CSV", "*.csv"), ("Fichiers Excel", "*.xlsx"), ("Tous les fichiers", "*.*")]
        )

        if not file_path:
            return

        try:
            self.batch_analyzer.export_results(file_path)
            messagebox.showinfo("Succès", "Résultats exportés avec succès!")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'exporter les résultats: {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image à analyser",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            self.result_label.config(text="")
            self.details_text.delete(1.0, tk.END)
            self.viz_label.config(text="Cliquez sur 'Analyser' pour voir les visualisations")

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)
            image.thumbnail((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image: {e}")

    def analyze_image(self):
        if hasattr(self, 'current_image_path'):
            try:
                self.result_label.config(text="🔄 Analyse en cours...", fg="#f39c12")
                self.analyze_btn.config(state=tk.DISABLED)
                self.root.update()

                result = self.detector.safe_predict(self.current_image_path)

                if result and 'error' not in result:
                    color = "#e74c3c" if result['is_ai_generated'] else "#27ae60"
                    self.result_label.config(text=result['message'], fg=color)

                    details = self.format_details(result)
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.insert(tk.END, details)

                    # Journaliser l'analyse
                    features = result.get('advanced_features', result)
                    self.log_analysis(self.current_image_path, result, features)

                    # Mettre à jour la visualisation
                    self.update_visualization(features)
                elif result and 'error' in result:
                    self.result_label.config(text=f"❌ Erreur: {result['error']}", fg="#e74c3c")
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.insert(tk.END, f"Erreur lors de l'analyse: {result['error']}")
                else:
                    self.result_label.config(text="❌ Échec de l'analyse", fg="#e74c3c")

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'analyse: {e}")
            finally:
                self.analyze_btn.config(state=tk.NORMAL)

    def format_details(self, result):
        """Formatte les détails de l'analyse"""
        details = "═" * 60 + "\n"
        details += "ANALYSE PAR DEEP LEARNING HYBRIDE\n"
        details += "═" * 60 + "\n\n"

        details += f"🎯 RÉSULTAT: {result['message']}\n\n"

        details += f"📊 Méthode utilisée: {result.get('method', 'inconnue')}\n"

        if 'dl_confidence' in result:
            details += f"🔬 Confiance du modèle DL: {result['dl_confidence'] * 100:.2f}%\n"

        if 'feature_score' in result:
            details += f"📈 Score des caractéristiques: {result['feature_score']:.2f}\n"

        if 'feature_details' in result:
            details += "\n🔍 ANALYSE DES CARACTÉRISTIQUES:\n"
            for feature in result['feature_details']:
                details += f"• {feature}\n"

        if 'advanced_features' in result:
            details += "\n⚡ CARACTÉRISTIQUES AVANCÉES:\n"
            features = result['advanced_features']
            details += f"• Entropie de texture (LBP): {features.get('lbp_entropy', 'N/A'):.2f}\n"
            details += f"• Moyenne FFT: {features.get('fft_mean', 'N/A'):.2f}\n"
            details += f"• Écart-type FFT: {features.get('fft_std', 'N/A'):.2f}\n"
            details += f"• Artefacts de compression: {'✅ Oui' if features.get('compression_artifacts', False) else '❌ Non'}\n"
            details += f"• Cohérence des couleurs: {features.get('color_consistency', 'N/A'):.2f}\n"

        details += f"\n⏰ Analyse effectuée le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        details += "═" * 60 + "\n"

        details += "\n💡 Pour améliorer la précision:\n"
        details += "• Entraînez le modèle avec vos propres données\n"
        details += "• Utilisez plus d'images pour l'entraînement\n"
        details += "• Ajoutez des images IA récentes (Stable Diffusion, DALL-E, etc.)\n"

        return details

    def open_training(self):
        """Ouvre l'interface d'entraînement"""
        TrainingInterface(self.detector)


def main():
    # Créer l'application
    root = tk.Tk()
    root.title("🔍 Détecteur d'Images IA avec Deep Learning")
    root.geometry("1000x800")

    # Charger l'application
    app = IADetectorGUI(root)

    # Centre la fenêtre
    root.eval('tk::PlaceWindow . center')

    # Démarrer l'application
    root.mainloop()


if __name__ == "__main__":
    main()
