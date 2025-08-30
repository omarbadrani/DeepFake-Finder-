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
from sklearn.model_selection import train_test_split
import threading
import time
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Activer l'exécution eager de TensorFlow
tf.config.run_functions_eagerly(True)


class DeepLearningDetector:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.load_or_create_model()

    def create_improved_model(self):
        """Crée un modèle CNN amélioré avec des techniques modernes"""
        model = models.Sequential([
            # Couche de preprocessing
            layers.Rescaling(1. / 255, input_shape=(128, 128, 3)),

            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classification
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
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

    def create_augmentation_layer(self):
        """Crée une couche d'augmentation de données"""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])

    def setup_pretrained_models(self):
        """Intègre des modèles pré-entraînés pour améliorer la détection"""
        # Charger EfficientNet pré-entrainé
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(128, 128, 3)
        )

        # Geler les couches de base
        base_model.trainable = False

        # Ajouter des couches de classification
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def load_or_create_model(self):
        """Charge un modèle existant ou en crée un nouveau"""
        model_path = "ia_image_detector.h5"

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
                print("Modèle de détection IA chargé et recompilé")
            except Exception as e:
                print(f"Erreur lors du chargement: {e}, création d'un nouveau modèle amélioré")
                self.model = self.create_improved_model()
        else:
            print("Création d'un nouveau modèle amélioré")
            self.model = self.create_improved_model()

    def preprocess_image(self, image_path):
        """Prétraite une image pour la prédiction"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Erreur lors du prétraitement: {e}")
            return None

    def predict(self, image_path):
        """Prédit si une image est générée par IA avec le modèle deep learning"""
        processed_img = self.preprocess_image(image_path)

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

    def train_model(self, real_images_dir, ai_images_dir, epochs=10, progress_callback=None):
        """Entraîne le modèle avec des images réelles et IA"""
        # Charger les images
        X, y = [], []

        # Vérifier et lister les images dans les dossiers
        real_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            real_images.extend([f for f in os.listdir(real_images_dir) if f.lower().endswith(ext)])

        ai_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            ai_images.extend([f for f in os.listdir(ai_images_dir) if f.lower().endswith(ext)])

        print(f"Images réelles trouvées: {len(real_images)}")
        print(f"Images IA trouvées: {len(ai_images)}")

        if len(real_images) == 0 or len(ai_images) == 0:
            raise ValueError(f"Aucune image trouvée. Réelles: {len(real_images)}, IA: {len(ai_images)}")

        # Charger les images réelles
        if progress_callback:
            progress_callback("Chargement des images réelles...", 10)

        for i, filename in enumerate(real_images):
            img_path = os.path.join(real_images_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(0)  # 0 pour images réelles
            except Exception as e:
                print(f"Erreur avec {filename}: {e}")

            if progress_callback and i % 10 == 0:
                progress = 10 + int(30 * i / len(real_images))
                progress_callback(f"Images réelles: {i + 1}/{len(real_images)}", progress)

        # Charger les images IA
        if progress_callback:
            progress_callback("Chargement des images IA...", 40)

        for i, filename in enumerate(ai_images):
            img_path = os.path.join(ai_images_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(1)  # 1 pour images IA
            except Exception as e:
                print(f"Erreur avec {filename}: {e}")

            if progress_callback and i % 10 == 0:
                progress = 40 + int(30 * i / len(ai_images))
                progress_callback(f"Images IA: {i + 1}/{len(ai_images)}", progress)

        # Vérifier qu'on a des données
        if len(X) == 0:
            raise ValueError("Aucune image valide trouvée dans les dossiers spécifiés")

        # Convertir en arrays numpy
        X = np.array(X)
        y = np.array(y)

        # Mélanger les données
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Diviser en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if progress_callback:
            progress_callback("Début de l'entraînement...", 70)

        # Entraîner le modèle
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=32,
            verbose=0,
            callbacks=[TrainingCallback(progress_callback, 70, 25)] if progress_callback else None
        )

        # Sauvegarder le modèle (sans l'optimiseur pour éviter les problèmes)
        self.model.save("ia_image_detector.h5", include_optimizer=False)

        if progress_callback:
            progress_callback("Entraînement terminé!", 95)
            time.sleep(1)
            progress_callback("Sauvegarde du modèle...", 100)

        # Afficher les résultats - conversion explicite des valeurs
        train_accuracy = float(history.history['accuracy'][-1])
        val_accuracy = float(history.history['val_accuracy'][-1])

        print(f"Précision sur l'ensemble d'entraînement: {train_accuracy * 100:.2f}%")
        print(f"Précision sur l'ensemble de validation: {val_accuracy * 100:.2f}%")

        return history, X_test, y_test


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


class HybridDetector:
    def __init__(self):
        self.dl_detector = DeepLearningDetector()
        self.img_size = (256, 256)

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


class TrainingInterface:
    def __init__(self, detector):
        self.detector = detector
        self.setup_training_ui()

    def setup_training_ui(self):
        """Interface pour l'entraînement du modèle"""
        self.train_window = tk.Toplevel()
        self.train_window.title("Entraînement du Modèle")
        self.train_window.geometry("700x600")
        self.train_window.resizable(False, False)

        # Centrer la fenêtre
        self.train_window.transient(self.train_window.master)
        self.train_window.grab_set()

        main_frame = tk.Frame(self.train_window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="Entraînement du Modèle de Détection IA",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Frame pour les chemins
        paths_frame = tk.Frame(main_frame)
        paths_frame.pack(fill=tk.X, pady=10)

        tk.Label(paths_frame, text="Dossier d'images réelles:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.real_path_entry = tk.Entry(paths_frame, width=40)
        self.real_path_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(paths_frame, text="Parcourir",
                  command=lambda: self.browse_folder(self.real_path_entry)).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(paths_frame, text="Dossier d'images IA:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ai_path_entry = tk.Entry(paths_frame, width=40)
        self.ai_path_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(paths_frame, text="Parcourir",
                  command=lambda: self.browse_folder(self.ai_path_entry)).grid(row=1, column=2, padx=5, pady=5)

        # Info sur les images trouvées
        self.info_label = tk.Label(main_frame, text="Sélectionnez les dossiers pour voir le nombre d'images", fg="gray")
        self.info_label.pack(pady=5)

        # Frame pour les paramètres
        params_frame = tk.Frame(main_frame)
        params_frame.pack(fill=tk.X, pady=10)

        tk.Label(params_frame, text="Nombre d'époques:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_entry = tk.Entry(params_frame, width=10)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        tk.Label(params_frame, text="Taille du batch:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.batch_entry = tk.Entry(params_frame, width=10)
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        # Options avancées
        self.use_augmentation = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="Utiliser l'augmentation de données",
                       variable=self.use_augmentation).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)

        # Barre de progression
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)

        self.progress_label = tk.Label(progress_frame, text="Prêt à entraîner", fg="blue")
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)

        # Boutons
        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=20)

        self.train_button = tk.Button(buttons_frame, text="Commencer l'Entraînement",
                                      command=self.start_training, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.train_button.pack(side=tk.LEFT, padx=10)

        self.cancel_button = tk.Button(buttons_frame, text="Annuler",
                                       command=self.train_window.destroy, bg="#f44336", fg="white")
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
            self.info_label.config(text=f"Images trouvées: {real_count} réelles, {ai_count} IA", fg="green")
        else:
            self.info_label.config(text="Aucune image trouvée (formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp)",
                                   fg="red")

    def update_progress(self, message, value):
        """Met à jour la barre de progression et le message"""
        self.progress_label.config(text=message)
        self.progress_bar['value'] = value
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

        # Désactiver le bouton pendant l'entraînement
        self.train_button.config(state=tk.DISABLED)

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
            history, X_test, y_test = self.detector.dl_detector.train_model(
                real_path, ai_path, epochs=epochs,
                progress_callback=self.update_progress
            )

            # Évaluer le modèle
            y_pred = (self.detector.dl_detector.model.predict(X_test) > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Afficher les résultats
            train_acc = float(history.history['accuracy'][-1]) * 100
            val_acc = float(history.history['val_accuracy'][-1]) * 100

            # Réactiver le bouton
            self.train_button.config(state=tk.NORMAL)

            # Afficher un message de succès
            messagebox.showinfo("Entraînement terminé",
                                f"Modèle entraîné avec succès!\n\n"
                                f"Précision entraînement: {train_acc:.2f}%\n"
                                f"Précision validation: {val_acc:.2f}%\n\n"
                                f"Métriques détaillées:\n"
                                f"Exactitude: {accuracy:.2f}\n"
                                f"Précision: {precision:.2f}\n"
                                f"Rappel: {recall:.2f}\n"
                                f"Score F1: {f1:.2f}")

        except Exception as e:
            # Réactiver le bouton en cas d'erreur
            self.train_button.config(state=tk.NORMAL)
            messagebox.showerror("Erreur", f"Erreur lors de l'entraînement: {str(e)}")


class IADetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Détecteur d'Images IA avec Deep Learning")
        self.root.geometry("1000x800")
        self.root.configure(bg='#2c3e50')

        self.detector = HybridDetector()
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
                features['contrast'],
                np.mean(features['color_variance']),
                features['edge_density'],
                features['noise_level']
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
        self.notebook.add(main_tab, text="Analyse")

        # Onglet visualisation
        viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(viz_tab, text="Visualisation")

        # Configurer l'onglet principal
        self.setup_main_tab(main_tab, bg_color, button_color, text_color, accent_color)

        # Configurer l'onglet de visualisation
        self.setup_visualization_tab(viz_tab)

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
            features['contrast'],
            np.mean(features['color_variance']),
            features['edge_density'],
            features['noise_level']
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
        color_channels = ['Red', 'Green', 'Blue']
        channel_values = features['color_variance']
        ax2.bar(color_channels, channel_values, color=['red', 'green', 'blue'])
        ax2.set_title('Variance des canaux de couleur')
        ax2.set_ylabel('Variance')

        # Graphique radar pour les caractéristiques avancées (si disponibles)
        if hasattr(features, 'advanced_features'):
            ax3 = self.fig.add_subplot(223, polar=True)
            advanced_features = features['advanced_features']
            feature_categories = ['Contrast', 'Texture', 'FFT', 'Compression']
            feature_values = [
                advanced_features.get('contrast', 0),
                advanced_features.get('lbp_entropy', 0),
                advanced_features.get('fft_mean', 0),
                advanced_features.get('compression_artifacts', 0) * 100  # Convertir booléen en valeur
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
            filetypes=[("Fichiers H5", "*.h5"), ("Tous les fichiers", "*.*")]
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
                messagebox.showinfo("Succès", "Modèle chargé avec succès!")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le modèle: {e}")

    def export_model(self):
        """Exporte le modèle actuel"""
        file_path = filedialog.asksaveasfilename(
            title="Exporter le modèle",
            defaultextension=".h5",
            filetypes=[("Fichiers H5", "*.h5")]
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

        Exactitude (Accuracy): {accuracy:.2f}%
        Précision (Precision): {precision:.2f}%
        Rappel (Recall): {recall:.2f}%
        Score F1: {f1:.2f}%
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
                result = self.detector.predict_image(img_path)
                if result:
                    y_pred.append(1 if result['is_ai_generated'] else 0)
                else:
                    y_pred.append(0)  # Par défaut, considérer comme réel en cas d'erreur
            except:
                y_pred.append(0)

        # Prédire pour les images IA
        for img_path in ai_images:
            try:
                result = self.detector.predict_image(img_path)
                if result:
                    y_pred.append(1 if result['is_ai_generated'] else 0)
                else:
                    y_pred.append(1)  # Par défaut, considérer comme IA en cas d'erreur
            except:
                y_pred.append(1)

        # Réactiver le bouton
        self.batch_btn.config(state=tk.NORMAL)

        # Calculer les métriques
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1

    def analyze_batch(self):
        """Analyse toutes les images d'un dossier"""
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if not folder_path:
            return

        # Créer une nouvelle fenêtre pour les résultats par lots
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Résultats de l'analyse par lots")
        batch_window.geometry("800x600")

        # Cadre pour les informations de progression
        progress_frame = tk.Frame(batch_window)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)

        progress_label = tk.Label(progress_frame, text="Analyse en cours...")
        progress_label.pack()

        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)

        # Tableau pour afficher les résultats
        tree_frame = tk.Frame(batch_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("Fichier", "Résultat", "Confiance")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200)

        # Ajouter une barre de défilement
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Analyser chaque image
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])

        total_files = len(image_files)
        ai_count = 0

        for i, file in enumerate(image_files):
            file_path = os.path.join(folder_path, file)
            try:
                result = self.detector.predict_image(file_path)

                if result:
                    is_ai = result['is_ai_generated']
                    confidence = result['confidence']

                    tree.insert("", "end", values=(
                        file,
                        "IA" if is_ai else "Réelle",
                        f"{confidence * 100:.2f}%"
                    ))

                    if is_ai:
                        ai_count += 1

                # Mettre à jour la barre de progression
                progress = (i + 1) / total_files * 100
                progress_bar['value'] = progress
                progress_label.config(text=f"Analyse de {i + 1}/{total_files} images")
                batch_window.update()

            except Exception as e:
                print(f"Erreur avec {file}: {e}")
                tree.insert("", "end", values=(file, "Erreur", "N/A"))

        # Afficher le résumé
        summary_text = f"Analyse terminée: {ai_count} images IA détectées sur {total_files} images analysées"
        progress_label.config(text=summary_text)

        # Ajouter un bouton pour exporter les résultats
        export_btn = tk.Button(batch_window, text="📊 Exporter les résultats",
                               command=lambda: self.export_batch_results(tree))
        export_btn.pack(pady=10)

    def export_batch_results(self, tree):
        """Exporte les résultats de l'analyse par lots"""
        file_path = filedialog.asksaveasfilename(
            title="Exporter les résultats",
            defaultextension=".csv",
            filetypes=[("Fichiers CSV", "*.csv")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Fichier", "Résultat", "Confiance"])

                for item in tree.get_children():
                    values = tree.item(item, 'values')
                    writer.writerow(values)

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

                result = self.detector.predict_image(self.current_image_path)

                if result:
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

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'analyse: {e}")
            finally:
                self.analyze_btn.config(state=tk.NORMAL)

    def format_details(self, result):
        """Formatte les détails de l'analyse"""
        details = "═" * 60 + "\n"
        details += "ANALYSE PAR DEEP LEARNING\n"
        details += "═" * 60 + "\n\n"

        details += f"RÉSULTAT: {result['message']}\n\n"

        details += f"Méthode utilisée: {result.get('method', 'inconnue')}\n"

        if 'dl_confidence' in result:
            details += f"Confiance du modèle DL: {result['dl_confidence'] * 100:.2f}%\n"

        if 'feature_score' in result:
            details += f"Score des caractéristiques: {result['feature_score']:.2f}\n"

        if 'feature_details' in result:
            details += "\nANALYSE DES CARACTÉRISTIQUES:\n"
            for feature in result['feature_details']:
                details += f"- {feature}\n"

        if 'advanced_features' in result:
            details += "\nCARACTÉRISTIQUES AVANCÉES:\n"
            features = result['advanced_features']
            details += f"- Entropie de texture (LBP): {features.get('lbp_entropy', 'N/A'):.2f}\n"
            details += f"- Moyenne FFT: {features.get('fft_mean', 'N/A'):.2f}\n"
            details += f"- Écart-type FFT: {features.get('fft_std', 'N/A'):.2f}\n"
            details += f"- Artefacts de compression: {'Oui' if features.get('compression_artifacts', False) else 'Non'}\n"
            details += f"- Cohérence des couleurs: {features.get('color_consistency', 'N/A'):.2f}\n"

        details += f"\n⏰ Analyse effectuée le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        details += "═" * 60 + "\n"

        details += "\n💡 Pour améliorer la précision:\n"
        details += "- Entraînez le modèle avec vos propres données\n"
        details += "- Utilisez plus d'images pour l'entraînement\n"
        details += "- Ajoutez des images IA récentes (Stable Diffusion, DALL-E, etc.)\n"

        return details

    def open_training(self):
        """Ouvre l'interface d'entraînement"""
        TrainingInterface(self.detector)


def main():
    # Désactiver les messages TensorFlow si nécessaire
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Afficher directement l'interface graphique
    root = tk.Tk()
    app = IADetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()