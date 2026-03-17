# Régression Linéaire Simple à partir de zéro

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Calcul%20Scientifique-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-green.svg)](https://matplotlib.org/)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-green.svg)](https://opensource.org/licenses/MIT)

Une implémentation complète de la régression linéaire simple construite à partir des bases mathématiques, avec des métriques d’évaluation complètes et une comparaison directe avec scikit-learn. Ce projet illustre tout le pipeline ML, de la théorie à la validation.

## 🎯 Vue d'ensemble du projet

Ce dépôt implémente une régression linéaire simple en utilisant uniquement NumPy, puis la valide avec scikit-learn.

**Fondement mathématique :** `y = ax + b`

| Composant                    | Implémentation                                |
| ---------------------------- | --------------------------------------------- |
| **Pente (a)**                | `Σ[(xi-x̄)(yi-ȳ)] / Σ[(xi-x̄)²]`              |
| **Ordonnée à l’origine (b)** | `ȳ - a·x̄`                                    |
| **Évaluation**               | MSE, RMSE, MAE, R² (implémentés manuellement) |
| **Validation**               | Comparaison exacte avec scikit-learn          |

## 💼 Pourquoi c'est important

| Compétence                   | Valeur                                               |
| ---------------------------- | ---------------------------------------------------- |
| Implémentation d’algorithmes | Capacité à construire et optimiser sans dépendances  |
| Rigueur mathématique         | Comprendre le *pourquoi*, pas seulement le *comment* |
| Validation de modèle         | Vérification via métriques codées à la main          |
| Conversion de code           | Migration MATLAB → Python                            |
| Calcul scientifique          | Utilisation efficace de NumPy                        |

## 🚀 Points techniques clés

```python
a = np.sum(X * Y) / np.sum(X ** 2)
b = y_bar - (a * x_bar)

mse = np.mean((y_true - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
```

### Fonctionnalités

* Implémentation pure NumPy
* Calcul manuel des métriques
* Validation exacte avec scikit-learn
* Visualisation comparative
* Code propre et modulaire

## 📊 Cas d'utilisation : Prédiction immobilière

Surface → Prix

Exemple :

```python
predict(130)  # 339.29 ($k)
```

## 📈 Performance

* R² = 0.9649 (96.49% de variance expliquée)
* Résultats identiques à scikit-learn

## 🛠️ Lancement rapide

```bash
git clone https://github.com/N1N0u/Simple-Linear-regresson.git
cd Simple-Linear-regresson
python linear_regression.py
```

## 🎓 Concepts appris

* Algèbre linéaire
* Statistiques
* Optimisation
* Métriques ML
* Validation scientifique

## 📝 Licence

Licence MIT — utilisation libre.

---

Construit à partir de zéro. Validé selon les standards industriels.

Auteur : N1N0u
