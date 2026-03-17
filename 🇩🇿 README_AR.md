# الانحدار الخطي البسيط من الصفر

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-حوسبة%20علمية-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-تصوير%20بياني-green.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

تنفيذ كامل لخوارزمية الانحدار الخطي البسيط انطلاقًا من الأسس الرياضية، مع حساب جميع مؤشرات التقييم ومقارنة مباشرة مع scikit-learn.

## 🎯 نظرة عامة على المشروع

هذا المشروع يطبق الانحدار الخطي باستخدام NumPy فقط، ثم يتم التحقق منه باستخدام scikit-learn.

**المعادلة الأساسية:** `y = ax + b`

| المكون     | التنفيذ                          |
| ---------- | -------------------------------- |
| الميل (a)  | `Σ[(xi-x̄)(yi-ȳ)] / Σ[(xi-x̄)²]` |
| الثابت (b) | `ȳ - a·x̄`                       |
| التقييم    | MSE, RMSE, MAE, R²               |
| التحقق     | مقارنة دقيقة مع scikit-learn     |

## 💼 لماذا هذا مهم

| المهارة           | القيمة                              |
| ----------------- | ----------------------------------- |
| بناء الخوارزميات  | فهم عميق بدون الاعتماد على المكتبات |
| الفهم الرياضي     | معرفة سبب عمل النموذج               |
| التحقق من النتائج | تطبيق المقاييس يدويًا               |
| تحويل الأكواد     | MATLAB إلى Python                   |
| الحوسبة العلمية   | استخدام NumPy بكفاءة                |

## 🚀 النقاط التقنية

```python
a = np.sum(X * Y) / np.sum(X ** 2)
b = y_bar - (a * x_bar)

mse = np.mean((y_true - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
```

## 📊 مثال عملي: التنبؤ بأسعار العقارات

المساحة → السعر

مثال:

```python
predict(130)  # 339.29 ألف دولار
```

## 📈 أداء النموذج

* R² = 0.9649
* نتائج مطابقة لـ scikit-learn

## 🛠️ التشغيل

```bash
git clone https://github.com/N1N0u/Simple-Linear-regresson.git
cd Simple-Linear-regresson
python linear_regression.py
```

## 🎓 ما ستتعلمه

* الجبر الخطي
* الإحصاء
* التحسين الرياضي
* تقييم النماذج
* التحقق العلمي

## 📝 الرخصة

MIT — استخدام حر.

---

تم البناء من الصفر والتحقق وفق المعايير الصناعية.

المؤلف: N1N0u
