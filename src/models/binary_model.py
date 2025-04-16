import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def train_and_evaluate(data_frame):
    """
    Обучает и оценивает несколько моделей бинарной классификации.
    
    Параметры:
        data_frame (pd.DataFrame): DataFrame
    Возвращает:
    None
    """
    
    # Разделение собранных данных на признаки и целевую переменную
    X = data_frame.drop(columns=['antibiotic', 'concentration'])
    y = data_frame['antibiotic'].apply(lambda x: 0 if x=='milk' else 1).to_numpy()
    
    # Нормализация и разделение
    norma = Normalizer()
    X_normal = norma.fit_transform(X)
    # Разделение данные на обучающую и тестовую выборки (80% для обучения, 20% для тестирования)
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2, random_state=42)
    
    # Модели
    models = {
        "Logistic Regression": LogisticRegression(),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    
    # Обучение и оценка
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(X_test)
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba)
        }
    
    # Визуализация
    best_model_name = max(results, key=lambda x: results[x]["AUC"])
    best_model = models[best_model_name]

    for name, metrics in results.items():
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    print(f"Лучшая модель: {best_model_name}")
    
    # Матрица ошибок
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", ax=ax1)
    ax1.set_title(f"Матрица ошибок для {best_model_name}")
    fig1.show()
    
    # ROC-кривые
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax2.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    fig2.show()
    
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC-кривые')
    ax2.legend(loc="lower right")