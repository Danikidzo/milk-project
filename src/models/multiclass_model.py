import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def train_multiclass_models(data_frame, target_column='antibiotic', class_names=None):
    """
    Обучает и оценивает модели многоклассовой классификации.
    
    Параметры:
    data_frame (pd.DataFrame): DataFrame
    
    Возвращает:
    None
    """
    # Подготовка данных
    X = data_frame.drop(columns=['antibiotic', 'concentration'])
    y = data_frame['antibiotic']
    
    # Кодирование целевой переменной
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Нормализация и разделение
    norma = Normalizer()
    X_normal = norma.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_encoded, test_size=0.2)
    
    # Модели
    models = {
        "Logistic Regression": OneVsRestClassifier(LogisticRegression()),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": OneVsRestClassifier(SVC(probability=True)),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    
    # Обучение и оценка
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        results[name] = {
            "Accuracy": accuracy,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print()
    
    # Определение лучшей модели
    best_model_name = max(results, key=lambda x: results[x]["AUC"])
    best_model = models[best_model_name]
    print(f"Лучшая модель: {best_model_name}")
    
    # Матрица ошибок
    y_pred_best = best_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Матрица ошибок для лучшей модели")
    plt.show()