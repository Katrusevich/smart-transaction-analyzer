import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from src.utils import load_and_preprocess_data, clean_text, preprocess_transactions

# Налаштування логування
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_category_model(df, description_column='cleaned_description', category_column='category',
                         model_type='logistic', use_transformers=False,
                         transformer_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Навчає модель для категоризації транзакцій.
    """
    if description_column not in df.columns or category_column not in df.columns:
        logger.error(
            f"Відсутні необхідні колонки: '{description_column}' або '{category_column}'.")
        return None, None, None, None

    X = df[description_column]
    y = df[category_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(
        f"Знайдено {len(label_encoder.classes_)} унікальних категорій: {label_encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    logger.info(
        f"Дані розбито: Тренувальних семплів: {len(X_train)}, Тестових семплів: {len(X_test)}")

    vectorizer = None
    if use_transformers:
        logger.info(
            f"Використовуємо sentence-transformers ({transformer_model_name}) для векторизації...")
        try:
            vectorizer = SentenceTransformer(transformer_model_name)
            X_train_vec = vectorizer.encode(
                X_train.tolist(), show_progress_bar=False, convert_to_tensor=True).cpu().numpy()
            X_test_vec = vectorizer.encode(
                X_test.tolist(), show_progress_bar=False, convert_to_tensor=True).cpu().numpy()
        except Exception as e:
            logger.error(
                f"Помилка при завантаженні SentenceTransformer: {e}. Встановіть sentence-transformers.")
            return None, None, None, None
    else:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        logger.info(
            f"Текст векторизовано (TF-IDF). Розмірність ознак: {X_train_vec.shape[1]}")

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'xgboost':
        model = XGBClassifier(
            random_state=42, eval_metric='mlogloss', n_jobs=-1)
    else:
        logger.error(
            f"Непідтримуваний тип моделі: {model_type}")
        return None, None, None, None

    logger.info(
        f"Починаємо навчання моделі категоризації ({model_type})...")
    model.fit(X_train_vec, y_train)
    logger.info(
        "Навчання моделі категоризації завершено.")

    logger.info(
        f"Виконуємо крос-валідацію ({model_type})...")
    if not use_transformers and model_type in ['random_forest', 'xgboost'] and X_train_vec.shape[1] > 1000:
        logger.warning(
            "Перетворення в густу матрицю може споживати багато пам’яті.")
        X_train_vec_dense = X_train_vec.toarray()
        cv_scores = cross_val_score(
            model, X_train_vec_dense, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    else:
        cv_scores = cross_val_score(
            model, X_train_vec, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    logger.info(
        f"Крос-валідація (5 фолдів): середня точність = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}")

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))

    logger.info(
        f"Точність моделі ({model_type}) на тестовому наборі: {accuracy:.4f}")
    logger.info(f"\nЗвіт класифікації:\n{report}")
    logger.info(
        f"\nМатриця плутанини (порядок: {label_encoder.classes_.tolist()}):\n{conf_matrix}")

    vectorizer_or_name = transformer_model_name if use_transformers else vectorizer
    return model, vectorizer_or_name, label_encoder.classes_, label_encoder


def predict_category(text, model, vectorizer_or_name, label_encoder=None, use_transformers=False):
    """
    Прогнозує категорію для нового текстового опису.
    """
    if not model or not vectorizer_or_name:
        logger.error(
            "Модель або векторизатор не завантажені/навчені.")
        return "Невідома"

    if use_transformers:
        try:
            vectorizer = SentenceTransformer(vectorizer_or_name)
        except Exception as e:
            logger.error(
                f"Помилка при завантаженні SentenceTransformer ({vectorizer_or_name}): {e}")
            return "Невідома"
    else:
        vectorizer = vectorizer_or_name

    cleaned_text = clean_text(
        text, language='ukrainian', remove_stopwords=True)
    if use_transformers:
        text_vec = vectorizer.encode(
            [cleaned_text], show_progress_bar=False, convert_to_tensor=True).cpu().numpy()
    else:
        text_vec = vectorizer.transform([cleaned_text])

    prediction_encoded = model.predict(text_vec)[0]
    if label_encoder:
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    else:
        prediction = prediction_encoded
    return prediction


def add_anomaly_features(df):
    """
    Додає нові ознаки для виявлення аномалій.
    """
    if df.empty:
        logger.warning(
            "DataFrame порожній, неможливо додати ознаки аномалій.")
        return df

    if 'category' not in df.columns or 'date' not in df.columns or 'amount' not in df.columns:
        logger.error(
            "Потрібні колонки 'category', 'date' та 'amount'.")
        return df

    df_copy = df.copy()
    if not df_copy['category'].empty and len(df_copy['category'].unique()) > 1:
        df_copy['mean_amount_by_category'] = df_copy.groupby(
            'category')['amount'].transform('mean')
        df_copy['amount_deviation'] = df_copy['amount'] - \
            df_copy['mean_amount_by_category']
    else:
        df_copy['mean_amount_by_category'] = df_copy['amount'].mean()
        df_copy['amount_deviation'] = df_copy['amount'] - \
            df_copy['mean_amount_by_category']
        logger.warning(
            "Недостатньо категорій для 'mean_amount_by_category'. Використано загальне середнє.")

    df_copy['date_only'] = df_copy['date'].dt.date
    df_copy['transaction_count_per_day'] = df_copy.groupby(
        'date_only')['date_only'].transform('count')
    df_copy.drop('date_only', axis=1, inplace=True)

    logger.info(
        "Додано ознаки: mean_amount_by_category, amount_deviation, transaction_count_per_day.")
    return df_copy


def train_anomaly_model(df, initial_features_for_anomaly, contamination=None):
    """
    Навчає модель для виявлення аномалій (Isolation Forest).
    """
    df_extended = add_anomaly_features(df.copy())
    features_for_anomaly = initial_features_for_anomaly + \
        ['mean_amount_by_category', 'amount_deviation', 'transaction_count_per_day']

    for col in features_for_anomaly:
        if col not in df_extended.columns:
            logger.error(f"Відсутня колонка: '{col}'.")
            return None, None

    X_anomaly = df_extended[features_for_anomaly]
    scaler = StandardScaler()
    X_anomaly_scaled = scaler.fit_transform(X_anomaly)
    logger.info(
        "Ознаки для аномалій нормалізовано.")

    if contamination is None and 'is_anomaly' in df_extended.columns:
        contamination = df_extended['is_anomaly'].mean()
        logger.info(
            f"Параметр contamination розрахований автоматично: {contamination:.4f}")
    elif contamination is None:
        contamination = 0.01
        logger.info(
            f"Параметр contamination не задано і немає 'is_anomaly'. Використано значення за замовчуванням: {contamination}")

    model = IsolationForest(
        random_state=42, contamination=contamination, n_jobs=-1)
    logger.info(
        f"Починаємо навчання моделі аномалій на {len(X_anomaly_scaled)} семплах...")
    try:
        model.fit(X_anomaly_scaled)
        logger.info(
            "Навчання моделі аномалій завершено.")
    except Exception as e:
        logger.error(
            f"Помилка при навчанні моделі аномалій: {e}")
        return None, None

    anomaly_scores = model.decision_function(X_anomaly_scaled)
    predictions = model.predict(X_anomaly_scaled)

    if 'is_anomaly' in df_extended.columns:
        true_labels = df_extended['is_anomaly'].astype(int)
        pred_labels = (predictions == -1).astype(int)
        if true_labels.sum() == 0 and pred_labels.sum() == 0:
            logger.info(
                "Немає аномалій. Метрики не застосовуються.")
            precision, recall, f1 = np.nan, np.nan, np.nan
        elif true_labels.sum() == 0:
            logger.warning("Немає істинних аномалій.")
            precision, recall, f1 = 0.0, 0.0, 0.0
        elif pred_labels.sum() == 0:
            logger.warning(
                "Модель не виявила аномалій.")
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels)

        logger.info(f"Оцінка моделі аномалій:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")
        logger.info(
            f"  Істинних аномалій: {true_labels.sum()}")
        logger.info(
            f"  Виявлено аномалій: {pred_labels.sum()}")

    return model, scaler


def predict_anomaly(data_point, model, scaler=None):
    """
    Прогнозує, чи є транзакція аномальною.
    """
    if not model:
        logger.error(
            "Модель аномалій не завантажена/навчена.")
        return False

    if isinstance(data_point, pd.Series):
        data_point_df = data_point.to_frame().T
    elif isinstance(data_point, np.ndarray):
        data_point_df = pd.DataFrame(data_point.reshape(1, -1))
    else:
        logger.error(
            f"Непідтримуваний тип data_point: {type(data_point)}.")
        return False

    if scaler:
        try:
            data_point_scaled = scaler.transform(data_point_df)
        except Exception as e:
            logger.error(
                f"Помилка при масштабуванні data_point: {e}")
            return False
    else:
        data_point_scaled = data_point_df

    prediction = model.predict(data_point_scaled)[0]
    return prediction == -1


def save_model(model, path, metadata=None):
    """Зберігає модель і метадані на диск."""
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data_to_save = {'model': model}
        if metadata:
            metadata_copy = metadata.copy()
            if 'vectorizer' in metadata_copy and isinstance(metadata_copy['vectorizer'], SentenceTransformer):
                metadata_copy['transformer_model_name'] = metadata_copy['vectorizer']._model_name_or_path
                del metadata_copy['vectorizer']
            data_to_save.update(metadata_copy)
        joblib.dump(data_to_save, path)
        logger.info(f"Модель збережено до: {path}")
        return True
    except Exception as e:
        logger.error(
            f"Помилка при збереженні моделі до {path}: {e}")
        return False


def load_model(path):
    """Завантажує модель і метадані з диска."""
    try:
        loaded_data = joblib.load(path)
        model = loaded_data['model']
        metadata = {key: value for key,
                    value in loaded_data.items() if key != 'model'}
        logger.info(f"Модель завантажено з: {path}")
        return model, metadata
    except Exception as e:
        logger.error(
            f"Помилка при завантаженні моделі з {path}: {e}")
        return None, None


if __name__ == "__main__":
    logger.info(
        "Запуск модуля AI-аналізу для демонстрації...")

    data_file = os.path.join('data', 'transactions.csv')
    os.makedirs(os.path.dirname(data_file) or '.', exist_ok=True)
    df = load_and_preprocess_data(data_file)
    if df is None:
        logger.error("Не вдалося завантажити дані.")
    else:
        important_keywords = ['атб', 'comfy', 'mono', 'супермаркет',
                              'кава', 'аптека', 'зарплата']
        df_cleaned = preprocess_transactions(
            df.copy(), language='ukrainian', keep_keywords=important_keywords)
        if df_cleaned is None:
            logger.error("Не вдалося очистити описи.")
        else:
            logger.info(
                f"Дані оброблено. Розмір: {df_cleaned.shape}")

            # Категоризація
            logger.info(
                "\n--- Навчання моделі категоризації (XGBoost) ---")
            category_model, vectorizer_or_name, categories, label_encoder = train_category_model(
                df_cleaned, model_type='xgboost', use_transformers=False
            )
            if category_model and vectorizer_or_name and label_encoder:
                logger.info(
                    "\n--- Тестування моделі категоризації ---")
                sample_texts = [
                    "Покупка в супермаркеті АТБ",
                    "Отримання зарплати на картку",
                    "Відвідання кінотеатру",
                    "Купив книжку в інтернет-магазині",
                    "Кава в кав'ярні Lviv Croissants",
                    "Нарахування кешбеку",
                    "Комунальні платежі за світло"
                ]
                for text in sample_texts:
                    predicted_category = predict_category(
                        text, category_model, vectorizer_or_name, label_encoder, use_transformers=False)
                    logger.info(
                        f"'{text}' -> Категорія: {predicted_category}")

                save_model(category_model, 'models/category_model.pkl',
                           metadata={'vectorizer': vectorizer_or_name, 'label_encoder': label_encoder, 'use_transformers': False})

            # Аномалії
            logger.info(
                "\n--- Навчання моделі аномалій ---")
            features_for_anomaly = [
                'amount', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
            anomaly_model, anomaly_scaler = train_anomaly_model(
                df_cleaned, features_for_anomaly, contamination=None)
            if anomaly_model and anomaly_scaler:
                logger.info(
                    "\n--- Тестування моделі аномалій ---")
                all_anomaly_features = features_for_anomaly + \
                    ['mean_amount_by_category', 'amount_deviation',
                        'transaction_count_per_day']
                normal_tx = pd.Series({
                    'amount': -500.0, 'hour_of_day': 12, 'day_of_week': 2, 'month': 7, 'is_weekend': 0,
                    'mean_amount_by_category': -450.0, 'amount_deviation': -50.0, 'transaction_count_per_day': 15
                }, index=all_anomaly_features)
                anomaly_tx = pd.Series({
                    'amount': -50000.0, 'hour_of_day': 3, 'day_of_week': 0, 'month': 1, 'is_weekend': 0,
                    'mean_amount_by_category': -450.0, 'amount_deviation': -49550.0, 'transaction_count_per_day': 2
                }, index=all_anomaly_features)

                logger.info(
                    f"Нормальна транзакція: {predict_anomaly(normal_tx, anomaly_model, anomaly_scaler)}")
                logger.info(
                    f"Аномальна транзакція: {predict_anomaly(anomaly_tx, anomaly_model, anomaly_scaler)}")

                save_model(anomaly_model, 'models/anomaly_model.pkl',
                           metadata={'scaler': anomaly_scaler})

            logger.info("\nДемонстрація завершена.")
