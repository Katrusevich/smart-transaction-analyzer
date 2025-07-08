import os
import pandas as pd
import logging
import joblib
from datetime import datetime
from src.ai_analyzer import train_category_model, predict_category, train_anomaly_model, predict_anomaly, save_model, load_model
from src.utils import load_and_preprocess_data, preprocess_transactions, clean_text

# Налаштування логування
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_or_train_models(df_cleaned):
    """Завантажує моделі з models/ або навчає їх, якщо їх немає."""
    model_path_cat = 'models/category_model.pkl'
    model_path_anomaly = 'models/anomaly_model.pkl'

    # Перевірка необхідних стовпців
    required_columns = ['cleaned_description', 'category',
                        'amount', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
    missing_columns = [
        col for col in required_columns if col not in df_cleaned.columns]
    if missing_columns:
        logger.error(
            f"Відсутні необхідні стовпці в даних: {missing_columns}")
        return None, None, None, None, None

    # Завантаження моделі категоризації
    if os.path.exists(model_path_cat):
        logger.info(
            "Завантаження моделі категоризації...")
        category_model, metadata = load_model(model_path_cat)
        vectorizer = metadata.get('vectorizer')
        label_encoder = metadata.get('label_encoder')
        use_transformers = metadata.get('use_transformers', False)
    else:
        logger.info(
            "Модель категоризації відсутня. Навчаємо нову...")
        category_model, vectorizer, classes, label_encoder = train_category_model(
            df_cleaned, model_type='xgboost', use_transformers=False
        )
        if category_model:
            save_model(category_model, model_path_cat, metadata={
                       'vectorizer': vectorizer, 'label_encoder': label_encoder, 'use_transformers': False})
        else:
            logger.error(
                "Не вдалося навчити модель категоризації через помилку в даних або алгоритмі.")
            return None, None, None, None, None

    # Завантаження моделі аномалій
    if os.path.exists(model_path_anomaly):
        logger.info("Завантаження моделі аномалій...")
        anomaly_model, metadata = load_model(model_path_anomaly)
        anomaly_scaler = metadata.get('scaler')
    else:
        logger.info(
            "Модель аномалій відсутня. Навчаємо нову...")
        initial_features_for_anomaly = [
            'amount', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
        anomaly_model, anomaly_scaler = train_anomaly_model(
            df_cleaned, initial_features_for_anomaly, contamination=None)
        if anomaly_model:
            save_model(anomaly_model, model_path_anomaly,
                       metadata={'scaler': anomaly_scaler})
        else:
            logger.error(
                "Не вдалося навчити модель аномалій через помилку в даних або алгоритмі.")
            return category_model, vectorizer, label_encoder, None, None

    return category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler


def process_single_transaction(amount, description, date_str, category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler):
    """Обробляє одну транзакцію та повертає результати."""
    # Очищення тексту
    cleaned_desc = clean_text(
        description, language='ukrainian', remove_stopwords=True)

    # Прогноз категорії
    predicted_category = predict_category(
        cleaned_desc, category_model, vectorizer, label_encoder, use_transformers=False)
    logger.info(
        f"Транзакція: '{description}' -> Категорія: {predicted_category}")

    # Створення даних для прогнозу аномалії (лише базові ознаки)
    try:
        anomaly_data = pd.Series({
            'amount': float(amount),
            'hour_of_day': pd.to_datetime(date_str).hour,
            'day_of_week': pd.to_datetime(date_str).dayofweek,
            'month': pd.to_datetime(date_str).month,
            'is_weekend': 1 if pd.to_datetime(date_str).dayofweek >= 5 else 0
        })
    except ValueError as e:
        logger.error(
            f"Помилка при створенні даних аномалії: {e}")
        return "Невідома", False

    # Прогноз аномалії
    is_anomaly = predict_anomaly(anomaly_data, anomaly_model, anomaly_scaler)
    logger.info(
        f"Транзакція: '{description}' -> Аномалія: {is_anomaly}")

    return predicted_category, is_anomaly


def main():
    logger.info(
        "Запуск консольного додатку для аналізу транзакцій...")

    # Завантаження та підготовка даних
    data_file = os.path.join('data', 'transactions.csv')
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    df = load_and_preprocess_data(data_file)
    if df is None:
        logger.error("Не вдалося завантажити дані. Переконайтеся, що data/transactions.csv існує і має коректний формат.")
        return

    important_keywords = ['атб', 'comfy', 'mono', 'супермаркет',
                          'кава', 'аптека', 'зарплата']
    df_cleaned = preprocess_transactions(
        df.copy(), language='ukrainian', keep_keywords=important_keywords)
    if df_cleaned is None:
        logger.error(
            "Не вдалося очистити описи. Перевірте дані та функцію preprocess_transactions.")
        return

    # Завантаження або навчання моделей
    category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler = load_or_train_models(
        df_cleaned)
    if not all([category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler]):
        logger.error(
            "Не вдалося ініціалізувати моделі. Перевірте дані або налаштування.")
        return

    # Цикл для введення транзакцій
    while True:
        logger.info(
            "\nВведіть нову транзакцію (або 'exit' для завершення):")
        amount = input("Сума (наприклад, -500.0): ")
        if amount.lower() == 'exit':
            break
        description = input(
            "Опис транзакції (наприклад, 'Покупка в АТБ'): ")
        date_str = input(
            "Дата (формат YYYY-MM-DD HH:MM, наприклад, 2025-07-05 14:00): ")

        try:
            amount = float(amount)
            datetime.strptime(date_str, '%Y-%m-%d %H:%M')
        except ValueError as e:
            logger.error(
                f"Невірний формат суми або дати. Очікується: сума (число), дата (YYYY-MM-DD HH:MM). Помилка: {e}")
            continue

        # Обробка транзакції
        predicted_category, is_anomaly = process_single_transaction(
            amount, description, date_str, category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler
        )

        # Опціонально: збереження транзакції у файл
        save_choice = input(
            "Зберегти транзакцію у файл? (y/n): ").lower()
        if save_choice == 'y':
            transaction_data = pd.DataFrame({
                'amount': [amount],
                'description': [description],
                'cleaned_description': [clean_text(description, language='ukrainian', remove_stopwords=True)],
                'date': [date_str],
                'category': [predicted_category],
                'is_anomaly': [is_anomaly]
            })
            transaction_data.to_csv('data/user_transactions.csv', mode='a',
                                    index=False, header=not os.path.exists('data/user_transactions.csv'))
            logger.info(
                "Транзакція збережена у data/user_transactions.csv.")
        elif save_choice not in ['y', 'n']:
            logger.warning(
                "Невірний вибір. Використано 'n' за замовчуванням.")

    logger.info("Додаток завершено.")


if __name__ == "__main__":
    main()
