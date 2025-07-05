import os
import pandas as pd
import logging
import joblib
import numpy as np
from src.ai_analyzer import train_category_model, predict_category, train_anomaly_model, predict_anomaly, save_model, load_model, add_anomaly_features
from src.utils import load_and_preprocess_data, preprocess_transactions, clean_text

# Налаштування логування
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_ai_tests():
    logger.info("Запускаємо тести для src/ai_analyzer.py...")

    # Створення необхідних папок
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    data_file = os.path.join('data', 'transactions.csv')
    df = load_and_preprocess_data(data_file)
    if df is None:
        logger.error("Не вдалося завантажити дані.")
        return

    important_keywords = ['атб', 'comfy', 'mono', 'супермаркет',
                          'кава', 'аптека', 'зарплата']
    df_cleaned = preprocess_transactions(
        df.copy(), language='ukrainian', keep_keywords=important_keywords)
    if df_cleaned is None:
        logger.error("Не вдалося очистити описи.")
        return

    logger.info(
        f"Дані підготовлені. Розмір: {df_cleaned.shape}")

    # Тест 1: Категоризація
    logger.info(
        "\n--- Тест 1: Навчання та передбачення категоризації ---")
    category_model, vectorizer, classes, label_encoder = train_category_model(
        df_cleaned, model_type='xgboost', use_transformers=False
    )
    if category_model and vectorizer and label_encoder:
        logger.info(
            "Тест 1: Модель категоризації навчена.")

        test_texts = [
            "Покупка в Сільпо",
            "Отримання зарплати",
            "Кава в кав'ярні",
            "Сплата за інтернет"
        ]
        try:
            for text in test_texts:
                predicted_category = predict_category(
                    text, category_model, vectorizer, label_encoder, use_transformers=False)
                logger.info(
                    f"Текст: '{text}' -> Категорія: {predicted_category}")
                assert predicted_category != "Невідома", f"Категорія для '{text}' передбачена як 'Невідома'."
            logger.info(
                "Тест 1: Прогнози категоризації успішні.")
        except AssertionError as e:
            logger.error(
                f"Тест 1: Прогноз категоризації не пройшов: {e}")
            return

        model_path_cat = 'models/test_category_model.pkl'
        if save_model(category_model, model_path_cat, metadata={'vectorizer': vectorizer, 'label_encoder': label_encoder, 'use_transformers': False}):
            loaded_cat_model, loaded_cat_meta = load_model(model_path_cat)
            loaded_vectorizer = loaded_cat_meta.get('vectorizer')
            loaded_label_encoder = loaded_cat_meta.get('label_encoder')
            if loaded_cat_model and loaded_vectorizer and loaded_label_encoder:
                logger.info(
                    "Тест 1: Модель категоризації збережена та завантажена.")
                pred_loaded = predict_category("Оплата комунальних послуг",
                                               loaded_cat_model, loaded_vectorizer, loaded_label_encoder, use_transformers=False)
                logger.info(
                    f"Завантажена модель: 'Оплата комунальних послуг' -> Категорія: {pred_loaded}")
                assert pred_loaded != "Невідома", f"Завантажена модель повернула 'Невідома' для 'Оплата комунальних послуг'."
            else:
                logger.error(
                    "Тест 1: Помилка при завантаженні моделі категоризації.")
                return
        else:
            logger.error(
                "Тест 1: Помилка при збереженні моделі категоризації.")
            return
    else:
        logger.error(
            "Тест 1: Не вдалося навчити модель категоризації.")
        return

    # Тест 2: Аномалії
    logger.info(
        "\n--- Тест 2: Навчання та передбачення аномалій ---")
    initial_features_for_anomaly = [
        'amount', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
    # Змінено на None для автоматичного розрахунку
    anomaly_model, anomaly_scaler = train_anomaly_model(
        df_cleaned, initial_features_for_anomaly, contamination=None)
    if anomaly_model and anomaly_scaler:
        logger.info("Тест 2: Модель аномалій навчена.")

        df_anomaly_test = add_anomaly_features(df_cleaned.copy())
        all_anomaly_features = initial_features_for_anomaly + \
            ['mean_amount_by_category', 'amount_deviation',
                'transaction_count_per_day']

        # Обробка відсутніх даних
        if df_anomaly_test['is_anomaly'].eq(False).any():
            normal_sample = df_anomaly_test[df_anomaly_test['is_anomaly'] == False].sample(
                1, random_state=42)[all_anomaly_features].iloc[0]
        else:
            logger.warning(
                "Немає нормальних транзакцій для тестування. Використано середнє значення.")
            normal_sample = df_anomaly_test[all_anomaly_features].mean(
            ).to_frame().T

        if df_anomaly_test['is_anomaly'].eq(True).any():
            anomaly_sample = df_anomaly_test[df_anomaly_test['is_anomaly'] == True].sample(
                1, random_state=42)[all_anomaly_features].iloc[0]
        else:
            logger.warning(
                "Немає аномальних транзакцій для тестування. Використано випадкове значення.")
            anomaly_sample = df_anomaly_test[all_anomaly_features].sample(
                1, random_state=42).iloc[0]

        pred_normal = predict_anomaly(
            normal_sample, anomaly_model, anomaly_scaler)
        pred_anomaly = predict_anomaly(
            anomaly_sample, anomaly_model, anomaly_scaler)
        logger.info(
            f"Нормальна транзакція: аномальна? {pred_normal}")
        logger.info(
            f"Аномальна транзакція: аномальна? {pred_anomaly}")
        assert not pred_normal, "Нормальна транзакція помилково класифікована як аномальна."
        assert pred_anomaly, "Аномальна транзакція не розпізнана як аномальна."

        model_path_anomaly = 'models/test_anomaly_model.pkl'
        if save_model(anomaly_model, model_path_anomaly, metadata={'scaler': anomaly_scaler}):
            loaded_anom_model, loaded_anom_meta = load_model(
                model_path_anomaly)
            loaded_scaler = loaded_anom_meta.get('scaler')
            if loaded_anom_model and loaded_scaler:
                logger.info(
                    "Тест 2: Модель аномалій збережена та завантажена.")
                pred_loaded_anom = predict_anomaly(
                    normal_sample, loaded_anom_model, loaded_scaler)
                logger.info(
                    f"Завантажена модель: Нормальна транзакція аномальна? {pred_loaded_anom}")
                assert not pred_loaded_anom, "Завантажена модель помилково класифікувала нормальну транзакцію як аномальну."
            else:
                logger.error(
                    "Тест 2: Помилка при завантаженні моделі аномалій.")
                return
        else:
            logger.error(
                "Тест 2: Помилка при збереженні моделі аномалій.")
            return
    else:
        logger.error(
            "Тест 2: Не вдалося навчити модель аномалій.")
        return

    logger.info("\nТестування завершено.")


if __name__ == "__main__":
    run_ai_tests()
