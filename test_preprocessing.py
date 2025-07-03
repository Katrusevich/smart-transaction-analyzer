import os
import string
import pandas as pd
from src.utils import load_and_preprocess_data, clean_text, preprocess_transactions
import logging
from io import StringIO

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_preprocessing_tests():
    """
    Запускає набір тестів для перевірки функцій load_and_preprocess_data,
    clean_text та preprocess_transactions з src/utils.py.
    """
    logger.info("Запускаємо тести для src/utils.py...")

    # --- Тест 1: Завантаження та попередня обробка даних ---
    logger.info("\n--- Тест 1: Завантаження та попередня обробка даних ---")
    data_file_path = os.path.join('data', 'transactions.csv')

    # Перевірка наявності файлу
    if not os.path.exists(data_file_path):
        logger.error(f"Файл даних не знайдено: {data_file_path}. Будь ласка, спочатку запустіть data_generator.py.")
        return

    df_processed = load_and_preprocess_data(data_file_path)

    if df_processed is not None:
        logger.info("Тест 1: Успішно завантажено та оброблено дані.")
        logger.info(f"Розмір DataFrame: {df_processed.shape[0]} рядків, {df_processed.shape[1]} колонок.")
        logger.info("Перші 5 рядків обробленого DataFrame:")
        logger.info("\n" + df_processed.head().to_string())

        logger.info("\nІнформація про типи даних колонок (df.info()):")
        buffer = StringIO()
        df_processed.info(buf=buffer)
        logger.info(buffer.getvalue())

        # Перевірка наявності нових колонок
        expected_new_columns = ['hour_of_day', 'day_of_week', 'month', 'day_of_month', 'is_weekend']
        for col in expected_new_columns:
            if col in df_processed.columns:
                logger.info(f"Колонка '{col}' присутня.")
            else:
                logger.error(f"Колонка '{col}' ВІДСУТНЯ!")

        # Перевірка типу 'date'
        if pd.api.types.is_datetime64_any_dtype(df_processed['date']):
            logger.info("'date' колонка має коректний тип datetime.")
        else:
            logger.error("'date' колонка має НЕКОРЕКТНИЙ тип.")

        # Перевірка типу 'amount'
        if pd.api.types.is_numeric_dtype(df_processed['amount']):
            logger.info("'amount' колонка має коректний числовий тип.")
        else:
            logger.error("'amount' колонка має НЕКОРЕКТНИЙ числовий тип.")
    else:
        logger.error("Тест 1: Не вдалося завантажити або обробити дані.")
        return

    # --- Тест 2: Очищення тексту ---
    logger.info("\n--- Тест 2: Очищення тексту ---")
    test_strings = [
        "Купив 3 кави по $5.50 у 'CoffeE Shop!' на 123 вулиці. (замовлення №12345)",
        "Зарплата за червень (2024 рік) на картку Mono.",
        "Оплата за проїзд в маршрутці #30",
        "Надзвичайно велике надходження: Фріланс оплата. 10000.00 грн",
        "This is an English test string with Punctuation and numbers 123.",
        "",  # Порожній рядок
        None  # Неправильний тип
    ]

    for i, text in enumerate(test_strings):
        cleaned_uk = clean_text(text, language='ukrainian', remove_stopwords=True)
        cleaned_en = clean_text(text, language='english', remove_stopwords=True)
        cleaned_no_stopwords = clean_text(text, remove_stopwords=False)

        logger.info(f"Тест 2.{i+1}: Оригінал: '{text}'")
        logger.info(f"    Очищено (укр, стоп-слова): '{cleaned_uk}'")
        logger.info(f"    Очищено (англ, стоп-слова): '{cleaned_en}'")
        logger.info(f"    Очищено (без стоп-слів): '{cleaned_no_stopwords}'")

        # Додаткові перевірки
        if isinstance(text, str) and text.strip():
            if any(char in string.punctuation for char in cleaned_uk):
                logger.warning(f"    Попередження: Пунктуація залишилася в очищеному тексті (укр).")
            if any(char.isdigit() for char in cleaned_uk):
                logger.warning(f"    Попередження: Цифри залишилися в очищеному тексті (укр).")
            if ' ' in text and ' ' not in cleaned_uk and len(text.split()) > 1 and cleaned_uk:
                logger.warning(f"    Попередження: Пробіли могли зникнути після очищення.")

    # --- Тест 3: Застосування очищення до DataFrame ---
    logger.info("\n--- Тест 3: Застосування очищення до DataFrame ---")
    if 'description' in df_processed.columns:
        df_final = preprocess_transactions(df_processed.copy(), description_column='description', language='ukrainian')
        if df_final is not None:
            logger.info("Тест 3: Успішно застосовано очищення до колонки 'description'.")
            logger.info("Перші 5 рядків з 'cleaned_description':")
            logger.info("\n" + df_final[['description', 'cleaned_description']].head().to_string())

            # Перевірка змін в описах
            diff_count = (df_final['description'] != df_final['cleaned_description']).sum()
            logger.info(f"Кількість описів, які змінилися після очищення: {diff_count} з {len(df_final)}.")
            if diff_count > 0:
                logger.info("Тест 3: Очищення тексту працює, змінюючи оригінальні описи.")
            else:
                logger.warning("Тест 3: Очищення тексту не призвело до змін в описах. Перевірте логіку.")
        else:
            logger.error("Тест 3: Не вдалося обробити колонки через помилку в preprocess_transactions.")
    else:
        logger.error("Тест 3: Колонка 'description' не знайдена в DataFrame. Пропущено тест.")

    logger.info("\nТестування src/utils.py завершено.")

if __name__ == "__main__":
    run_preprocessing_tests()