import pandas as pd
import nltk
import re
import string
import os
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Налаштування логування
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Перевірка та завантаження NLTK ресурсів


def ensure_nltk_resources():
    """Завантажує необхідні NLTK ресурси, якщо вони ще не встановлені."""
    for resource in ['punkt_tab', 'stopwords']:
        try:
            nltk.data.find(
                f'tokenizers/{resource}' if resource == 'punkt_tab' else f'corpora/{resource}')
        except LookupError:
            logger.info(
                f"Завантаження NLTK ресурсу: {resource}")
            nltk.download(resource, quiet=True)


# Виклик функції для забезпечення ресурсів NLTK при імпорті модуля
ensure_nltk_resources()

# Список українських стоп-слів
UKRAINIAN_STOPWORDS = {
    'а', 'бо', 'буде', 'будеш', 'будете', 'будуть', 'були', 'було', 'бути', 'в', 'від', 'він', 'вона',
    'вони', 'все', 'всі', 'всю', 'гі', 'да', 'де', 'для', 'до', 'дуже', 'є', 'їм', 'її', 'за', 'знову', 'і', 'інші',
    'іноді', 'їх', 'його', 'й', 'каже', 'кажуть', 'коли', 'кому', 'кого', 'кожен', 'кожного', 'кожне', 'кожні', 'лиш',
    'майже', 'ми', 'мені', 'мене', 'може', 'можуть', 'можна', 'мой', 'моя', 'мої', 'на', 'навіть', 'нам', 'нами', 'нас',
    'наш', 'наша', 'наші', 'не', 'немає', 'ні', 'нічого', 'новий', 'нового', 'нові', 'ну', 'о', 'об', 'один', 'одного',
    'одна', 'одні', 'от', 'перед', 'після', 'по', 'поки', 'про', 'проти', 'раз', 'раніше', 'робити', 'сам', 'саме',
    'свій', 'свої', 'свою', 'собі', 'так', 'такий', 'також', 'там', 'те', 'тепер', 'ти', 'то', 'той', 'тобі',
    'тобою', 'тому', 'тут', 'у', 'хоча', 'хто', 'це', 'цей', 'чи', 'що', 'як', 'який', 'яке', 'які', 'якщо'
}


def load_and_preprocess_data(file_path):
    """
    Завантажує дані транзакцій з CSV, виконує базову передобробку та додає часові ознаки.
    """
    if not os.path.exists(file_path):
        logger.error(f"Файл {file_path} не знайдено.")
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        logger.info(
            f"Успішно завантажено {len(df)} транзакцій з {file_path}")
    except Exception as e:
        logger.error(
            f"Помилка при завантаженні CSV: {e}")
        return None

    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            logger.warning(
                f"Знайдено {df['date'].isna().sum()} некоректних дат. Рядки будуть видалені.")
    except Exception as e:
        logger.error(
            f"Помилка при обробці колонки 'date': {e}")
        return None

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    initial_len = len(df)
    df.dropna(subset=['date', 'amount', 'description',
              'category'], inplace=True)
    if len(df) < initial_len:
        logger.warning(
            f"Видалено {initial_len - len(df)} рядків через пропущені значення.")

    df['hour_of_day'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    logger.info(
        "Часові ознаки додано: hour_of_day, day_of_week, month, day_of_month, is_weekend")
    return df


def clean_text(text, language='ukrainian', remove_stopwords=True, keep_keywords=None):
    """
    Очищає текст транзакції: нижній регістр, видалення пунктуації, цифр, і, опціонально, стоп-слів.
    Дозволяє зберегти ключові слова через keep_keywords.
    """
    if not isinstance(text, str):
        logger.warning(
            f"Некоректний тип вхідного тексту: {type(text)}. Повертається порожній рядок.")
        return ""

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    try:
        word_tokens = word_tokenize(text, language='english')
    except LookupError as e:
        logger.error(
            f"Помилка при токенізації тексту: {e}. Повертаємо текст без токенізації.")
        return text
    except Exception as e:
        logger.error(
            f"Невідома помилка при токенізації тексту: {e}")
        return text

    if remove_stopwords:
        stop_words_set = UKRAINIAN_STOPWORDS
        if language == 'english':
            try:
                stop_words_set = set(stopwords.words('english'))
            except LookupError:
                logger.warning(
                    "Англійські стоп-слова NLTK не завантажені. Пропускаємо видалення стоп-слів.")
                stop_words_set = set()

        if keep_keywords:
            keep_keywords_set = set(k.lower() for k in keep_keywords)
            filtered_text = [w for w in word_tokens if (
                w not in stop_words_set or w in keep_keywords_set) and len(w) > 1]
        else:
            filtered_text = [
                w for w in word_tokens if w not in stop_words_set and len(w) > 1]
    else:
        filtered_text = [w for w in word_tokens if len(w) > 1]

    return " ".join(filtered_text).strip()


def preprocess_transactions(df, description_column='description', language='ukrainian', remove_stopwords=True, keep_keywords=None):
    """
    Застосовує функцію clean_text до колонки з описами в DataFrame.
    """
    if description_column not in df.columns:
        logger.error(
            f"Колонка '{description_column}' не знайдена в DataFrame.")
        return df

    try:
        logger.info(
            f"Застосовуємо очищення тексту до колонки '{description_column}'...")
        df['cleaned_description'] = df[description_column].apply(
            lambda x: clean_text(
                x, language=language, remove_stopwords=remove_stopwords, keep_keywords=keep_keywords)
        )
        logger.info(
            "Очищення тексту завершено. Створено нову колонку 'cleaned_description'.")
        return df
    except Exception as e:
        logger.error(
            f"Помилка при обробці колонки '{description_column}': {e}")
        return None
