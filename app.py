import streamlit as st
import pandas as pd
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile

# Імпортуємо функції з модулів
from src.ai_analyzer import train_category_model, predict_category, train_anomaly_model, predict_anomaly, save_model, load_model
try:
    from src.ai_analyzer import add_anomaly_features
except ImportError:
    add_anomaly_features = None
from src.utils import load_and_preprocess_data, preprocess_transactions, clean_text

# Налаштування логування
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Допоміжна функція для обробки завантаженого файлу


def adapt_uploaded_file(uploaded_file):
    """
    Конвертує завантажений файл у тимчасовий файл для обробки.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        logger.info(
            f"Створено тимчасовий файл: {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.error(f"Помилка обробки файлу: {e}")
        st.error(f"Помилка: {e}")
        return None

# Функція для завантаження або тренування моделей


@st.cache_resource
def load_or_train_models():
    """
    Завантажує або тренує моделі категоризації та аномалій.
    """
    model_cat_path = 'models/category_model.pkl'
    model_anomaly_path = 'models/anomaly_model.pkl'

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    category_model, vectorizer, label_encoder = None, None, None
    anomaly_model, anomaly_scaler = None, None

    if not (os.path.exists(model_cat_path) and os.path.exists(model_anomaly_path)):
        st.info(
            "Моделі відсутні. Розпочинаємо тренування...")
        data_path = os.path.join('data', 'transactions.csv')
        if not os.path.exists(data_path):
            st.error(
                "Файл data/transactions.csv не знайдено. Створіть його.")
            logger.error("Файл транзакцій відсутній.")
            return None, None, None, None, None

        df_raw = load_and_preprocess_data(data_path)
        if df_raw is None:
            st.error("Помилка завантаження даних.")
            logger.error(
                "Не вдалося завантажити дані.")
            return None, None, None, None, None

        keywords = ['атб', 'comfy', 'mono', 'супермаркет',
                    'кава', 'аптека', 'зарплата']
        df_processed = preprocess_transactions(
            df_raw.copy(), language='ukrainian', keep_keywords=keywords)
        if df_processed is None:
            st.error("Помилка обробки даних.")
            logger.error("Не вдалося обробити дані.")
            return None, None, None, None, None

        required_cols = ['cleaned_description', 'category', 'amount',
                         'hour_of_day', 'day_of_week', 'month', 'is_weekend']
        if not all(col in df_processed.columns for col in required_cols):
            st.error(
                "Дані не містять необхідних стовпців.")
            logger.error("Відсутні потрібні стовпці.")
            return None, None, None, None, None

        category_model, vectorizer, _, label_encoder = train_category_model(
            df_processed, model_type='xgboost', use_transformers=False)
        if category_model:
            save_model(category_model, model_cat_path, {
                       'vectorizer': vectorizer, 'label_encoder': label_encoder, 'use_transformers': False})
            st.success(
                "Модель категоризації збережена.")
            logger.info(
                "Модель категоризації навчена.")
        else:
            st.error(
                "Помилка тренування моделі категоризації.")
            logger.error(
                "Не вдалося навчити модель категоризації.")
            return None, None, None, None, None

        anomaly_features = ['amount', 'hour_of_day',
                            'day_of_week', 'month', 'is_weekend']
        anomaly_model, anomaly_scaler = train_anomaly_model(
            df_processed, anomaly_features, contamination=None)
        if anomaly_model:
            save_model(anomaly_model, model_anomaly_path,
                       {'scaler': anomaly_scaler})
            st.success("Модель аномалій збережена.")
            logger.info("Модель аномалій навчена.")
        else:
            st.error(
                "Помилка тренування моделі аномалій.")
            logger.error(
                "Не вдалося навчити модель аномалій.")
            return None, None, None, None, None

    else:
        st.info("Завантажуємо існуючі моделі...")
        category_data = load_model(model_cat_path)
        anomaly_data = load_model(model_anomaly_path)
        if category_data and anomaly_data:
            category_model, meta_cat = category_data
            anomaly_model, meta_anomaly = anomaly_data
            vectorizer = meta_cat.get('vectorizer')
            label_encoder = meta_cat.get('label_encoder')
            anomaly_scaler = meta_anomaly.get('scaler')
            logger.info("Моделі успішно завантажено.")
        else:
            st.error("Помилка завантаження моделей.")
            logger.error(
                "Не вдалося завантажити моделі.")
            return None, None, None, None, None

    return category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler

# Головна функція додатку


def main_dashboard():
    st.set_page_config(layout="wide", page_title="Smart Transaction Analyzer")
    st.title("💰 Аналізатор Транзакцій Monobank")
    st.markdown("Додаток для аналізу транзакцій, категоризації та виявлення аномалій.")

    # Завантаження файлу
    st.header("Завантажте файл транзакцій")
    uploaded_file = st.file_uploader(
        "Виберіть CSV-файл", type="csv")

    df_transactions = None
    if uploaded_file:
        with st.spinner("Обробка файлу..."):
            tmp_path = adapt_uploaded_file(uploaded_file)
            if tmp_path:
                df_processed = load_and_preprocess_data(tmp_path)
                os.unlink(tmp_path)
                if df_processed is not None:
                    keywords = ['атб', 'comfy', 'mono', 'супермаркет',
                                'кава', 'аптека', 'зарплата']
                    df_transactions = preprocess_transactions(
                        df_processed.copy(), language='ukrainian', keep_keywords=keywords)
                    st.success(
                        "Дані успішно завантажено!")
                    st.write("Перші 5 рядків:",
                             df_transactions.head())
                else:
                    st.error("Помилка обробки даних.")
            else:
                st.error(
                    "Помилка при завантаженні файлу.")
    else:
        st.info(
            "Будь ласка, завантажте CSV-файл для аналізу.")

    if df_transactions is not None:
        # Ініціалізація моделей
        with st.spinner("Підготовка моделей..."):
            category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler = load_or_train_models()
            if not all([category_model, vectorizer, label_encoder, anomaly_model, anomaly_scaler]):
                st.error(
                    "Помилка ініціалізації моделей.")
                return
            st.success("Моделі готові!")

        # Аналіз даних
        with st.spinner("Аналіз транзакцій..."):
            df_transactions['predicted_category'] = df_transactions['cleaned_description'].apply(
                lambda x: predict_category(x, category_model, vectorizer, label_encoder, use_transformers=False))

            if add_anomaly_features is not None:
                df_transactions = add_anomaly_features(df_transactions.copy())
            anomaly_features = ['amount', 'hour_of_day',
                                'day_of_week', 'month', 'is_weekend']
            if any(col not in df_transactions.columns for col in anomaly_features):
                st.error(
                    "Відсутні необхідні ознаки для аномалій.")
                return

            if df_transactions[anomaly_features].isnull().any().any():
                df_transactions[anomaly_features] = df_transactions[anomaly_features].fillna(
                    df_transactions[anomaly_features].mean())

            try:
                scaled_data = anomaly_scaler.transform(
                    df_transactions[anomaly_features])
                df_transactions['is_anomaly_predicted'] = anomaly_model.predict(
                    scaled_data) == -1
                st.success("Аналіз завершено!")
            except Exception as e:
                st.error(f"Помилка аналізу: {e}")
                logger.error(f"Помилка: {e}")
                return

        # Фільтри
        st.header("Результати Аналізу")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Початкова дата", df_transactions['date'].min(
            ).date() if not df_transactions['date'].empty else datetime.now().date())
        with col2:
            end_date = st.date_input("Кінцева дата", df_transactions['date'].max(
            ).date() if not df_transactions['date'].empty else datetime.now().date())

        filtered_df = df_transactions[(df_transactions['date'] >= pd.Timestamp(start_date)) &
                                      (df_transactions['date'] <= pd.Timestamp(end_date).replace(hour=23, minute=59, second=59))]

        # Виведення таблиці
        st.subheader("Транзакції з прогнозами")
        st.dataframe(filtered_df[['date', 'amount', 'description',
                     'category', 'predicted_category', 'is_anomaly_predicted']])

        # Графіки з меншими розмірами
        if not filtered_df.empty:
            st.subheader("Розподіл категорій")
            # Зменшено з (10, 6)
            fig_cat, ax_cat = plt.subplots(figsize=(5, 3))
            sns.barplot(x=filtered_df['predicted_category'].value_counts().index,
                        y=filtered_df['predicted_category'].value_counts().values, ax=ax_cat, palette='viridis')
            ax_cat.set_title("Розподіл категорій")
            ax_cat.set_xlabel("Категорія")
            ax_cat.set_ylabel("Кількість")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_cat)

            st.subheader("Аномалії")
            fig_anom, ax_anom = plt.subplots(
                figsize=(6, 3))  # Зменшено з (12, 6)
            sns.scatterplot(data=filtered_df, x='date', y='amount', hue='is_anomaly_predicted',
                            palette={True: 'red', False: 'blue'}, alpha=0.7, ax=ax_anom)
            ax_anom.set_title("Аномалії за сумою")
            ax_anom.set_xlabel("Дата")
            ax_anom.set_ylabel("Сума")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_anom)

            st.subheader("Динаміка витрат")
            # Зменшено з (12, 6)
            fig_dyn, ax_dyn = plt.subplots(figsize=(6, 3))
            filtered_df['date_month'] = filtered_df['date'].dt.to_period('M')
            monthly_data = filtered_df.groupby(
                'date_month')['amount'].sum().reset_index()
            monthly_data['date_month'] = monthly_data['date_month'].astype(str)
            sns.lineplot(data=monthly_data, x='date_month',
                         y='amount', marker='o', ax=ax_dyn)
            ax_dyn.set_title("Місячна динаміка")
            ax_dyn.set_xlabel("Місяць")
            ax_dyn.set_ylabel("Сума")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_dyn)

        # Експорт
        if st.button("Експортувати результати"):
            csv_data = filtered_df[['date', 'amount', 'description', 'category',
                                    'predicted_category', 'is_anomaly_predicted']].to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="Завантажити CSV", data=csv_data,
                               file_name="analyzed_transactions.csv", mime="text/csv")


if __name__ == "__main__":
    main_dashboard()
