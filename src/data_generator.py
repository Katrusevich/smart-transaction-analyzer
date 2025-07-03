import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


def generate_transactions(num_transactions=10000, anomaly_ratio=0.05, seed=None):
    """
    Генерує синтетичні дані банківських транзакцій.

    Параметри:
    num_transactions (int): Кількість транзакцій для генерації.
    anomaly_ratio (float): Базова ймовірність аномалії (0 до 1).
    seed (int, optional): Початкове значення для генератора випадкових чисел.

    Повертає:
    pandas.DataFrame: DataFrame з згенерованими транзакціями.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()
    # Секундна частота
    date_range = pd.date_range(start=start_date, end=end_date, freq='s')

    categories_data = {
        "Дохід": {
            "descriptions": ["Зарплата", "Переказ від родичів", "Стипендія", "Повернення боргу", "Фріланс оплата"],
            # Нормальний розподіл: середнє і стандартне відхилення
            "amount_mean_std": (12000, 4000),
            "anomaly_rate": 0.005,
            "is_income": True
        },
        "Їжа та Продукти": {
            "descriptions": ["АТБ", "Сільпо", "Фора", "Кафе 'Львівські Пляцки'", "Піцерія Domino's", "Суші Master"],
            "amount_mean_std": (500, 200),
            "anomaly_rate": 0.01,
            "is_income": False
        },
        "Транспорт": {
            "descriptions": ["Метрополітен", "Таксі Bolt", "АЗС WOG", "Укрзалізниця квиток", "Проїзд в автобусі"],
            "amount_mean_std": (300, 150),
            "anomaly_rate": 0.008,
            "is_income": False
        },
        "Оплата послуг": {
            "descriptions": ["Інтернет оплата", "Мобільний зв'язок Kyivstar", "Комунальні послуги", "Netflix"],
            "amount_mean_std": (600, 250),
            "anomaly_rate": 0.003,
            "is_income": False
        },
        "Розваги": {
            "descriptions": ["Кінотеатр Multiplex", "Клуб 'Атлас'", "Концерт в Оперному", "Steam покупки"],
            "amount_mean_std": (1000, 400),
            "anomaly_rate": 0.08,
            "is_income": False
        },
        "Покупки": {
            "descriptions": ["ЦУМ Київ", "Rozetka.ua замовлення", "Amazon покупка", "Одяг Zara", "Електроніка Comfy"],
            "amount_mean_std": (2000, 800),
            "anomaly_rate": 0.03,
            "is_income": False
        },
        "Здоров'я та Краса": {
            "descriptions": ["Аптека 'Доброго Дня'", "Клініка 'Борис'", "Стоматологія", "Фітнес-клуб SportLife"],
            "amount_mean_std": (1000, 500),
            "anomaly_rate": 0.01,
            "is_income": False
        },
        "Перекази": {
            "descriptions": ["Переказ на картку ПриватБанку", "Вихідний переказ", "Поповнення Sense SuperApp"],
            "amount_mean_std": (5000, 2000),
            "anomaly_rate": 0.06,
            "is_income": False
        },
        "Інше": {
            "descriptions": ["Поповнення мобільного", "Зняття готівки в банкоматі", "Невідомий платіж"],
            "amount_mean_std": (1000, 500),
            "anomaly_rate": 0.20,
            "is_income": False
        }
    }

    transactions = []

    for i in range(num_transactions):
        transaction_id = f"TRX-{i+1:06d}"
        date = random.choice(date_range).strftime("%Y-%m-%d %H:%M:%S")

        category_name = random.choices(
            list(categories_data.keys()),
            # Вагові коефіцієнти для категорій
            weights=[0.1, 0.25, 0.15, 0.15, 0.1, 0.15, 0.1, 0.05, 0.05],
            k=1
        )[0]
        category_info = categories_data[category_name]
        description = random.choice(category_info["descriptions"])

        # Генерація суми за нормальним розподілом
        amount = round(np.random.normal(
            category_info["amount_mean_std"][0], category_info["amount_mean_std"][1]), 2)
        # Обмеження мінімальної суми
        amount = max(amount, category_info["amount_mean_std"][0] / 10)
        if not category_info["is_income"]:
            amount = -amount

        # Генерація аномалій
        is_anomaly = random.random() < max(
            category_info["anomaly_rate"], anomaly_ratio)
        if is_anomaly:
            if random.random() < 0.5:  # Аномалія за сумою
                if category_info["is_income"]:
                    amount = round(np.random.uniform(
                        amount * 5, amount * 10), 2)
                    description = f"Аномально великий дохід: {description}"
                else:
                    amount = -round(np.random.uniform(abs(amount)
                                    * 5, abs(amount) * 10), 2)
                    description = f"Аномально велика витрата: {description}"
            else:  # Аномалія за описом
                suspicious_phrases = [
                    "Шахрайський переказ", "Несанкціонована транзакція", "Міжнародний платіж", "Криптобіржа"
                ]
                description = f"{random.choice(suspicious_phrases)}: {description}"
                category_name = "Інше"

        transactions.append({
            "transaction_id": transaction_id,
            "date": date,
            "amount": amount,
            "description": description,
            "category": category_name,
            "is_anomaly": is_anomaly
        })

    df = pd.DataFrame(transactions)
    return df


if __name__ == "__main__":
    # Налаштування
    num_transactions = 10000
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'transactions.csv')

    # Генерація транзакцій
    print(
        f"Генеруємо {num_transactions} синтетичних транзакцій...")
    transactions_df = generate_transactions(
        num_transactions=num_transactions, seed=42)

    # Збереження та статистика
    transactions_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[✔] Транзакції збережено у '{output_path}'")

    num_anomalies = transactions_df['is_anomaly'].sum()
    print(f"\nЗгенеровано {len(transactions_df)} транзакцій, з них {num_anomalies} аномалій "
          f"({num_anomalies / len(transactions_df):.2%}).")

    print("\nПерші 5 транзакцій:")
    print(transactions_df.head())
    print("\nРозподіл за категоріями:")
    print(transactions_df['category'].value_counts())
    print("\nРозподіл аномалій:")
    print(transactions_df['is_anomaly'].value_counts())
