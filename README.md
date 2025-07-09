# 💰 Smart Transaction Analyzer

> Інтелектуальний аналізатор банківських транзакцій з автоматичною категоризацією та виявленням аномалій

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 🎯 Огляд

Smart Transaction Analyzer - це потужний інструмент для аналізу банківських транзакцій, що використовує машинне навчання для автоматичної категоризації витрат/доходів та виявлення потенційних аномалій. Веб-дашборд на основі Streamlit забезпечує інтуїтивно зрозумілий інтерфейс для завантаження CSV-файлів та отримання миттєвого аналізу.

## ✨ Ключові особливості

- **🤖 Автоматична категоризація**: Модель XGBoost для прогнозування категорій транзакцій
- **🔍 Виявлення аномалій**: Алгоритм Isolation Forest для ідентифікації незвичайних операцій
- **📊 Інтерактивний дашборд**: Streamlit-інтерфейс з реальним часом
- **🧹 Розумна обробка даних**: Автоматичне очищення та створення ознак
- **📈 Багата візуалізація**: Графіки розподілу, динаміки та аномалій
- **🔄 Фільтрація та експорт**: Гнучкі інструменти для роботи з результатами
- **🏗️ Модульна архітектура**: Чистий та розширюваний код

## 🗂️ Структура проєкту

```
smart-transaction-analyzer/
├── 📁 data/
│   ├── transactions.csv             # Синтетичні дані для навчання
│   └── user_transactions.csv        # Користувацькі транзакції
├── 📁 models/
│   ├── category_model.pkl           # Модель категоризації
│   └── anomaly_model.pkl            # Модель виявлення аномалій
├── 📁 src/
│   ├── __init__.py
│   ├── ai_analyzer.py               # AI-аналіз
│   ├── data_generator.py            # Генератор синтетичних даних
│   └── utils.py                     # Допоміжні функції
├── 📄 app.py                        # Головний Streamlit додаток
├── 📄 main.py                       # Консольна програма
├── 📄 requirements.txt              # Залежності
└── 📄 test_ai_analyzer.py           # Тести
```

## 🚀 Швидкий старт

🌟 **Спробуйте додаток онлайн без встановлення!**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-transaction-analyzer-ndgzvuf45be2ji78kfhluk.streamlit.app/)

### Встановлення

1. **Клонуйте репозиторій**
   ```bash
   git clone https://github.com/ВашКористувач/smart-transaction-analyzer.git
   cd smart-transaction-analyzer
   ```

2. **Створіть віртуальне середовище**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Встановіть залежності**
   ```bash
   pip install -r requirements.txt
   ```

4. **Згенеруйте початкові дані**
   ```bash
   python data_generator.py
   ```

### Запуск

```bash
streamlit run app.py
```

Додаток буде доступний за адресою: `http://localhost:8501`

## 📝 Використання

### Веб-дашборд

1. **Завантажте CSV-файл** з транзакціями
2. **Дочекайтеся автоматичного аналізу** - система обробить дані та застосує моделі
3. **Переглядайте результати**:
   - Категоризовані транзакції
   - Виявлені аномалії
   - Графіки розподілу
   - Динаміка витрат/доходів
4. **Використовуйте фільтри** для деталізованого аналізу
5. **Експортуйте результати** у CSV

### Консольна програма

```bash
python main.py
```

Введіть деталі транзакції для миттєвого аналізу.

## 📊 Формат даних

Ваш CSV-файл повинен містити такі колонки:

| Колонка       | Опис                        | Приклад                  |
| ------------- | --------------------------- | ------------------------ |
| `date`        | Дата транзакції             | `2023-01-15`             |
| `description` | Опис транзакції             | `PAYMENT TO SUPERMARKET` |
| `amount`      | Сума (негативна для витрат) | `-150.50`                |
| `balance`     | Баланс після транзакції     | `2849.50`                |

## 🔧 Технології

- **Backend**: Python 3.8+, Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Frontend**: Streamlit
- **Візуалізація**: Matplotlib, Seaborn
- **NLP**: NLTK
- **Data Processing**: Pandas, NumPy

## 🧪 Тестування

```bash
python test_ai_analyzer.py
```

## 🤝 Внесок

Ваші внески вітаються! Будь ласка:

1. Форкніть репозиторій
2. Створіть feature branch (`git checkout -b feature/AmazingFeature`)
3. Зробіть commit (`git commit -m 'Add some AmazingFeature'`)
4. Push до branch (`git push origin feature/AmazingFeature`)
5. Відкрийте Pull Request

## 📜 Ліцензія

Розповсюджується під ліцензією MIT. Дивіться `LICENSE` для деталей.

---

<p align="center">
  Зроблено з ❤️ для спільноти
</p>