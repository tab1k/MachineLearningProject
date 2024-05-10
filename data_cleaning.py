import pandas as pd

def clean_data(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Удаление дубликатов
    df.drop_duplicates(inplace=True)

    # Удаление строк с отсутствующими значениями
    df.dropna(inplace=True)

    # Дополнительные операции предварительной обработки могут быть добавлены здесь

    return df

# Путь к файлу с данными
file_path = "data/dataset.csv"

# Вызов функции для очистки данных
cleaned_df = clean_data(file_path)

# Сохранение очищенных данных в новый файл
cleaned_df.to_csv("data/cleaned_dataset.csv", index=False)
