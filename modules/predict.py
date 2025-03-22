import os
import pandas as pd
import dill
import json
import glob


def load_latest_model(models_dir):
    """Находит и загружает последнюю модель из указанной директории."""
    # Ищем все файлы моделей по шаблону
    model_files = glob.glob(os.path.join(models_dir, 'cars_pipe_*.pkl'))

    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    # Выбираем файл с самой поздней датой в имени
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")

    with open(latest_model, 'rb') as f:
        return dill.load(f)


def load_test_data(test_data_path):
    """Загружает тестовые данные из JSON-файлов в папке."""
    test_files = [os.path.join(test_data_path, f)
                  for f in os.listdir(test_data_path)
                  if f.endswith('.json')]

    data_frames = []
    for file in test_files:
        with open(file, 'r') as f:
            data = json.load(f)
            data_frames.append(pd.DataFrame([data]))

    return pd.concat(data_frames, ignore_index=True)


def make_predictions(model, data):
    """Делает предсказания на основе загруженной модели."""
    return model.predict(data)


def save_predictions(predictions, output_path):
    """Сохраняет предсказания в файл."""
    predictions_df = pd.DataFrame(predictions, columns=['predictions'])
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def predict():
    """Основная функция для выполнения предсказаний."""
    path = os.environ.get('PROJECT_PATH', '.')

    # Пути с автоматическим выбором модели
    models_dir = os.path.join(path, 'data/models')
    test_data_path = os.path.join(path, 'data/test')
    output_dir = os.path.join(path, 'data/predictions')

    # Создаем директорию для предсказаний, если нужно
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions.csv')

    # Загрузка модели и данных
    model = load_latest_model(models_dir)
    test_data = load_test_data(test_data_path)

    # Предсказание и сохранение
    predictions = make_predictions(model, test_data)
    save_predictions(predictions, output_path)


if __name__ == "__main__":
    predict()