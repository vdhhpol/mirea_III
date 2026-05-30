# Инструкция для проверки проект

## 1. Получение исходных данных

Проект использует файл `application_train.csv` из соревнования Kaggle  
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data).  
Размер файла ~300 МБ.

### Ручное скачивание (не требует регистрации API)

1. Перейдите по ссылке:  
   [https://www.kaggle.com/competitions/home-credit-default-risk/data]
2. Найдите в списке файл `application_train.csv`.
3. Нажмите на иконку скачивания рядом с ним. Файл будет загружен в папку `Загрузки`.
4. Поместите скачанный файл в папку `data/raw/` вашей локальной копии репозитория.  
   Если папки `data/raw/` нет, создайте её.

**Альтернатива** (если у вас есть доступ к Kaggle API):  
Выполните команду в терминале из корня репозитория:
```bash
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw/
```

## 2. Установка зависимостей

Убедитесь, что установлен Python версии 3.9 или выше. Затем установите необходимые библиотеки:

```bash
pip install -r requirements.txt
```

## 3. Запуск ноутбуков

Ноутбуки следует запускать в порядке их нумерации:

1. `01_EDA_and_baseline.ipynb` – разведочный анализ данных и baseline-модели. Результат: визуализации, выводы, но никакие данные не перезаписываются.

2. `02_Preprocessing_and_FeatureEngineering.ipynb` – предобработка и создание новых признаков. Результат:* в папке `data/processed/` появятся файлы `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`, а также `X_train_lin.csv`, `y_train_lin.csv` и т.д. Будет сохранён препроцессор `preprocessor.pkl` в папку `artifacts/`.

3. `03_Baseline_Models.ipynb` – обучение и сравнение базовых моделей (логистическая регрессия, Random Forest, LightGBM). Результат: лучшая модель (`best_model.pkl`) сохраняется в `artifacts/`.

4. `04_Neural_Network_MLP.ipynb` – построение и улучшение MLP. Результат: лучшая MLP-модель (`mlp_model_best.keras`) и оптимальный порог (`best_threshold.json`) сохраняются в `artifacts/`.

5. `05_Hyperparameter_Tuning_and_Calibration.ipynb` – настройка гиперпараметров LightGBM, калибровка вероятностей. Результат: финальная модель (`lightgbm_tuned.pkl`), калибратор (`calibrator.pkl`), порог (`threshold.json`) и лучшие гиперпараметры (`best_params.json`) сохраняются в `artifacts/`.