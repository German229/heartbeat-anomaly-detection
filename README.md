# Heartbeat Anomaly Detection

Классификация аномалий сердечных сокращений с помощью сверточной нейронной сети (Conv1D) на TensorFlow/Keras.

---

## 📋 Описание проекта

Автоматическое распознавание аномальных типов сердечных сокращений по данным ЭКГ.

- **Датасет:** [MIT-BIH Arrhythmia]([https://www.physionet.org/content/mitdb/1.0.0/](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (`files/mitbih_test.csv`)
- **Модель:** 1D Convolutional Neural Network (Conv1D)
- **Метрики:** Accuracy, Confusion Matrix, F1-score, Classification Report

---

## 📂 Структура проекта
```
heartbeat-anomaly-detection/
├── files/
│ └── mitbih_test.csv # CSV с данными ЭКГ
├── notebooks/
│ └── heartbeats_analysis.ipynb # Jupyter-ноутбук с анализом и моделью
├── requirements.txt # Необходимые библиотеки
└── .gitignore

```

---

## ⚙️ Требования и совместимость

- Python 3.9+
- TensorFlow: 2.10.1
- CUDA Toolkit: 11.8
- cuDNN: 8.9.4 (для CUDA 11.x)

> Для корректной работы на GPU убедитесь, что у вас установлены совместимые версии CUDA и cuDNN для TensorFlow 2.10.x.
>  
> ⚠️ См. официальную таблицу совместимости: [TensorFlow GPU Support](https://www.tensorflow.org/install/source#gpu)

---


## 🚀 Быстрый старт

1. **Склонируйте репозиторий и установите зависимости:**
```
git clone https://github.com/yourusername/heartbeat-anomaly-detection.git
```
```
cd heartbeat-anomaly-detection
```
```
pip install -r requirements.txt
```


3. **Запустите ноутбук:**
- Перейдите в папку `notebooks/`
- Откройте `heartbeats_analysis.ipynb` в Jupyter Notebook, JupyterLab или VSCode

---

## 🛠 Используемые библиотеки

- Python 3.9+
- pandas, numpy, scikit-learn
- tensorflow (>=2.10, рекомендуется >=2.13, поддержка CUDA 11.8)
- keras  
Полный список — в `requirements.txt`.

---

## 🔗 Pipeline ноутбука

1. Импорт библиотек
2. Фиксация random seed для воспроизводимости
3. Определение устройства (GPU/CPU)
4. Загрузка и анализ данных
5. Масштабирование признаков и one-hot кодирование
6. Деление на train/test
7. Расчет весов классов
8. Подбор гиперпараметров (Grid Search)
9. Обучение финальной модели
10. Оценка: accuracy, confusion matrix, F1-score, classification report

---

## ⚡️ Автоматический выбор устройства
```
has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
device_name = '/GPU:0' if has_gpu else '/CPU:0'
print(f"Используется устройство: {device_name}")
```


---

## 📊 Основные результаты

**Точность на тестовой выборке:**
- Loss: 0.44
- Accuracy: 0.943
- Macro F1-score: 0.7417

<details>
<summary>Показать confusion matrix и classification report</summary>

```
Confusion Matrix:
[[3533   41   13    6   31]
 [  53   56    2    0    0]
 [  51    0  220    6   13]
 [  17    0    2   13    0]
 [   9    0    4    0  309]]

Classification Report:
              precision    recall  f1-score   support
           0     0.9645    0.9749    0.9697      3624
           1     0.5773    0.5045    0.5385       111
           2     0.9129    0.7586    0.8286       290
           3     0.5200    0.4062    0.4561        32
           4     0.8754    0.9596    0.9156       322

    accuracy                         0.9434      4379
   macro avg     0.7700    0.7208    0.7417      4379
weighted avg     0.9415    0.9434    0.9417      4379
```

</details>

---

## 💡 Возможные улучшения

- K-fold cross-validation для более объективной оценки
- Аугментация данных
- Другие архитектуры (например, LSTM/GRU)

---

## 📬 Контакты

Вопросы по коду или установке — German229

---

**P.S.**  
Если проект был полезен — поставьте ⭐️ или сделайте fork!
