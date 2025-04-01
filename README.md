# Img2Latex: нейроcеть по распознаванию математических выражений. 

**Img2Latex** — OCR-model encoder-decoder архитектуры для автоматического преобразования изображений математических выражений в LaTeX-код. Использует передовые методы глубокого обучения, обеспечивая высокую точность, в будущем может стать бюджетной альтернатиой коммерческим решениям. Интегрировано с Telegram-ботом, расширением для Chrome, модель размещена на сервере. 

**Разработчики**: Гейвандов Д.Д., Цинман С.П., Прудников М.М.\
**Руководитель**: Никитин А.А., к.ф.-м.н., доцент ВМК МГУ

## Обзор

LaTeX Vision решает задачу распознавания сложных математических выражений, включая низкокачественные изображения. По метрикам качества данный проект стремится к открытым аналогам (Pics2Latex, Tesseract-OCR), в будущем, после решения проблемы с сегментацией изображений, может быть использован как аналог MathPix

## Архитектура модели

Модель основана на энкодер-декодер архитектуре с механизмами внимания, оптимизированным для OCR LaTeX.

| Компонент | Версии | Описание |
| --- | --- | --- |
| **CNN** | 1\. Самописная (Conv+ReLU+MaxPooling)<br>2. ResNet-50<br>3. ConvNeXt-Large | Извлечение признаков. ConvNeXt-Large обеспечивает высокую точность. |
| **Энкодер** | 1\. BiLSTM<br>2. Transformer (2D Positional Encoding) | Обработка признаков с учетом структуры. Transformer улучшает контекст. |
| **Декодер** | 1\. LSTM + Bahdanau Attention<br>2. Transformer + Teaching Force | Генерация LaTeX-токенов. Transformer-декодер повышает точность. |
| **Загрузчик данных** | 1\. Img2Latex-100k<br>2. Бакетирование+зашумление<br>3. Синтетические данные<br>4. Токенизация | Улучшение устойчивости. Токенизация повышает эффективность обучения. |

## Технологический стек

- Python, PyTorch, TorchScript  
- Google Colab, Kaggle (обучение на GPU T4, L4)  
- aiogram (Telegram-бот)  
- FastAPI, Nginx (SSL, балансировка)  
- React + TypeScript (браузерное расширение)  
- KaTeX (рендеринг LaTeX)  
- YOLOv8 (сегментация страниц и детекция объектов)  

## Результаты

| Версия модели | Эпохи | BLEU Score | Примечания |
| --- | --- | --- | --- |
| LSTM + ResNet | 20 | 63% | Стабильная, но ограниченная точность. |
| Transformer + ConvNeXt (без токенизатора) | 90 | 78% | Низкая эффективность без токенизации. |
| Transformer + ConvNeXt (с токенизатором) | 40 | 89% | Высокая точность и устойчивость. |

## Интеграции

- **Telegram-бот**: На `aiogram`, преобразует изображения в LaTeX через Torch JIT.
- **FastAPI-сервис**: Высокопроизводительный API на Ubuntu с Nginx.
- **Chrome-расширение**: React + TypeScript, захват экрана, рендеринг через KaTeX.

## Планы на будущее

- Расширение датасета: матрицы, таблицы, графики функций.
- Поддержка Markdown, Typst.
- Улучшение сегментации с YOLO.
- Оптимизация для сложных выражений и больших данных.

## Установка

```bash
git clone https://github.com/your-username/latex-vision.git
cd latex-vision
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Использование

- **Бот**: `python bot/main.py`, найдите `@im2latex_bot` в Telegram.
- **Расширение**: Загрузите папку `extension` в Chrome (режим разработчика).

## Литература

1. Harvard NLP, «What You Get Is What You See: A Visual Markup Decompiler», 2017.
2. Ian Pointer, *Programming PyTorch for Deep Learning*, 2019.
3. Stanford CS231n, «Adaptation of OCR Models for LaTeX Vision», 2024.

