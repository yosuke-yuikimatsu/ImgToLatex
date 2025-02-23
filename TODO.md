## Имееем:
1. Рабочая модель - custom CNN + encoder(BiLSTM / AttentionTransformer) + decoder(Bahdanau Attention)
2. BLEU-Score метрика
3. DataLoader с эффективным паддингом(max length = 20)
## Надо сделать:
1. Дообучить модель с AttentionTransformer энкодером(Сейчас средний лосс порядка 0.3 но на инференсе модель только начала генерализироваться)
2. Прикрутить ResNet50 вместо собственной CNN и проверить качество работы


