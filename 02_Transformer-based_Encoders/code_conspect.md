
## 1. Токенизаторы из Hugging Face

### Основные принципы токенизации

```python
from transformers import AutoTokenizer, BertTokenizer
import torch

# Загрузка различных токенизаторов
# AutoTokenizer автоматически подбирает нужный токенизатор для модели
tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_distilbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer_rubert = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

# Альтернативный способ - прямое указание класса токенизатора
tokenizer_direct = BertTokenizer.from_pretrained('bert-base-uncased')

print("Токенизаторы загружены успешно!")
```

### Основные методы токенизаторов

```python
# Пример текста для работы
text = "Hello, how are you doing today? I'm learning NLP!"
texts = [
    "First sentence for batch processing.",
    "Second sentence with different length.",
    "Third sentence."
]

# 1. Базовая токенизация - разбиение на токены
tokens = tokenizer_bert.tokenize(text)
print(f"Токены: {tokens}")
print(f"Количество токенов: {len(tokens)}")

# 2. Кодирование в ID (числовое представление)
token_ids = tokenizer_bert.encode(text)
print(f"ID токенов: {token_ids}")

# 3. Полное кодирование с дополнительной информацией
# return_tensors='pt' возвращает PyTorch тензоры
# padding=True выравнивает длины последовательностей
# truncation=True обрезает слишком длинные тексты
# max_length устанавливает максимальную длину
encoded = tokenizer_bert(
    text,
    return_tensors='pt',          # Возвращать PyTorch тензоры
    padding=True,                 # Добавлять паддинг
    truncation=True,              # Обрезать длинные тексты
    max_length=128,               # Максимальная длина
    return_attention_mask=True,    # Возвращать маску внимания
    return_token_type_ids=True     # Возвращать типы токенов (для BERT)
)

print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Attention mask: {encoded['attention_mask']}")
print(f"Token type IDs: {encoded['token_type_ids']}")

# 4. Пакетная обработка текстов
batch_encoded = tokenizer_bert(
    texts,
    return_tensors='pt',
    padding=True,                 # Важно для батчей разной длины
    truncation=True,
    max_length=64
)

print(f"Batch shape: {batch_encoded['input_ids'].shape}")
print(f"Batch input IDs:\n{batch_encoded['input_ids']}")
```

### Полезные методы токенизаторов

```python
# 5. Декодирование - перевод ID обратно в текст
decoded_text = tokenizer_bert.decode(token_ids)
print(f"Декодированный текст: {decoded_text}")

# Декодирование без служебных токенов [CLS], [SEP]
decoded_clean = tokenizer_bert.decode(token_ids, skip_special_tokens=True)
print(f"Чистый декодированный текст: {decoded_clean}")

# 6. Получение информации о словаре
print(f"Размер словаря: {tokenizer_bert.vocab_size}")
print(f"Максимальная длина: {tokenizer_bert.model_max_length}")

# 7. Специальные токены
print(f"CLS токен: {tokenizer_bert.cls_token} (ID: {tokenizer_bert.cls_token_id})")
print(f"SEP токен: {tokenizer_bert.sep_token} (ID: {tokenizer_bert.sep_token_id})")
print(f"PAD токен: {tokenizer_bert.pad_token} (ID: {tokenizer_bert.pad_token_id})")
print(f"UNK токен: {tokenizer_bert.unk_token} (ID: {tokenizer_bert.unk_token_id})")

# 8. Работа с парами предложений (для NSP задач)
sentence_a = "The weather is nice today."
sentence_b = "I want to go for a walk."

pair_encoded = tokenizer_bert(
    sentence_a,
    sentence_b,                   # Вторе предложение
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=128
)

print(f"Token types для пары: {pair_encoded['token_type_ids']}")
# 0 - токены первого предложения, 1 - токены второго предложения

# 9. Получение позиций специальных токенов
tokens_with_ids = tokenizer_bert.encode_plus(
    text,
    return_tensors='pt',
    return_special_tokens_mask=True  # Маска специальных токенов
)
print(f"Маска специальных токенов: {tokens_with_ids['special_tokens_mask']}")
```

## 2. Работа с BERT моделями для MLM и NSP

### Загрузка моделей

```python
from transformers import (
    AutoModel, 
    AutoModelForMaskedLM, 
    AutoModelForNextSentencePrediction,
    BertForMaskedLM,
    BertForNextSentencePrediction
)
import torch
import torch.nn.functional as F
import numpy as np

# Модели для разных задач
# Базовая BERT модель (без головы для конкретной задачи)
bert_base = AutoModel.from_pretrained('bert-base-uncased')

# BERT для Masked Language Modeling
bert_mlm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

# BERT для Next Sentence Prediction
bert_nsp = AutoModelForNextSentencePrediction.from_pretrained('bert-base-uncased')

# Переводим модели в режим оценки (отключаем dropout и batch norm обновления)
bert_base.eval()
bert_mlm.eval()
bert_nsp.eval()

print("Модели загружены успешно!")
```

### Masked Language Modeling (MLM)

```python
def predict_masked_tokens(text_with_mask, model, tokenizer, top_k=5):
    """
    Предсказывает замаскированные токены в тексте
    
    Args:
        text_with_mask: текст с [MASK] токенами
        model: модель для MLM
        tokenizer: токенизатор
        top_k: количество лучших предсказаний
    """
    # Токенизируем текст
    inputs = tokenizer(text_with_mask, return_tensors='pt')
    
    # Находим позицию маскированного токена
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
    
    # Отключаем вычисление градиентов для инференса
    with torch.no_grad():
        # Получаем предсказания модели
        outputs = model(**inputs)
        predictions = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    results = []
    for mask_pos in mask_positions:
        # Получаем логиты для маскированной позиции
        mask_logits = predictions[0, mask_pos, :]
        
        # Применяем softmax для получения вероятностей
        mask_probs = F.softmax(mask_logits, dim=0)
        
        # Получаем top-k предсказаний
        top_k_indices = torch.topk(mask_probs, top_k).indices
        top_k_probs = torch.topk(mask_probs, top_k).values
        
        # Декодируем токены
        predicted_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
        
        results.append({
            'position': mask_pos.item(),
            'predictions': list(zip(predicted_tokens, top_k_probs.tolist()))
        })
    
    return results

# Примеры использования MLM
examples_mlm = [
    "The capital of France is [MASK].",
    "I love to eat [MASK] for breakfast.",
    "The [MASK] is shining brightly today.",
    "She works as a [MASK] in a hospital."
]

print("=== MASKED LANGUAGE MODELING ===")
for text in examples_mlm:
    print(f"\nТекст: {text}")
    results = predict_masked_tokens(text, bert_mlm, tokenizer_bert)
    
    for result in results:
        print(f"Позиция маски: {result['position']}")
        for token, prob in result['predictions']:
            print(f"  {token.strip()}: {prob:.4f}")

# Пример с несколькими масками
multi_mask_text = "The [MASK] sat on the [MASK]."
print(f"\n\nТекст с несколькими масками: {multi_mask_text}")
multi_results = predict_masked_tokens(multi_mask_text, bert_mlm, tokenizer_bert)
for i, result in enumerate(multi_results):
    print(f"Маска {i+1} (позиция {result['position']}):")
    for token, prob in result['predictions']:
        print(f"  {token.strip()}: {prob:.4f}")
```

### Next Sentence Prediction (NSP)

```python
def predict_next_sentence(sentence_a, sentence_b, model, tokenizer):
    """
    Предсказывает, является ли sentence_b логическим продолжением sentence_a
    
    Args:
        sentence_a: первое предложение
        sentence_b: второе предложение
        model: модель для NSP
        tokenizer: токенизатор
    
    Returns:
        dict с результатами предсказания
    """
    # Кодируем пару предложений
    inputs = tokenizer(
        sentence_a,
        sentence_b,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Получаем предсказания
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, 2] - логиты для двух классов
    
    # Применяем softmax для получения вероятностей
    probs = F.softmax(logits, dim=1)
    
    # 0 - предложения связаны (IsNext), 1 - не связаны (NotNext)
    is_next_prob = probs[0][0].item()
    not_next_prob = probs[0][1].item()
    
    prediction = "IsNext" if is_next_prob > not_next_prob else "NotNext"
    confidence = max(is_next_prob, not_next_prob)
    
    return {
        'sentence_a': sentence_a,
        'sentence_b': sentence_b,
        'prediction': prediction,
        'confidence': confidence,
        'is_next_prob': is_next_prob,
        'not_next_prob': not_next_prob
    }

# Примеры для NSP
print("\n=== NEXT SENTENCE PREDICTION ===")

# Положительные примеры (логически связанные предложения)
positive_pairs = [
    ("I went to the store.", "I bought some milk and bread."),
    ("The weather is very cold today.", "I need to wear a warm coat."),
    ("She studied hard for the exam.", "She got an excellent grade."),
]

# Отрицательные примеры (логически не связанные предложения)
negative_pairs = [
    ("I went to the store.", "The cat is sleeping on the sofa."),
    ("The weather is very cold today.", "My favorite color is blue."),
    ("She studied hard for the exam.", "Pizza is a popular food in Italy."),
]

print("\nПоложительные примеры (связанные предложения):")
for sent_a, sent_b in positive_pairs:
    result = predict_next_sentence(sent_a, sent_b, bert_nsp, tokenizer_bert)
    print(f"A: {result['sentence_a']}")
    print(f"B: {result['sentence_b']}")
    print(f"Предсказание: {result['prediction']} (уверенность: {result['confidence']:.4f})")
    print(f"IsNext: {result['is_next_prob']:.4f}, NotNext: {result['not_next_prob']:.4f}\n")

print("Отрицательные примеры (несвязанные предложения):")
for sent_a, sent_b in negative_pairs:
    result = predict_next_sentence(sent_a, sent_b, bert_nsp, tokenizer_bert)
    print(f"A: {result['sentence_a']}")
    print(f"B: {result['sentence_b']}")
    print(f"Предсказание: {result['prediction']} (уверенность: {result['confidence']:.4f})")
    print(f"IsNext: {result['is_next_prob']:.4f}, NotNext: {result['not_next_prob']:.4f}\n")
```

## 3. Использование Pipeline

```python
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')  # Подавляем предупреждения для чистоты вывода

# Создание различных пайплайнов
# Pipeline автоматически загружает подходящую модель и токенизатор

# 1. Анализ тональности
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True  # Возвращать оценки для всех классов
)

# 2. Заполнение масок (аналог MLM, но проще в использовании)
fill_mask_pipeline = pipeline(
    "fill-mask",
    model="bert-base-uncased",
    top_k=3  # Возвращать 3 лучших предсказания
)

# 3. Классификация текста (zero-shot)
zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# 4. Распознавание именованных сущностей (NER)
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"  # Объединяет токены одной сущности
)

# 5. Ответы на вопросы
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# 6. Суммаризация текста
summarization_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

print("Все пайплайны созданы успешно!\n")

# Примеры использования пайплайнов
def demonstrate_pipelines():
    """Демонстрация работы различных пайплайнов"""
    
    # === АНАЛИЗ ТОНАЛЬНОСТИ ===
    print("=== АНАЛИЗ ТОНАЛЬНОСТИ ===")
    sentiment_texts = [
        "I love this movie! It's absolutely fantastic!",
        "This product is okay, nothing special.",
        "I hate waiting in long lines. It's so annoying!",
        "The weather today is perfect for a picnic."
    ]
    
    for text in sentiment_texts:
        result = sentiment_pipeline(text)
        print(f"Текст: {text}")
        for score in result[0]:  # result[0] потому что return_all_scores=True
            print(f"  {score['label']}: {score['score']:.4f}")
        print()
    
    # === ЗАПОЛНЕНИЕ МАСОК ===
    print("=== ЗАПОЛНЕНИЕ МАСОК ===")
    mask_texts = [
        "The capital of France is [MASK].",
        "I work as a [MASK] in a hospital.",
        "My favorite [MASK] is chocolate ice cream.",
    ]
    
    for text in mask_texts:
        results = fill_mask_pipeline(text)
        print(f"Текст: {text}")
        for result in results:
            print(f"  {result['token_str']}: {result['score']:.4f}")
        print()
    
    # === ZERO-SHOT КЛАССИФИКАЦИЯ ===
    print("=== ZERO-SHOT КЛАССИФИКАЦИЯ ===")
    text_to_classify = "I just bought a new smartphone and it has an amazing camera!"
    candidate_labels = ["technology", "food", "sports", "politics", "entertainment"]
    
    result = zero_shot_pipeline(text_to_classify, candidate_labels)
    print(f"Текст: {text_to_classify}")
    print("Классификация:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.4f}")
    print()
    
    # === РАСПОЗНАВАНИЕ ИМЕНОВАННЫХ СУЩНОСТЕЙ ===
    print("=== РАСПОЗНАВАНИЕ ИМЕНОВАННЫХ СУЩНОСТЕЙ ===")
    ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
    
    entities = ner_pipeline(ner_text)
    print(f"Текст: {ner_text}")
    print("Найденные сущности:")
    for entity in entities:
        print(f"  {entity['word']}: {entity['entity_group']} (уверенность: {entity['score']:.4f})")
    print()
    
    # === ОТВЕТЫ НА ВОПРОСЫ ===
    print("=== ОТВЕТЫ НА ВОПРОСЫ ===")
    context = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest 
    in the Amazon biome that covers most of the Amazon basin of South America. The basin is 
    shared by nine countries: Brazil, Peru, Colombia, Venezuela, Ecuador, Bolivia, Guyana, 
    Suriname, and French Guiana. The rainforest contains over 390 billion individual trees 
    divided into 16,000 species.
    """
    
    questions = [
        "How many countries share the Amazon basin?",
        "How many tree species are in the rainforest?",
        "What is another name for the Amazon rainforest?"
    ]
    
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Вопрос: {question}")
        print(f"Ответ: {result['answer']} (уверенность: {result['score']:.4f})")
        print()
    
    # === СУММАРИЗАЦИЯ ===
    print("=== СУММАРИЗАЦИЯ ТЕКСТА ===")
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the 
    natural intelligence displayed by humans and animals. Leading AI textbooks define the field 
    as the study of "intelligent agents": any device that perceives its environment and takes 
    actions that maximize its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic "cognitive" functions 
    that humans associate with the human mind, such as "learning" and "problem solving". As machines 
    become increasingly capable, tasks considered to require "intelligence" are often removed from 
    the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says 
    "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently 
    excluded from things considered to be AI, having become a routine technology.
    """
    
    summary = summarization_pipeline(
        long_text, 
        max_length=50,  # Максимальная длина резюме
        min_length=20,  # Минимальная длина резюме
        do_sample=False  # Детерминированная генерация
    )
    
    print("Оригинальный текст:")
    print(long_text[:200] + "...")
    print(f"\nКраткое содержание:")
    print(summary[0]['summary_text'])

# Запускаем демонстрацию
demonstrate_pipelines()

# Полезные советы по работе с пайплайнами
print("\n=== ПОЛЕЗНЫЕ СОВЕТЫ ===")

# Пакетная обработка с пайплайнами
batch_texts = [
    "This movie is great!",
    "I don't like this product.",
    "The service was okay."
]

# Обработка батча более эффективна, чем по одному тексту
batch_results = sentiment_pipeline(batch_texts)
print("Пакетная обработка:")
for text, result in zip(batch_texts, batch_results):
    best_result = max(result, key=lambda x: x['score'])
    print(f"{text} -> {best_result['label']} ({best_result['score']:.4f})")

# Настройка устройства (GPU/CPU)
# device=0 для первой GPU, device=-1 для CPU
sentiment_pipeline_gpu = pipeline(
    "sentiment-analysis",
    device=-1  # Используем CPU (-1), для GPU указать номер устройства (0, 1, ...)
)
```

## 4. Дообучение BERT для классификации

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Создаем датасет для демонстрации (в реальности загружаете свои данные)
def create_sample_dataset():
    """Создает пример датасета для классификации тональности"""
    data = {
        'text': [
            "I love this movie! It's absolutely fantastic and amazing!",
            "This film is terrible, I hated every minute of it.",
            "The movie was okay, nothing special but not bad either.",
            "Outstanding performance by all actors, highly recommended!",
            "Boring and predictable plot, waste of time.",
            "Great cinematography and excellent storyline.",
            "Not worth watching, very disappointing.",
            "Amazing special effects and compelling characters!",
            "The movie is decent, has its good and bad moments.",
            "Incredible movie, one of the best I've ever seen!",
            "Poor acting and weak script throughout.",
            "Good movie overall, enjoyed most parts of it.",
            "Absolutely awful, couldn't finish watching it.",
            "Excellent direction and beautiful soundtrack.",
            "Average movie, nothing memorable about it.",
            "Fantastic plot twists and great ending!",
            "Terrible dialogue and boring scenes.",
            "Really good film, would watch it again.",
            "Disappointing compared to the hype around it.",
            "Brilliant performances and stunning visuals!"
        ] * 10,  # Увеличиваем датасет для лучшего обучения
        'label': [1, 0, 2, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1] * 10
        # 0: негативный, 1: позитивный, 2: нейтральный
    }
    return pd.DataFrame(data)

# Создаем кастомный класс датасета для PyTorch
class TextClassificationDataset(Dataset):
    """Датасет для классификации текста"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: список текстов
            labels: список меток
            tokenizer: токенизатор для обработки текста
            max_length: максимальная длина последовательности
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Возвращает один элемент датасета"""
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Токенизируем текст
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Функция для вычисления метрик во время обучения"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Вычисляем точность
    accuracy = accuracy_score(labels, predictions)
    
    # Вычисляем precision, recall, f1 для каждого класса
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Основная функция для дообучения модели
def fine_tune_bert_classifier():
    """Дообучает BERT для задачи классификации текста"""
    
    print("Создание датасета...")
    # Создаем и подготавливаем данные
    df = create_sample_dataset()
    
    # Разделяем на train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']  # Сохраняем пропорции классов
    )
    
    print(f"Размер обучающей выборки: {len(train_texts)}")
    print(f"Размер валидационной выборки: {len(val_texts)}")
    print(f"Распределение классов: {df['label'].value_counts().to_dict()}")
    
    # Загружаем токенизатор и модель
    print("\nЗагрузка модели и токенизатора...")
    model_name = "distilbert-base-uncased"  # Легкая версия BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Модель для классификации последовательностей с 3 классами
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # 3 класса: негативный, позитивный, нейтральный
        id2label={0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 2}
    )
    
    # Создаем датасеты
    print("Создание датасетов...")
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer
    )
    
    # Создаем коллатор данных для эффективной пакетной обработки
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Настройки обучения
    training_args = TrainingArguments(
        output_dir='./results',              # Директория для сохранения результатов
        num_train_epochs=3,                  # Количество эпох
        per_device_train_batch_size=16,      # Размер батча для обучения
        per_device_eval_batch_size=16,       # Размер батча для валидации
        warmup_steps=100,                    # Количество шагов разогрева
        weight_decay=0.01,                   # L2 регуляризация
        logging_dir='./logs',                # Директория для логов
        logging_steps=10,                    # Частота логирования
        evaluation_strategy="steps",         # Стратегия оценки
        eval_steps=50,                       # Частота оценки
        save_strategy="steps",               # Стратегия сохранения
        save_steps=100,                      # Частота сохранения
        load_best_model_at_end=True,         # Загружать лучшую модель в конце
        metric_for_best_model="f1",          # Метрика для выбора лучшей модели
        greater_is_better=True,              # Больше = лучше для F1
        save_total_limit=2,                  # Сохранять только 2 лучшие модели
        seed=42,                             # Seed для воспроизводимости
        fp16=False,                          # Использовать ли 16-bit precision (для GPU)
    )
    
    # Создаем тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Начинаем обучение
    print("\nНачало обучения...")
    trainer.train()
    
    # Оценка на валидационной выборке
    print("\nОценка модели...")
    eval_results = trainer.evaluate()
    print(f"Результаты оценки: {eval_results}")
    
    # Сохраняем модель
    print("\nСохранение модели...")
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    
    return model, tokenizer, eval_results

def test_fine_tuned_model(model, tokenizer):
    """Тестирует дообученную модель на новых примерах"""
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Тестовые примеры
    test_texts = [
        "This movie is absolutely incredible! Best film ever!",
        "Worst movie I've ever seen, complete waste of time.",
        "The movie was okay, not great but watchable.",
        "Amazing storyline and fantastic acting!",
        "Boring and uninteresting, fell asleep watching it."
    ]
    
    print("=== ТЕСТИРОВАНИЕ ДООБУЧЕННОЙ МОДЕЛИ ===")
    
    for text in test_texts:
        # Токенизируем текст
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Получаем предсказания
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
        
        # Получаем предсказанный класс
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        # Маппинг классов
        class_names = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
        
        print(f"Текст: {text}")
        print(f"Предсказание: {class_names[predicted_class]} (уверенность: {confidence:.4f})")
        print(f"Все вероятности: {dict(zip(class_names.values(), predictions[0].tolist()))}")
        print("-" * 50)

# Запуск дообучения (раскомментируйте для запуска)
print("=== ДООБУЧЕНИЕ BERT ДЛЯ КЛАССИФИКАЦИИ ===")
print("Внимание: процесс может занять несколько минут...")

# model, tokenizer, results = fine_tune_bert_classifier()
# test_fine_tuned_model(model, tokenizer)

# Альтернативный способ - загрузка уже обученной модели
def load_and_test_pretrained_classifier():
    """Загружает предобученную модель для классификации"""
    
    # Загружаем готовую модель для анализа тональности
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    model.eval()
    
    test_texts = [
        "I love this new phone! It's amazing!",
        "This product is terrible, worst purchase ever.",
        "The item is okay, nothing special about it.",
    ]
    
    print("=== ТЕСТИРОВАНИЕ ГОТОВОЙ МОДЕЛИ КЛАССИФИКАЦИИ ===")
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        # Для этой модели: 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE
        class_names = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        
        print(f"Текст: {text}")
        print(f"Предсказание: {class_names[predicted_class]} (уверенность: {confidence:.4f})")
        print("-" * 50)

# Тестируем готовую модель
load_and_test_pretrained_classifier()
```

## Практические советы для ML инженера

### 1. Оптимизация производительности

```python
# Советы по оптимизации работы с моделями

# 1. Используйте GPU если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Перемещение модели на устройство
model = model.to(device)

# 2. Оптимизация памяти - используйте gradient checkpointing для больших моделей
# model.gradient_checkpointing_enable()

# 3. Пакетная обработка для эффективности
def efficient_batch_processing(texts, model, tokenizer, batch_size=32):
    """Эффективная пакетная обработка текстов"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Токенизация батча
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Перемещаем на устройство
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_results = outputs.logits
            
        results.extend(batch_results.cpu().numpy())
    
    return results

# 4. Кэширование моделей и токенизаторов
# Модели автоматически кэшируются в ~/.cache/huggingface/
```

### 2. Обработка ошибок и валидация

```python
def safe_model_loading(model_name, max_retries=3):
    """Безопасная загрузка модели с повторными попытками"""
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Экспоненциальная задержка

def validate_input_text(text, max_length=512):
    """Валидация входного текста"""
    if not isinstance(text, str):
        raise ValueError("Текст должен быть строкой")
    
    if len(text.strip()) == 0:
        raise ValueError("Текст не может быть пустым")
    
    if len(text) > max_length * 4:  # Примерная оценка
        print(f"Предупреждение: текст очень длинный ({len(text)} символов)")
    
    return text.strip()
```

### 3. Мониторинг и логирование

```python
import logging
import time
from functools import wraps

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Декоратор для логирования времени выполнения"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} выполнен за {end_time - start_time:.2f} секунд")
        return result
    return wrapper

@log_execution_time
def process_text_batch(texts, pipeline):
    """Пример функции с логированием"""
    return pipeline(texts)
```

## Заключение

1. **Токенизаторы**: загрузка, основные методы, работа с парами предложений
2. **Модели BERT**: MLM и NSP задачи с практическими примерами
3. **Pipeline**: готовые решения для различных NLP задач
4. **Fine-tuning**: дообучение BERT для классификации с полным циклом

**Ключевые моменты для практики:**
- Всегда используйте `model.eval()` для инференса
- Применяйте `torch.no_grad()` для экономии памяти
- Используйте пакетную обработку для эффективности
- Валидируйте входные данные
- Логируйте процессы для отладки
- Сохраняйте модели после обучения