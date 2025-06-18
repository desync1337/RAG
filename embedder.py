import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import time
import glob

class RAGEmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=None):
        """
        Инициализация модели для генерации эмбеддингов
        :param model_name: название предобученной модели
        :param device: устройство для вычислений (cuda, cpu или mps)
        """
        # Автовыбор устройства если не указано
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Используется устройство: {self.device}")
        print(f"Загрузка модели: {model_name}...")
        
        start_time = time.time()
        self.model = SentenceTransformer(model_name, device=self.device)
        load_time = time.time() - start_time
        
        print(f"Модель загружена за {load_time:.2f} секунд")
        print(f"Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Генерация эмбеддингов для списка текстов
        :param texts: список текстовых чанков
        :param batch_size: размер батча для обработки
        :return: numpy array с эмбеддингами
        """
        print(f"Генерация эмбеддингов для {len(texts)} текстов...")
        start_time = time.time()
        
        # Включение прогресс-бара
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device
        )
        
        process_time = time.time() - start_time
        print(f"Эмбеддинги сгенерированы за {process_time:.2f} секунд")
        print(f"Скорость обработки: {len(texts)/process_time:.2f} текстов/сек")
        
        return embeddings

    def process_directory(self, input_dir, output_dir):
        """
        Обработка всех JSON-файлов в директории
        :param input_dir: путь к директории с JSON-файлами
        :param output_dir: директория для сохранения результатов
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Создана директория для результатов: {output_dir}")
        
        # Поиск всех JSON-файлов
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            print(f"Ошибка: JSON-файлы не найдены в {input_dir}")
            return
        
        print(f"Найдено {len(json_files)} файлов для обработки")
        
        total_chunks = 0
        processed_files = 0
        
        for i, json_file in enumerate(json_files, 1):
            filename = os.path.basename(json_file)
            output_path = os.path.join(output_dir, f"embeddings_{filename}")
            
            print(f"\n[{i}/{len(json_files)}] Обработка: {filename}")
            
            try:
                # Чтение JSON-файла
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)  # Загружаем список фрагментов

                # Извлечение текстов для обработки
                texts = [item['text'] for item in data]  # Извлекаем тексты из каждого объекта
                total_chunks += len(texts)

                # Генерация эмбеддингов
                embeddings = self.generate_embeddings(texts)

                # Добавляем эмбеддинги к каждому фрагменту
                for idx, item in enumerate(data):
                    item['embedding'] = embeddings[idx].tolist()  # Добавляем эмбеддинг в каждый объект

                # Сохранение результатов
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)  # Сохраняем весь список объектов

                print(f"  Успешно: сохранено {len(data)} эмбеддингов в {output_path}")
                processed_files += 1
                
            except Exception as e:
                print(f"  Ошибка при обработке файла {filename}: {str(e)}")
        
        print(f"\nИтого обработано: {processed_files} файлов, {total_chunks} чанков")

if __name__ == "__main__":
    # Конфигурация
    INPUT_DIR = "data"  # Директория с JSON-файлами из предыдущего скрипта
    OUTPUT_DIR = "embedding"  # Директория для сохранения эмбеддингов
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Лучшая модель для русского
    
    # Инициализация и запуск
    generator = RAGEmbeddingGenerator(model_name=MODEL_NAME)
    generator.process_directory(INPUT_DIR, OUTPUT_DIR)
    
    print("\nОбработка завершена! Эмбеддинги готовы для RAG-системы.")