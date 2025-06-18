import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any

class RAGSystem:
    def __init__(self, embeddings_dir: str, 
                 api_key: str,
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 llm_model: str = "google/gemini-2.0-flash-001"):
        """
        Инициализация RAG-системы с OpenRouter
        :param embeddings_dir: директория с JSON-файлами эмбеддингов
        :param api_key: API ключ для OpenRouter
        :param model_name: модель для кодирования запросов
        :param llm_model: модель LLM для генерации ответов
        """
        self.api_key = api_key
        self.llm_model = llm_model
        
        # Загрузка эмбеддингов и чанков
        self.chunks, self.embeddings = self.load_embeddings(embeddings_dir)
        print(f"Загружено {len(self.chunks)} чанков")
        
        # Инициализация модели для эмбеддингов
        self.embedding_model = SentenceTransformer(model_name)
        
        # Создание индекса для быстрого поиска
        self.embeddings_index = np.array(self.embeddings)
        print("Индекс эмбеддингов готов")
        print(f"Система инициализирована с моделью: {llm_model}")

    def load_embeddings(self, embeddings_dir: str) -> (List[Dict], List[List[float]]):
        """Загрузка эмбеддингов из JSON-файлов"""
        all_chunks = []
        all_embeddings = []
    
        for file in os.listdir(embeddings_dir):
            if file.startswith('embeddings_') and file.endswith('.json'):
                try:
                    with open(os.path.join(embeddings_dir, file), 'r', encoding='utf-8') as f:
                        # Загружаем данные - это должен быть список
                        chunks_list = json.load(f)
                    
                        # Проверяем тип данных
                        if not isinstance(chunks_list, list):
                            print(f"⚠️ Неверный формат в {file}: ожидается список, получен {type(chunks_list)}")
                            continue
                    
                        # Обрабатываем каждый чанк в списке
                        for chunk in chunks_list:
                            # Формируем метаданные из доступных полей
                            metadata = {
                                "source": chunk.get("source", "неизвестно"),
                                "page": chunk.get("page", 0),
                                "fragment_id": chunk.get("fragment_id", "")
                            }
                        
                            all_chunks.append({
                                "text": chunk["text"],
                                "metadata": metadata
                            })
                            all_embeddings.append(chunk["embedding"])
                
                    print(f"✅ Загружено {len(chunks_list)} чанков из {file}")
                except Exception as e:
                    print(f"⛔ Ошибка при загрузке {file}: {str(e)}")
    
        return all_chunks, all_embeddings

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск наиболее релевантных чанков для запроса"""
        # Кодирование запроса
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Вычисление косинусного сходства
        similarities = np.dot(self.embeddings_index, query_embedding) / (
            np.linalg.norm(self.embeddings_index, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Получение индексов наиболее релевантных чанков
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Формирование результатов
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(similarities[idx])
            })
        
        return results

    def generate_response(self, query: str, context_chunks: List[Dict], stream: bool = True) -> str:
        """Генерация ответа с использованием OpenRouter API"""
        # Формирование контекста
        context = "\n\n".join(
            [f"[Документ: {chunk['metadata']['source']}, стр. {chunk['metadata']['page']}]\n{chunk['text']}" 
             for chunk in context_chunks]
        )
        
        # Формирование промта
        messages = [
            {
                "role": "system",
                "content": f"Твое имя - R.A.G ты эффективный поисковик данных по контексту"
                           f"пользователь пишет тебе запрос , и ты должен дать ему наиболее релевантный ответ в зависимости от контекста\n\n"
                           f"Ты можешь брат информацию только из контекста!!"
                           f"Если в контексте не очень релевантная информация то отвечай что не нашел конкретно того чего хочет пользователь"
                           f"но тогда ты должен предложить пользователю 'возможно вы искали...' и 3 максимально близких topic из контекста к запросу"
                           f"КОНТЕКСТ:\n{context}"
            },
            {"role": "user", "content": query}
        ]
        
        # Подготовка запроса к OpenRouter
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": stream
        }
        
        # Отправка запроса
        try:
            if stream:
                return self.stream_response(data, headers)
            else:
                return self.non_stream_response(data, headers)
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"

    def stream_response(self, data: dict, headers: dict) -> str:
        """Обработка потокового ответа"""
        full_response = []
        
        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True
        ) as response:
            if response.status_code != 200:
                error_msg = response.json().get("error", {}).get("message", "Неизвестная ошибка API")
                return f"Ошибка API ({response.status_code}): {error_msg}"

            for chunk in response.iter_lines():
                if chunk:
                    chunk_str = chunk.decode('utf-8').replace('data: ', '').strip()
                    if chunk_str == "[DONE]":
                        break
                    
                    try:
                        chunk_json = json.loads(chunk_str)
                        if "choices" in chunk_json:
                            content = chunk_json["choices"][0]["delta"].get("content", "")
                            if content:
                                # Убираем теги <think>, если они есть
                                cleaned = content.replace('<think>', '').replace('</think>', '')
                                print(cleaned, end='', flush=True)
                                full_response.append(cleaned)
                    except json.JSONDecodeError:
                        pass
        
        print()  # Перенос строки после завершения потока
        return ''.join(full_response)

    def non_stream_response(self, data: dict, headers: dict) -> str:
        """Обработка обычного (не потокового) ответа"""
        data["stream"] = False  # Убедимся, что stream отключен
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("error", {}).get("message", "Неизвестная ошибка API")
            return f"Ошибка API ({response.status_code}): {error_msg}"
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content.replace('<think>', '').replace('</think>', '')

    def ask(self, question: str, top_k: int = 3, stream: bool = True) -> Dict[str, Any]:
        """Полный процесс: поиск + генерация ответа"""
        # Поиск релевантных чанков
        relevant_chunks = self.find_relevant_chunks(question, top_k=top_k)
        
        # Вывод информации о найденных чанках
        print("\n\033[1mНайдены релевантные фрагменты:\033[0m")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"{i}. [Документ: {chunk['metadata']['source']}, Стр. {chunk['metadata']['page']}, Сходство: {chunk['score']:.3f}]")
        
        # Генерация ответа
        print("\n\033[1mОтвет:\033[0m ", end='', flush=True)
        answer = self.generate_response(question, relevant_chunks, stream=stream)
        
        # Формирование результата
        return {
            "question": question,
            "answer": answer,
            "contexts": relevant_chunks
        }

# Основной цикл чата
def main(api_key: str):
    # Инициализация системы
    rag = RAGSystem(
        embeddings_dir="embedding",
        api_key=api_key,
        llm_model="deepseek/deepseek-r1"  # Можете изменить на другую модель
    )
    
    print("\n" + "="*50)
    print("RAG Система с OpenRouter API")
    print("Доступные команды:")
    print("/exit - выход из системы")
    print("/model <название> - сменить модель LLM")
    print("/top <число> - изменить количество используемых фрагментов (по умолчанию 3)")
    print("="*50 + "\n")
    
    current_top_k = 5
    current_model = "deepseek/deepseek-r1"
    
    while True:
        try:
            user_input = input("\n\033[1mВы:\033[0m ")
            
            if user_input.lower() == '/exit':
                print("Завершение работы...")
                break
                
            # Обработка команд
            if user_input.startswith('/model'):
                new_model = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else None
                if new_model:
                    rag.llm_model = new_model
                    current_model = new_model
                    print(f"Модель изменена на: {new_model}")
                else:
                    print("Текущая модель:", current_model)
                continue
                    
            if user_input.startswith('/top'):
                try:
                    new_top = int(user_input.split(maxsplit=1)[1])
                    if 1 <= new_top <= 10:
                        current_top_k = new_top
                        print(f"Используется {new_top} фрагментов контекста")
                    else:
                        print("Ошибка: значение должно быть между 1 и 10")
                except:
                    print("Ошибка: используйте /top <число>")
                continue
                
            # Обработка вопроса
            rag.ask(user_input, top_k=current_top_k)
            
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nОшибка: {str(e)}")

if __name__ == "__main__":
    API_KEY = "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"  # Ваш API-ключ
    
    if not API_KEY:
        print("ОШИБКА: Необходимо указать API ключ для OpenRouter!")
        print("Получите ключ: https://openrouter.ai/settings/keys")
    else:
        main(API_KEY)