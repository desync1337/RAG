# AskDocs
  
  ![Screenshot_3](https://github.com/user-attachments/assets/2ef10fb7-cda2-4e1f-8fb3-e734964b738e)

- Поддерживает PDF и DOCX документы
- Бэкенд на FastApi
- LLM -> OpenRouter api
- uvicorn для быстрого внесения изменений
- Embedder на Sentence Transformers
- Просмотр оригинального документа для проверки точности

# Установка

1. ``` git clone https://github.com/desync1337/RAG/ ```
2. Cкачивание зависимостей ``` pip install -r requirements.txt ``` 

# Использование 

- Расположите все документы в папку DOCS
- Запустите файл embedder.bat для конвертации всех доков
- Запускаем start.bat и ждем инициализации сервера (10 секунд) 
- Откройте настройки справа сверху и введите свой Token (openrouter.ai)

# Возможные ошибки 

- Api error 40-42: Проблема openrouter, поменяйте токен или модель в webmain.py
- Плохой ответ LLM: Я так и не довел промпт до идеала потому что постоянно менял модели и они реагировали на промпты по разному. Поэтому я сделал его максимально простым - для совместимости 




