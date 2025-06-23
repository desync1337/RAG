const openSettingsBtn = document.getElementById('settings-button');
const tokenModal = document.getElementById('tokenModal');
const closeBtn = document.querySelector('.close-btn');
const cancelBtn = document.getElementById('cancelBtn');
const saveTokenBtn = document.getElementById('saveTokenBtn');
const apiTokenInput = document.getElementById('apiToken');
const statusMessage = document.getElementById('statusMessage');

openSettingsBtn.addEventListener('click', () => {
            // Загрузить сохраненный токен при открытии
            const savedToken = localStorage.getItem('apiToken');
            if (savedToken) {
                apiTokenInput.value = savedToken;
            }
            tokenModal.style.display = 'flex';
        });

        // Закрыть модальное окно
        function closeModal() {
            tokenModal.style.display = 'none';
            clearStatus();
        }

        // Обработчики закрытия
        closeBtn.addEventListener('click', closeModal);
        cancelBtn.addEventListener('click', closeModal);

        // Закрыть при клике вне окна
        window.addEventListener('click', (e) => {
            if (e.target === tokenModal) {
                closeModal();
            }
        });

    saveTokenBtn.addEventListener('click', () => {
            const token = apiTokenInput.value.trim();
            
            if (!token) {
                showStatus('Ошибка: Токен не может быть пустым', 'error');
                return;
            }
            
            // Валидация токена (пример)
            if (token.length < 20) {
                showStatus('Ошибка: Токен слишком короткий', 'error');
                return;
            }
            
            try {
                // Сохраняем в localStorage
                localStorage.setItem('apiToken', token);
                showStatus('Токен успешно сохранен!', 'success');
                
                // Автоматически закрываем через 1.5 секунды
                setTimeout(() => {
                    closeModal();
                }, 1500);
                
            } catch (e) {
                showStatus(`Ошибка сохранения: ${e.message}`, 'error');
            }
        });

        // Показать статусное сообщение
        function showStatus(message, type) {
            statusMessage.textContent = message;
            statusMessage.className = 'status-message';
            statusMessage.classList.add(`status-${type}`);
            statusMessage.style.display = 'block';
        }

        // Очистить статус
        function clearStatus() {
            statusMessage.style.display = 'none';
            statusMessage.textContent = '';
        }

        // Инициализация: загружаем токен при старте (если нужно)
        window.addEventListener('DOMContentLoaded', () => {
            const savedToken = localStorage.getItem('apiToken');
            if (savedToken) {
                // Можно использовать токен в приложении
                console.log('Токен загружен из хранилища');
            }
        });

document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const questionInput = document.getElementById('question-input');
    const sendBtn = document.getElementById('send-btn');
    const typingIndicator = document.getElementById('typing-indicator');
    const contextSources = document.getElementById('context-sources');
    const documentContent = document.getElementById('document-content');

    // Показать экран загрузки
    const loadingScreen = document.getElementById('loading-screen');
    const errorMessage = document.getElementById('error-message');
    const API_BASE = 'http://localhost:8000';

    let isBotResponding = false;
    let isServerAvailable = false;
    // Функция проверки сервера
    async function checkServerConnection() {
        try {
            const response = await fetch(`${API_BASE}/system_info`);
            if (response.ok) {
                isServerAvailable = true;
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.style.display = 'none';
                }, 500);
            } else {
                throw new Error('Сервер не отвечает');
            }
        } catch (error) {
            errorMessage.textContent = `Ошибка: ${error.message}. Пожалуйста, убедитесь что сервер запущен.`;
            errorMessage.style.display = 'block';
        
            // Повторная проверка каждые 5 секунд
            setTimeout(checkServerConnection, 1000);
        }
    }

    // Проверить соединение перед инициализацией
    checkServerConnection();


    function renderMarkdown(text) {
        return marked.parse(text);
    }
    
    // Add message to chat
    function addMessage(text, isUser, isStreaming = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    if (isUser || !isStreaming) {
        if (isUser) {
            messageDiv.textContent = text;
        } else {
            messageDiv.innerHTML = renderMarkdown(text);
        }
        chatMessages.appendChild(messageDiv);
    } else {
        // Для плавного вывода ответа бота
        messageDiv.id = 'streaming-message';
        chatMessages.appendChild(messageDiv);
        typeResponse(messageDiv, text);
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Добавьте новую функцию для плавного вывода:
function typeResponse(element, text, speed = 20) {
    let i = 0;
    const fullText = text;
    element.innerHTML = '';
    
    function type() {
        if (i < fullText.length) {
            element.innerHTML = renderMarkdown(fullText.substring(0, i + 1));
            i+=2;
            setTimeout(type, speed);
        }
    }
    
    type();
}
    
    // Show/hide typing indicator
    function showTyping(show) {
        typingIndicator.style.display = show ? 'flex' : 'none';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Load document content
    async function loadDocumentContent(filename) {
        try {
            const response = await fetch(`${API_BASE}/source_content?filename=${encodeURIComponent(filename)}`);
            if (!response.ok) {
                throw new Error('Не удалось загрузить содержимое документа');
            }
            
            const data = await response.json();
            documentContent.innerHTML = `<div class="document-text">${data.content.replace(/\n/g, '<br>')}</div>`;
        } catch (error) {
            console.error('Ошибка загрузки документа:', error);
            documentContent.innerHTML = `<p class="error">Ошибка: ${error.message}</p>`;
        }
    }
    
    // Render context sources
    function renderContextSources(contexts) {
        contextSources.innerHTML = '';
        
        if (!contexts || contexts.length === 0) {
            contextSources.innerHTML = '<div class="source-ref">Источники не найдены</div>';
            return;
        }
        
        contexts.forEach((ctx, index) => {
            const sourceRef = document.createElement('div');
            sourceRef.className = 'source-ref';
            sourceRef.innerHTML = `
                <div class="title">${ctx.metadata.source.split('/').pop()}</div>
                <div class="page">Страница: ${ctx.metadata.page + 1}, Сходство: ${ctx.score.toFixed(3)}</div>
                <div class="excerpt">${ctx.text.substring(0, 120)}...</div>
            `;
            
            // Add click event to view full document
            sourceRef.addEventListener('click', () => {
                loadDocumentContent(ctx.metadata.source);
            });
            
            contextSources.appendChild(sourceRef);
        });
    }
    
    // Send question to API
 async function sendQuestion() {
    if (!isServerAvailable) {
        alert('Сервер недоступен. Пожалуйста, дождитесь подключения.');
        return;
    }
    
    const question = questionInput.value.trim();
    if (!question || isBotResponding) return;
    
    // Получаем токен из localStorage
    const apiToken = localStorage.getItem('apiToken');

    if (!apiToken) {
        alert('API токен не найден! Пожалуйста, нажмите на кнопку настроек и введите токен.');
        openSettingsBtn.click();
        return;
    }
    
    isBotResponding = true;
    questionInput.disabled = true;
    sendBtn.disabled = true;
    
    contextSources.innerHTML = '<div class="source-ref">Поиск источников...</div>';
    addMessage(question, true);
    questionInput.value = '';
    sendBtn.disabled = true;
    
    showTyping(true);
    
    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json; charset=utf-8' // Явно указываем кодировку
            },
            body: JSON.stringify({ 
                question: question,
                top_k: 3,
                stream: false,
                api_key: apiToken
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Ошибка API: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        
        // Добавляем ответ бота в чат
        addMessage(data.answer, false, true);
        
        // Показываем источники
        renderContextSources(data.contexts);
        
    } catch (error) {
        console.error('Error:', error);
        
        if (error.message.includes('401') || 
            error.message.includes('токен') || 
            error.message.includes('аутентификац')) {
            addMessage("Ошибка авторизации: неверный или отсутствующий API токен. Пожалуйста, проверьте настройки.", false);
            openSettingsBtn.click();
        } else {
            addMessage(`Ошибка: ${error.message}`, false);
        }
        
        contextSources.innerHTML = `<div class="source-ref">Ошибка получения источников: ${error.message}</div>`;
    } finally {
        showTyping(false);
        sendBtn.disabled = false;
        questionInput.disabled = false;
        isBotResponding = false;
    }
}
    
    // Event listeners
    sendBtn.addEventListener('click', sendQuestion);
    
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQuestion();
    });
    
    // Initial message
    addMessage("Привет! Я ваш корпоративный помощник. Задайте вопрос о документах в базе знаний.", false);
}); 