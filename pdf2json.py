import os
import json
import re
import PyPDF2
import nltk
from pathlib import Path
from nltk.tokenize import sent_tokenize
from nltk import download as nltk_download
import docx  # Добавлено для работы с DOCX

nltk.download('punkt_tab')

# Автоматическая загрузка необходимых ресурсов NLTK
def download_nltk_resources(lang="russian"):
    try:
        nltk_download('punkt', quiet=True)
        
        # Проверка доступности конкретной языковой модели
        resource_path = f"tokenizers/punkt/{lang}.pickle"
        try:
            nltk.data.find(resource_path)
        except LookupError:
            # Специальная загрузка для русского языка
            nltk_download('perluniprops', quiet=True)
            nltk_download('nonbreaking_prefixes', quiet=True)
            
            # Пробуем альтернативное название для русского
            if lang == "russian":
                try:
                    nltk_download('punkt_tab', quiet=True)
                except:
                    print("Используем универсальную модель токенизации")
    except Exception as e:
        print(f"Ошибка загрузки ресурсов NLTK: {e}")

def segment_text(text, lang="russian", max_tokens=200):
    """Разбивает текст на смысловые фрагменты"""
    # Нормализация текста
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    # Стратегии сегментации
    fragments = []
    
    # 1. Разделение по абзацам
    if '\n\n' in text:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs and all(len(p) > 30 for p in paragraphs):
            fragments = paragraphs
    
    # 2. Разделение по предложениям (если абзацы не сработали)
    if not fragments:
        try:
            sentences = sent_tokenize(text, language=lang)
        except:
            sentences = sent_tokenize(text)  # fallback на стандартную модель
        
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(' '.join(current_chunk)) > max_tokens:
                fragments.append(' '.join(current_chunk[:-1]))
                current_chunk = [sentence]
        
        if current_chunk:
            fragments.append(' '.join(current_chunk))
    
    # 3. Резервная стратегия (фикс. длина)
    if not fragments:
        fragments = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    
    return fragments

def pdf_to_json(pdf_path, output_dir, lang="russian"):
    """Конвертирует PDF файл в JSON со структурированными фрагментами"""
    fragments = []
    filename = Path(pdf_path).name
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                
                page_fragments = segment_text(text, lang)
                for frag_num, fragment in enumerate(page_fragments, start=1):
                    fragments.append({
                        "text": fragment,
                        "page": page_num,
                        "fragment_id": f"{filename}-p{page_num}-f{frag_num}",
                        "source": filename
                    })
    
    except Exception as e:
        print(f"Ошибка обработки {pdf_path}: {str(e)}")
        return
    
    save_fragments(fragments, pdf_path, output_dir)

def docx_to_json(docx_path, output_dir, lang="russian"):
    """Конвертирует DOCX файл в JSON с улучшенной обработкой страниц"""
    fragments = []
    filename = Path(docx_path).name
    
    try:
        doc = docx.Document(docx_path)
        
        # Собираем текст с сохранением информации о разрывах страниц
        page_texts = []
        current_page = []
        
        # Обрабатываем все элементы документа
        for element in doc.element.body:
            if element.tag.endswith('p'):  # Абзац
                para = docx.text.paragraph.Paragraph(element, doc)
                text = para.text.strip()
                if text:
                    # Проверяем разрыв страницы перед абзацем
                    if para.paragraph_format.page_break_before:
                        if current_page:
                            page_texts.append("\n\n".join(current_page))
                            current_page = []
                    current_page.append(text)
                    
            elif element.tag.endswith('tbl'):  # Таблица
                table = docx.table.Table(element, doc)
                table_text = []
                for row in table.rows:
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            table_text.append(cell_text)
                if table_text:
                    current_page.append("\n".join(table_text))
        
        # Добавляем последнюю страницу
        if current_page:
            page_texts.append("\n\n".join(current_page))
        
        # Обрабатываем каждую "страницу"
        for page_num, text in enumerate(page_texts, start=1):
            if not text:
                continue
                
            page_fragments = segment_text(text, lang)
            for frag_num, fragment in enumerate(page_fragments, start=1):
                fragments.append({
                    "text": fragment,
                    "page": page_num,
                    "fragment_id": f"{filename}-p{page_num}-f{frag_num}",
                    "source": filename
                })
    
    except Exception as e:
        print(f"Ошибка обработки {docx_path}: {str(e)}")
        return
    
    save_fragments(fragments, docx_path, output_dir)

def save_fragments(fragments, file_path, output_dir):
    """Сохраняет фрагменты в JSON файл"""
    if not fragments:
        print(f"Файл {Path(file_path).name} не содержит текста")
        return
        
    output_path = Path(output_dir) / f"{Path(file_path).stem}.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fragments, f, ensure_ascii=False, indent=2)
        print(f"Успешно обработан: {Path(file_path).name} -> {output_path.name}")
    except Exception as e:
        print(f"Ошибка записи {output_path}: {str(e)}")

def convert_files_to_json(input_dir, output_dir, lang="russian"):
    """Обрабатывает все PDF и DOCX файлы в директории"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    download_nltk_resources(lang)
    
    # Находим все поддерживаемые файлы
    files = []
    for ext in ['*.pdf', '*.docx']:
        files.extend(Path(input_dir).glob(ext))
    
    total = len(files)
    
    if total == 0:
        print("Файлы не найдены!")
        return
    
    print(f"Найдено файлов для обработки: {total}")
    
    for i, filepath in enumerate(files, 1):
        print(f"\nОбработка ({i}/{total}): {filepath.name}")
        
        if filepath.suffix.lower() == '.pdf':
            pdf_to_json(filepath, output_dir, lang)
        elif filepath.suffix.lower() == '.docx':
            docx_to_json(filepath, output_dir, lang)

if __name__ == "__main__":
    # Конфигурация
    INPUT_DIRECTORY = "DOCS"  # Путь к исходной директории с файлами
    OUTPUT_DIRECTORY = "data"  # Путь для сохранения JSON
    LANGUAGE = "russian"  # Язык обработки
    
    convert_files_to_json(INPUT_DIRECTORY, OUTPUT_DIRECTORY, LANGUAGE)
    print("\nКонвертация завершена.")