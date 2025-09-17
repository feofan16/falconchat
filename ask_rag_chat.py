import re
import requests
import sys
import os
import hashlib
from typing import List, Dict, Optional, TYPE_CHECKING
import logging
from dataclasses import dataclass
from datetime import datetime
import time
import threading
from contextlib import contextmanager
if TYPE_CHECKING:
    from search_windows import AdvancedHybridSearch
 
# Исправление кодировки для Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
# --- LLM model selection (Ollama tags) ---

# 1) Лёгкий и "инструкционный" (рекомендую начать с него)
# MODEL_DEFAULT = "qwen3:4b-instruct-2507-q4_K_M"
# 2) Сильнее на рассуждении (медленнее)
# MODEL_DEFAULT = "qwen3:4b-thinking-2507-q4_K_M"
# MODEL_DEFAULT = "qwen3:4b-thinking-2507-q8_0"
# 3) Более высокое качество
# MODEL_DEFAULT = "qwen3:8b-q8_0"
# MODEL_DEFAULT = "qwen3:30b-a3b-instruct-2507-q4_K_M"
# 4) Старый вариант (для отката)
MODEL_DEFAULT = "qwen2.5:7b-instruct-q5_K_M"

MODEL = os.getenv("LLM_MODEL", MODEL_DEFAULT)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
REQUEST_TIMEOUT = (10, 600)
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
MAX_FRAGMENTS = int(os.getenv("RAG_MAX_FRAGMENTS", "10"))         # сколько фрагментов тянем из RAG
MAX_GROUPS    = int(os.getenv("RAG_MAX_GROUPS", "5"))             # сколько «групп источников» в enhance_context
MAX_CITATIONS = int(os.getenv("RAG_CITATIONS_MAX", "8"))          # максимум цитат в ответе
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
MAX_CHARS_PER_FRAGMENT = int(os.getenv("RAG_CHARS_PER_FRAGMENT", "1500"))
NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "1024"))
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "16384"))

# --- OKPD2 hint (no prompt injection) ---
OKPD_HINT_ENABLE = os.getenv("OKPD_HINT_ENABLE", "1") not in ("0", "false", "False")
OKPD_HINT_MIN_CONF = float(os.getenv("OKPD_HINT_MIN_CONF", "0.5")) # порог уверенности для приписки 
# --- Smart-Retry config ---
SMART_RETRY_ENABLE = os.getenv("SMART_RETRY_ENABLE", "1") not in ("0", "false", "False")
SMART_RETRY_MIN_HITS = int(os.getenv("SMART_RETRY_MIN_HITS", "3"))          # если нашли меньше — пробуем ещё раз
SMART_RETRY_TOPK_PER_QUERY = int(os.getenv("SMART_RETRY_TOPK_PER_QUERY", "5"))
SMART_RETRY_GROUNDING_THRESHOLD = float(os.getenv("SMART_RETRY_GROUNDING_THRESHOLD", "0.35"))

try:
    # ваш классификатор; безопасно, если модуля нет
    from scripts.utils.labels import classify_factory_item, okpd_canon  # noqa
except Exception:
    classify_factory_item = None  # type: ignore
    okpd_canon = None  # type: ignore

# сигналы «вопрос про ОКПД2»
OKPD_INTENT_RE = re.compile(r"\b(окпд2?|okpd)\b|\bкод\s+окпд2?\b", re.I)
SHAPE_RE  = re.compile(r"\b(круг|квадрат|шестигранник|полоса|лист|лента|катанка|проволока|труба|уголок|швеллер|двутавр|арматура)\b", re.I)
MATERIAL_RE = re.compile(
    r"\b("
    r"ст[0-9]+|сталь\s*\d+|aisi\s*\d+|"
    r"12[хx]18[нnh]10[тt]|14[хx]17[нnh]2|40[хx]13|30[хx]гса|40[хx]|09г2с|65г|р6м5|[хx]12мф|"
    r"амг\d?|1561|ад31|л[0-9]+|лс59|л63|бр[а-я0-9]+|м1|"
    r"(?:vt|вт)\d?-?\d?"
    r")\b",
    re.I
)

def _okpd_detect_intent(q: str) -> bool:
    qn = (q or "").lower()
    if OKPD_INTENT_RE.search(qn):
        return True
    # форма + материал — тоже считаем намерением
    return bool(SHAPE_RE.search(qn) and MATERIAL_RE.search(qn))

def _okpd_extract_item(q: str) -> str:
    tail = re.split(r'(?i)\b(для|по|на)\b', q, maxsplit=1)
    tail = tail[-1] if len(tail) > 1 else q
    tail = re.sub(r'(?i)\b(какой|найд[ий]|подскажи|код|окпд2?|okpd)\b', ' ', tail)
    return re.sub(r'\s+', ' ', tail).strip()

class CtxLog(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        prefix = []
        if self.extra.get("qa_id"): prefix.append(f"qa={self.extra['qa_id']}")
        if self.extra.get("cid"):   prefix.append(f"cid={self.extra['cid']}")
        return (("[" + " ".join(prefix) + "] " if prefix else "") + str(msg), kwargs)

@contextmanager
def timed(log, label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.debug(f"{label} finished in {(time.perf_counter()-t0)*1000:.1f} ms")

@dataclass
class PromptTemplate:
    """Шаблоны промптов для разных типов вопросов"""
    
    SYSTEM_EXPERT = """Ты — экспертный ассистент по технической документации Falcon.
Твоя задача — предоставлять точные, практичные и хорошо структурированные ответы на основе предоставленных фрагментов документации.

ПРИНЦИПЫ РАБОТЫ:
1. Точность: Используй ТОЛЬКО информацию из предоставленных фрагментов
2. Структура: Организуй ответ логично с заголовками и списками
3. Практичность: Давай конкретные шаги
4. Честность: Если информации недостаточно, явно об этом сообщи
5. Контекст: Учитывай взаимосвязи между фрагментами

ФОРМАТ ОТВЕТА:
- Используй Markdown для форматирования
- Начинай с краткого резюме (1-3 предложения)
- Структурируй основную часть с подзаголовками
- (Опционально) Завершай практическими рекомендациями или следующими шагами"""

    SYSTEM_ANALYTICAL = """Ты — аналитический помощник, специализирующийся на глубоком анализе технической документации.
Твоя задача — не просто отвечать на вопросы, но и:
- Выявлять связи между концепциями
- Предоставлять контекст и предпосылки
- Объяснять последствия и зависимости
- Предупреждать о потенциальных проблемах"""

    CHAIN_OF_THOUGHT = """Подумай пошагово и проверь себя внутренне, 
НО выводи только готовый ответ с заголовками, списками, примерами."""

    FEW_SHOT_EXAMPLES = """"""

class EnhancedRAGChat:
    """Улучшенная система генерации ответов с проверкой grounding"""
    
    def __init__(self, searcher: Optional['AdvancedHybridSearch'] = None):
        self.templates = PromptTemplate()
        self.conversation_history = []
        self.context_cache = {}
        self.searcher = searcher  # Сохраняем ссылку на searcher
        self._lock = threading.RLock()

    def classify_question(self, question: str) -> str:
        """Классификация типа вопроса для выбора оптимальной стратегии"""
        question_lower = question.lower()
        
        patterns = {
            'howto': ['как', 'каким образом', 'способ', 'настроить', 'установить', 'создать'],
            'troubleshoot': ['ошибка', 'проблема', 'не работает', 'исправить', 'решение', 'почему'],
            'explain': ['что такое', 'объясни', 'зачем', 'почему', 'принцип', 'различие'],
            'reference': ['список', 'параметры', 'функции', 'методы', 'свойства', 'команды'],
            'compare': ['сравни', 'отличие', 'разница', 'лучше', 'выбрать', 'или']
        }
        
        for q_type, keywords in patterns.items():
            if any(kw in question_lower for kw in keywords):
                return q_type
        
        return 'general'
    
    def check_answer_grounding(
        self,
        answer: str,
        chunks: List[Dict],
        *,
        min_sent_len: int = 20,
        pool_max: int = 8,
        fuzz_threshold: int = 70
    ) -> float:
        """Оценка доли предложений ответа, подтверждённых источниками."""
        if not chunks or not answer:
            return 0.0

        # 1) Очистка разметки/шума
        answer_clean = re.sub(r"```.*?```", "", answer, flags=re.S)
        answer_clean = re.sub(r"`[^`]+`", "", answer_clean)                      # inline code
        answer_clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", answer_clean)     # [text](url)
        answer_clean = re.sub(r"https?://\S+", "", answer_clean)                 # bare urls
        # Убираем приписку с подсказкой OKPD
        answer_clean = re.sub(r"(?m)^\s*>\s*ℹ️\s*Подсказка классификатора[^\n]*\n?", "", answer_clean)

        # 2) Сплит на предложения + лёгкая зачистка буллетов
        sentences = [
            s.strip(" \t\n\r-–—•*") for s in re.split(r'(?<=[.!?…])\s+', answer_clean) if s.strip()
        ]

        total = 0
        covered = 0

        def _norm(s: str) -> str:
            s = s.lower().replace('ё', 'е')
            s = re.sub(r"\s+", " ", s)
            return s.strip()

        # 3) Пул источников (нормализованный)
        pool_texts = [_norm(c.get('text', ''))[:1200] for c in chunks[:pool_max] if c.get('text')]
        if not pool_texts:
            return 0.0

        # 4) Предкомпиляция и токены источников
        tok_re = re.compile(r'[а-яa-z0-9]{3,}', re.I)
        pool_tokens = set()
        for t in pool_texts:
            pool_tokens.update(tok_re.findall(t))

        # 5) Попытка импортировать rapidfuzz
        try:
            from rapidfuzz import process, fuzz  # type: ignore
            have_rf = True
        except Exception:
            have_rf = False

        for sent in sentences:
            if len(sent) < min_sent_len:
                continue
            total += 1

            s = _norm(sent)
            kws = set(tok_re.findall(s))

            # 5.1 Быстрое пересечение по токенам
            if kws:
                overlap = len(kws & pool_tokens) / max(1, len(kws))
                if overlap >= 0.5:
                    covered += 1
                    continue

            # 5.2 Точная подстрока для коротких предложений
            if len(s) <= 220 and any(s in src for src in pool_texts):
                covered += 1
                continue

            # 5.3 Fuzzy по нескольким источникам с динамическим порогом
            if have_rf:
                thr = fuzz_threshold
                if len(s) < 80:
                    thr += 10
                elif len(s) > 200:
                    thr -= 5
                thr = max(60, min(90, thr))

                try:
                    match = process.extractOne(s, pool_texts, scorer=fuzz.partial_token_set_ratio)
                    if match and match[1] >= thr:
                        covered += 1
                except Exception:
                    pass

        # 6) Итог
        if total == 0:
            return 0.0
        return covered / total

    def enhance_context(self, chunks: List[Dict], question: str) -> str:
        """Улучшенная обработка контекста с группировкой и приоритизацией"""
        if not chunks:
            return ""
        
        # Группируем фрагменты по источникам и разделам
        grouped = {}
        for chunk in chunks:
            key = (chunk.get('book', 'Unknown'), chunk.get('section', 'Unknown'))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(chunk)
        
        # Сортируем группы по релевантности
        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: max(c.get('score', 0) for c in x[1]),
            reverse=True
        )
        
        # Формируем структурированный контекст
        context_parts = []

        for (book, section), group_chunks in sorted_groups[:MAX_GROUPS]:
            context_parts.append(f"\n### {book} - {section}\n")
            
            # Объединяем фрагменты из одного раздела
            combined_text = self._merge_overlapping_chunks(group_chunks)
            
            # ограничение длины
            for text_part in combined_text:
                context_parts.append(
                    f"{text_part[:MAX_CHARS_PER_FRAGMENT]}\n"
                )

        return "\n".join(context_parts)
    
    def _merge_overlapping_chunks(self, chunks: List[Dict]) -> List[str]:
        """Объединение перекрывающихся фрагментов (устойчиво к page=None)."""
        if not chunks:
            return []
        
        def _is_parent(c: Dict) -> bool:
            return (c.get('is_parent') is True) or ((c.get('meta') or {}).get('is_parent') is True)

        # убрать родителей
        pure = [c for c in chunks if not _is_parent(c)]
        if not pure:
            return [(c.get('text') or "") for c in chunks if c.get('text')]

        def _page_key(x: Dict) -> tuple:
            p = x.get('page', None)
            # None уводим в конец
            return (p is None, p if isinstance(p, int) else 10**9, x.get('id', 0))

        sorted_chunks = sorted(pure, key=_page_key)
        if not sorted_chunks:  # доп. страховка
            return [ (c.get('text') or "") for c in chunks if c.get('text') ]

        current_text = sorted_chunks[0].get('text', '') or ''
        current_page = sorted_chunks[0].get('page')
        current_page = current_page if isinstance(current_page, int) else None

        merged = []

        for chunk in sorted_chunks[1:]:
            p = chunk.get('page')
            p = p if isinstance(p, int) else None

            # склеиваем только если обе страницы известны и разница <= 1
            if (current_page is not None) and (p is not None) and (p - current_page <= 1):
                overlap = self._find_overlap(current_text, chunk.get('text', '') or '')
                if overlap and len(overlap) > 50:
                    current_text = current_text + (chunk.get('text', '') or '')[len(overlap):]
                else:
                    current_text = current_text + "\n\n" + (chunk.get('text', '') or '')
                current_page = p
            else:
                merged.append(current_text)
                current_text = (chunk.get('text', '') or '')
                current_page = p

        merged.append(current_text)
        return merged

    def _find_overlap(self, text1: str, text2: str) -> str:
        """Поиск общей части между концом text1 и началом text2"""
        max_overlap = min(200, len(text1), len(text2))
        
        for i in range(max_overlap, 20, -1):
            if text1[-i:] == text2[:i]:
                return text2[:i]
        return ""
    
    def build_enhanced_prompt(self, question: str, context: str, q_type: str) -> str:
        """Построение улучшенного промпта с учетом типа вопроса"""
        
        base_prompt = f"""КОНТЕКСТ ИЗ ДОКУМЕНТАЦИИ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

{self.templates.CHAIN_OF_THOUGHT}

ТВОЙ ОТВЕТ:"""
        
        type_instructions = {
            'howto': """""",
            
            'troubleshoot': """
ФОРМАТ ОТВЕТА ДЛЯ РЕШЕНИЯ ПРОБЛЕМЫ:
1. Описание проблемы
2. Возможные причины (от наиболее вероятных)
3. Диагностика (как проверить каждую причину)
4. Решения для каждой причины
5. Профилактика на будущее""",
            
            'explain': """
ФОРМАТ ОТВЕТА ДЛЯ ОБЪЯСНЕНИЯ:
1. Простое определение (ELI5)
2. Техническое описание
3. Ключевые компоненты/аспекты
4. Примеры использования
5. Связь с другими концепциями""",
            
            'reference': """
ФОРМАТ ОТВЕТА ДЛЯ СПРАВКИ:
1. Краткое описание
2. Таблица/список с параметрами
3. Описание каждого элемента
4. Значения по умолчанию
5. Примеры использования""",
            
            'compare': """
ФОРМАТ ОТВЕТА ДЛЯ СРАВНЕНИЯ:
1. Краткий вывод
2. Таблица сравнения
3. Детальный анализ различий
4. Рекомендации по выбору
5. Примеры сценариев использования"""
        }
        
        if q_type in type_instructions:
            base_prompt = base_prompt.replace(
                "ТВОЙ ОТВЕТ:",
                type_instructions[q_type] + "\n\nТВОЙ ОТВЕТ:"
            )
        
        return base_prompt
    
    def post_process_answer(self, answer: str, chunks: List[Dict]) -> str:
        """Постобработка ответа: очистка и форматирование"""
        if not answer:
            return "Не удалось сгенерировать ответ."
        
        # Удаляем артефакты рассуждений
        answer = re.sub(r'ШАГ \d+:.*?\n', '', answer)
        answer = re.sub(r'(Анализирую|Думаю|Рассматриваю|Проверяю).*?\n', '', answer)
        
        # Улучшаем форматирование кода
        answer = re.sub(r'```(\w+)?\n', r'```\1\n', answer)
        
        # Проверяем минимальную длину
        if len(answer) < 100:
            answer = self._expand_short_answer(answer, chunks)
        
        return answer
    
    def _expand_short_answer(self, answer: str, chunks: List[Dict]) -> str:
        """Расширение слишком короткого ответа"""
        if not chunks:
            return answer
        
        expanded = answer + "\n\n### Дополнительная информация:\n"
        
        for chunk in chunks[:2]:
            text = chunk.get('text', '')[:200]
            if text:
                expanded += f"\n{text}...\n"
                expanded += f"*См. {chunk.get('book', '')}, {chunk.get('section', '')}, стр. {chunk.get('page', 0)}*\n"
        
        return expanded
    
    def generate_answer(
            self, question: str, top_k: int = None, use_cot: bool = True,
            chunks: Optional[List[Dict]] = None, *, qa_id: Optional[str] = None,
            conversation_id: Optional[str] = None
        ) -> Dict:
        """Принимаем chunks как параметр"""
        start_time = datetime.now()

        if top_k is None:
            top_k = MAX_FRAGMENTS

        log = CtxLog(logger, {"qa_id": qa_id, "cid": conversation_id})
        log.debug(f"generate_answer start: top_k={top_k}, use_cot={use_cot}, chunks_in={len(chunks) if chunks else 0}")
        
        # OKPD2: мягкое вычисление подсказки (не влияет на промпт)
        okpd_hint = None
        okpd_item = None
        if OKPD_HINT_ENABLE and _okpd_detect_intent(question):
            okpd_item = _okpd_extract_item(question)
            if okpd_item and len(okpd_item) > 255:
                okpd_item = okpd_item[:255]
            if classify_factory_item:
                try:
                    code, conf, desc, info = classify_factory_item(okpd_item)  # type: ignore
                    if code and code != "UNSURE":
                        if okpd_canon:
                            code = okpd_canon(code) or code  # type: ignore
                        okpd_hint = {"item": okpd_item, "code": code, "conf": float(conf or 0.0), "desc": desc or "", "info": info or {}}
                        log.info(f"okpd_hint=True item='{okpd_item}' code={code} conf={conf:.2f}")
                except Exception as e:
                    log.warning(f"okpd_hint_classifier_error: {e}")
        
        # Классифицируем вопрос
        q_type = self.classify_question(question)
        logger.info(f"Тип вопроса: {q_type}")
        log.info(f"Тип вопроса: {q_type}")
        
        # Используем переданные chunks ИЛИ ищем сами
        if chunks is None:
            if not self.searcher:
                logger.error("Нет поискового движка и не переданы chunks")
                log.error("Нет searcher и не переданы chunks")
                return {
                    "answer": "Ошибка конфигурации: отсутствует поисковый движок.",
                    "citations": [],
                    "error": True,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Импортируем здесь чтобы избежать циклической зависимости
            logger.info("Выполняем поиск через переданный searcher")
            # ВАЖНО: для поиска добавляем только 'item' (подсказку),
            #        но НЕ меняем промпт LLM.
            search_q = f"{question} {okpd_item}" if okpd_item else question
            # 1-й прогон: обычный гибридный поиск
            chunks = self.searcher.search(search_q, top_k=top_k, context_window=1)
            # Smart-retry: если нашли мало — делаем универсальные расширения и повторяем
            if SMART_RETRY_ENABLE and (not chunks or len(chunks) < SMART_RETRY_MIN_HITS):
                q_norm = normalize_for_retrieval(search_q)
                exp = build_multiquery_expansions(q_norm)
                retry = retrieve_and_rerank(self.searcher, exp, topk_per_query=SMART_RETRY_TOPK_PER_QUERY)
                if retry:
                    chunks = retry[:top_k]
        else:
            logger.info(f"Используем переданные chunks: {len(chunks)} фрагментов")
            log.debug(f"Используем переданные chunks: {len(chunks)}")

        if not chunks:
            log.warning("Пустые chunks — возвращаем not-found")
            return {
                "answer": "К сожалению, по вашему запросу не найдено релевантной информации в базе знаний.",
                "citations": [],
                "question_type": q_type,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Формируем улучшенный контекст
        context = self.enhance_context(chunks, question)
        if len(context) > MAX_CONTEXT_CHARS:
            log.info(f"Context {len(context)} > budget {MAX_CONTEXT_CHARS}, truncate")
            logger.info("Контекст %d chars > budget %d, подрезаем", len(context), MAX_CONTEXT_CHARS)
            context = context[:MAX_CONTEXT_CHARS]
        log.debug(f"context_size={len(context)}")

        # Выбираем системный промпт
        system_prompt = (
            self.templates.SYSTEM_ANALYTICAL 
            if q_type in ['explain', 'compare'] 
            else self.templates.SYSTEM_EXPERT
        )
        
        if use_cot:
            system_prompt += "\n\n" + self.templates.FEW_SHOT_EXAMPLES
        
        # Строим промпт (БЕЗ подсказки OKPD!)
        user_prompt = self.build_enhanced_prompt(question, context, q_type)
        
        temp_map = {
            'reference': 0.3,
            'compare': 0.4,
            'howto': 0.4,
            'troubleshoot': 0.4,
            'explain': 0.5,
            'general': 0.45
        }
        temperature = temp_map.get(q_type, 0.45)
        
        # Формируем запрос к LLM с параметром keep_alive
        payload = {
            "model": MODEL,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
                "num_predict": NUM_PREDICT,
                "seed": 42
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        log.info(f"LLM request: model={MODEL}, num_ctx={NUM_CTX}, num_predict={NUM_PREDICT}, "
             f"temperature={temperature}, keep_alive={KEEP_ALIVE}")
        log.debug(f"messages={len(payload['messages'])}, context_chars={len(context)}, question_chars={len(question)}")

        # Добавляем историю разговора для контекста
        with self._lock:
            hist = list(self.conversation_history[-4:])
        if hist:
            for msg in hist:
                payload["messages"].insert(1, msg)
        
        try:
            # Запрос к Ollama
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            log.info(f"LLM status={response.status_code}, elapsed≈{response.elapsed.total_seconds():.3f}s")
            if response.status_code >= 400:
                log.error(f"LLM error body (first 500): {response.text[:500]}")
            response.raise_for_status()
            
            data = response.json()
            raw_answer = (
                data.get("message", {}).get("content")
                or data.get("response")
                or ""
            )
            
            # Постобработка ответа
            with timed(log, "post_process_answer"):
                final_answer = self.post_process_answer(raw_answer, chunks)
            log.debug(f"final_answer_len={len(final_answer)}")
            
            # Добавим ненавязчивую приписку с подсказкой, если LLM сам не назвал этот код
            if okpd_hint:
                code_str = str(okpd_hint.get("code", "") or "")
                if code_str and code_str not in final_answer:
                    desc = (okpd_hint.get('desc','') or '').replace('\n', ' ')[:180]
                    if okpd_hint and okpd_hint["conf"] >= OKPD_HINT_MIN_CONF:
                        tip = (
                            f"\n\n> ℹ️ Подсказка классификатора ОКПД2: "
                            f"**{code_str}** ({okpd_hint.get('conf', 0.0):.0%}) — {desc}."
                        )
                        final_answer = final_answer + tip
            
            # Проверка grounding
            with timed(log, "check_answer_grounding"):
                grounding_score = self.check_answer_grounding(final_answer, chunks)
            log.info(f"Grounding score: {grounding_score:.2%}")
            logger.info(f"Grounding score: {grounding_score:.2%}")

            # Второй шанс: если grounding слабый — расширяем поиск и пере-спрашиваем LLM
            if SMART_RETRY_ENABLE and grounding_score < SMART_RETRY_GROUNDING_THRESHOLD and self.searcher:
                context_used = context
                q_norm = normalize_for_retrieval(question + (" " + (okpd_item or "")))
                exp = build_multiquery_expansions(q_norm)
                retry = retrieve_and_rerank(self.searcher, exp, topk_per_query=SMART_RETRY_TOPK_PER_QUERY)
                if retry:
                    context2 = self.enhance_context(retry[:top_k], question)
                    if len(context2) > MAX_CONTEXT_CHARS:
                        context2 = context2[:MAX_CONTEXT_CHARS]
                    user_prompt2 = self.build_enhanced_prompt(question, context2, q_type)
                    payload["messages"][-1]["content"] = user_prompt2
                    try:
                        response2 = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
                        response2.raise_for_status()
                        data2 = response2.json()
                        raw_answer2 = data2.get("message", {}).get("content") or data2.get("response") or ""
                        final_answer2 = self.post_process_answer(raw_answer2, retry)
                        grounding_score2 = self.check_answer_grounding(final_answer2, retry)
                        log.info(f"Grounding (retry): {grounding_score2:.2%}")
                        if grounding_score2 > grounding_score:
                            final_answer, grounding_score, chunks = final_answer2, grounding_score2, retry
                            context_used = context2
                    except Exception as e:
                        log.warning(f"LLM retry error: {e}")
                
                # Сохраняем в историю (потокобезопасно)
                with self._lock:
                    self.conversation_history.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": final_answer}
                    ])
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]

                try:
                    context_len = len(context_used)
                except NameError:
                    context_len = len(context)

                # Формируем цитаты
                citations = []
                seen = set()
                def _is_parent(c: Dict) -> bool:
                    return (c.get('is_parent') is True) or ((c.get('meta') or {}).get('is_parent') is True)

                # Берём чуть больший пул и сортируем: сначала дети, потом родители
                pool = sorted(chunks[:MAX_CITATIONS * 2], key=lambda x: _is_parent(x))

                for c in pool:
                    key = (c.get("book", ""), c.get("section", ""), c.get("page", 0))
                    # если ключ уже занят ребёнком — родителя пропускаем
                    if key in seen and _is_parent(c):
                        continue
                    if key not in seen:
                        citations.append({
                            "book": c.get("book", "Unknown"),
                            "section": c.get("section", "Unknown"),
                            "page": c.get("page", 0),
                            "relevance": c.get("score", 0),
                            "is_parent": _is_parent(c),   # опционально — удобно для UI бейджа «секция»
                        })
                        seen.add(key)
                    if len(citations) >= MAX_CITATIONS:
                        break
                
                result = {
                    "answer": final_answer,
                    "citations": citations,
                    "question_type": q_type,
                    "chunks_found": len(chunks),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "model": MODEL,
                    "context_size": context_len,
                    "grounding_score": grounding_score,
                }
                if okpd_hint:
                    result["meta"] = {"okpd_hint": okpd_hint}
                return result
            
        except requests.exceptions.ConnectionError:
            logger.error("Не удалось подключиться к Ollama")
            # Fallback на простой ответ
            return ask_simple(question, chunks=chunks)
            
        except requests.exceptions.Timeout:
            logger.error("Timeout при запросе к Ollama")
            # Fallback на простой ответ
            return ask_simple(question, chunks=chunks)
            
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return {
                "answer": f"Произошла ошибка при генерации ответа: {str(e)}",
                "citations": [],
                "error": True,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # --- УСПЕШНЫЙ ПУТЬ (без исключений): сохраняем историю, готовим цитаты, возвращаем результат ---
        with self._lock:
            self.conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_answer}
            ])
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

        citations = []
        seen = set()
        def _is_parent(c: Dict) -> bool:
            return (c.get('is_parent') is True) or ((c.get('meta') or {}).get('is_parent') is True)
        pool = sorted(chunks[:MAX_CITATIONS * 2], key=lambda x: _is_parent(x))
        for c in pool:
            key = (c.get("book", ""), c.get("section", ""), c.get("page", 0))
            if key in seen and _is_parent(c):
                continue
            if key not in seen:
                citations.append({
                    "book": c.get("book", "Unknown"),
                    "section": c.get("section", "Unknown"),
                    "page": c.get("page", 0),
                    "relevance": c.get("score", 0),
                    "is_parent": _is_parent(c),
                })
                seen.add(key)
            if len(citations) >= MAX_CITATIONS:
                break

        result = {
            "answer": final_answer,
            "citations": citations,
            "question_type": q_type,
            "chunks_found": len(chunks),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "model": MODEL,
            "context_size": len(context),
            "grounding_score": grounding_score,
        }
        if okpd_hint:
            result["meta"] = {"okpd_hint": okpd_hint}
        return result
    
    def clear_history(self):
        """Очистка истории разговора"""
        with self._lock:
            self.conversation_history = []
            self.context_cache = {}

# ---------------------- Smart-Retry helpers ----------------------
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
_MULT = "×"

def normalize_for_retrieval(q: str) -> str:
    """Нормализуем запрос для ретривера: тире, разделители размеров, лишние пробелы, кейс."""
    if not q:
        return ""
    s = q
    # унифицируем тире
    s = re.sub(f"[{_DASHES}]", "-", s)
    # объединяем возможные «x/х/×/*» между числами
    s = re.sub(r'(?<=\d)\s*[xх'+_MULT+r'*]\s*(?=\d)', 'x', s, flags=re.I)
    # убираем двойные пробелы
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_multiquery_expansions(q_norm: str) -> List[str]:
    """Генерим универсальные версии запроса (ГОСТ, размеры, материалы, синонимы форм)."""
    out = [q_norm]
    base = q_norm
    # 1) слепленные/дефисные варианты (ПОСК 50-18 → ПОСК50-18 / ПОСК-50-18)
    base = base.replace("Ё","E").replace("ё","e")
    out.append(re.sub(r'\s*-\s*', '-', base))
    out.append(re.sub(r'(?<=\b[А-ЯA-Z]{2,})\s+(?=\d)', '', base))
    # 2) варианты лат/кир для марок сталей (12Х18Н10Т ↔ 12X18H10T)
    xlat = (base
            .replace("Х","X").replace("х","x")
            .replace("Н","H").replace("н","h")
            .replace("С","C").replace("с","c"))
    if xlat != base: out.append(xlat)
    # 3) синонимы форм проката
    synonyms = [
        ("штрипс", "лента"), ("лента", "штрипс"),
        ("лист", "плита"), ("плита", "лист"),
        ("пруток", "круг"), ("круг", "пруток"),
        ("проволока", "катанка"), ("катанка", "проволока"),
    ]
    for a,b in synonyms:
        if re.search(rf'\b{a}\b', base, re.I):
            out.append(re.sub(rf'\b{a}\b', b, base, flags=re.I))
    # 4) ГОСТ-якоря (если в тексте есть номер — создаём «ГОСТ NNNN» и «GOST NNNN»)
    for m in re.findall(r'\b(ГОСТ|GOST)?\s*([0-9]{3,6})(?:-\d{2,4})?\b', base, flags=re.I):
        num = m[1]
        out += [f"ГОСТ {num}", f"GOST {num}"]
    # 5) жёсткая склейка размеров (0.8 x 90 → 0.8x90)
    out.append(re.sub(r'\s*x\s*', 'x', base, flags=re.I))
    # de-dup
    seen, uniq = set(), []
    for q in out:
        qn = q.strip()
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            uniq.append(qn)
    return uniq[:12]

def retrieve_and_rerank(searcher, queries: List[str], *, topk_per_query: int = 5) -> List[Dict]:
    """Гибридный повторный поиск: несколько запросов → объединение → переранжировка."""
    if not queries:
        return []
    bucket: Dict[str, Dict] = {}
    for q in queries:
        try:
            res = searcher.search(q, top_k=topk_per_query, context_window=1) or []
        except Exception:
            res = []
        for c in res:
            # ключ: источник+страница+хэш текста
            raw = (c.get('book',''), c.get('section',''), c.get('page',0), c.get('id',None))
            key = "|".join(map(str, raw)) or hashlib.md5((c.get('text','') or '').encode('utf-8','ignore')).hexdigest()
            prev = bucket.get(key)
            if not prev or float(c.get('score',0)) > float(prev.get('score',0)):
                bucket[key] = c
    merged = list(bucket.values())
    merged.sort(key=lambda x: float(x.get('score',0)), reverse=True)
    return merged

# --- Глобальные синглтоны ---
_searcher_instance: Optional['AdvancedHybridSearch'] = None
_chats: Dict[str, EnhancedRAGChat] = {}
_chats_last_used: Dict[str, float] = {}
_CHATS_LOCK = threading.RLock()
_CHATS_MAX = int(os.getenv("RAG_MAX_CONV", "200"))        # лимит активных сессий
_CHATS_TTL_SEC = int(os.getenv("RAG_CONV_TTL_SEC", "3600"))  # авто-очистка неактивных (сек)

def _get_searcher_singleton() -> 'AdvancedHybridSearch':
    global _searcher_instance
    if _searcher_instance is None:
        from search_windows import get_searcher
        _searcher_instance = get_searcher()
    return _searcher_instance

def _prune_chats(now: Optional[float] = None) -> None:
    """Оппортунистическая чистка: по TTL и по размеру."""
    if now is None:
        now = time.time()
    to_drop = []
    # TTL
    for cid, ts in list(_chats_last_used.items()):
        if now - ts > _CHATS_TTL_SEC:
            to_drop.append(cid)
    # по размеру
    if len(_chats) - len(to_drop) > _CHATS_MAX:
        # выбросим самые старые сначала
        survivors = {cid: _chats_last_used[cid] for cid in _chats if cid not in to_drop}
        overflow = len(_chats) - _CHATS_MAX - len(to_drop)
        if overflow > 0:
            for cid, _ in sorted(survivors.items(), key=lambda kv: kv[1])[:overflow]:
                to_drop.append(cid)
    # применяем
    for cid in to_drop:
        _chats.pop(cid, None)
        _chats_last_used.pop(cid, None)

def _get_chat(conversation_id: Optional[str]) -> EnhancedRAGChat:
    """Возвращает чат для сессии. Если conversation_id не задан — одноразовый чат без сохранения."""
    searcher = _get_searcher_singleton()
    if conversation_id is None or str(conversation_id).strip() == "":
        return EnhancedRAGChat(searcher)  # ephemeral: не кэшируем историю между вызовами

    cid = str(conversation_id)[:128]
    now = time.time()
    with _CHATS_LOCK:
        chat = _chats.get(cid)
        if chat is None:
            chat = EnhancedRAGChat(searcher)
            _chats[cid] = chat
        _chats_last_used[cid] = now
        _prune_chats(now)
        return chat

def ask(
    question: str,
    top_k: int = 10,
    chunks: Optional[List[Dict]] = None,
    *,
    conversation_id: Optional[str] = None,
    qa_id: Optional[str] = None
) -> Dict:
    """Основной вход: чат привязывается к conversation_id. Без id — одноразовый чат."""
    chat = _get_chat(conversation_id)
    effective_top_k = top_k if top_k is not None else MAX_FRAGMENTS
    return chat.generate_answer(
        question,
        effective_top_k,
        chunks=chunks,
        qa_id=qa_id,
        conversation_id=conversation_id,
    )

def clear_conversation(conversation_id: str) -> bool:
    """Стереть историю конкретной сессии (если есть)."""
    with _CHATS_LOCK:
        conversation_id = str(conversation_id)[:128]
        chat = _chats.get(conversation_id)
        if chat:
            chat.clear_history()
            _chats_last_used[conversation_id] = time.time()
            return True
        return False

def drop_conversation(conversation_id: str) -> bool:
    """Полностью удалить сессию из кэша."""
    conversation_id = str(conversation_id)[:128]
    with _CHATS_LOCK:
        existed = _chats.pop(conversation_id, None) is not None
        _chats_last_used.pop(conversation_id, None)
        return existed

def ask_simple(question: str, top_k: int = 5, chunks: Optional[List[Dict]] = None, qa_id: Optional[str] = None, conversation_id: Optional[str] = None) -> Dict:
    """Упрощенная версия без LLM - просто возвращает найденные фрагменты"""
    log = CtxLog(logger, {"qa_id": qa_id, "cid": conversation_id})
    log.debug(f"ask_simple start: top_k={top_k}, chunks_in={len(chunks) if chunks else 0}")

    # Если chunks не переданы, нужно получить их
    if chunks is None:
        from search_windows import search
        chunks = search(question, top=top_k)
    log.debug(f"ask_simple chunks={len(chunks) if chunks else 0}")
    
    if not chunks:
        return {
            "answer": "По вашему запросу ничего не найдено.",
            "citations": [],
            "grounding_score": 0.0
        }
    
    # Формируем структурированный ответ из фрагментов
    answer = "## Найденные релевантные фрагменты:\n\n"
    
    for i, chunk in enumerate(chunks[:top_k], 1):
        answer += f"### {i}. {chunk.get('book', 'Unknown')} - {chunk.get('section', 'Unknown')}\n"
        answer += f"*Страница {chunk.get('page', 0)} | Релевантность: {chunk.get('score', 0):.2%}*\n\n"
        
        # Добавляем превью текста
        text = chunk.get('text', '')
        text_preview = text[:500]
        if len(text) > 500:
            text_preview += "..."
        
        answer += f"{text_preview}\n\n"
        answer += "---\n\n"
    
    citations = [
        {
            "book": c.get("book", "Unknown"),
            "section": c.get("section", "Unknown"), 
            "page": c.get("page", 0),
            "relevance": c.get("score", 0)
        }
        for c in chunks[:top_k]
    ]
    
    return {
        "answer": answer,
        "citations": citations,
        "chunks_found": len(chunks),
        "grounding_score": 1.0  # Простой ответ всегда основан на источниках
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Улучшенная система вопрос-ответ')
    parser.add_argument('question', nargs='*', help='Ваш вопрос')
    parser.add_argument('--simple', action='store_true', help='Простой режим без LLM')
    parser.add_argument('--top-k', type=int, default=MAX_FRAGMENTS, help='Количество фрагментов')
    parser.add_argument('--no-cot', action='store_true', help='Отключить Chain of Thought')
    parser.add_argument('--clear-history', action='store_true', help='Очистить историю')
    
    args = parser.parse_args()
    
    # Для CLI версии создаем chat с searcher
    from search_windows import AdvancedHybridSearch, get_searcher
    searcher = get_searcher()
    chat = EnhancedRAGChat(searcher)
    
    if args.clear_history:
        chat.clear_history()
        print("История разговора очищена.\n")
    
    # Интерактивный режим
    if not args.question:
        print("=" * 80)
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ RAG CHAT")
        print("=" * 80)
        print("Введите 'exit' для выхода, 'clear' для очистки истории\n")
        
        while True:
            try:
                q = input("\n📝 Ваш вопрос: ").strip()
                
                if q.lower() == 'exit':
                    break
                elif q.lower() == 'clear':
                    chat.clear_history()
                    print("✓ История очищена")
                    continue
                elif not q:
                    continue
                
                print("\n⏳ Обработка запроса...\n")
                
                if args.simple:
                    result = ask_simple(q, top_k=args.top_k)
                else:
                    result = chat.generate_answer(
                        q, 
                        top_k=args.top_k,
                        use_cot=not args.no_cot
                    )
                
                # Выводим результат
                print("=" * 80)
                print("💡 ОТВЕТ:")
                print("=" * 80)
                print(result["answer"])
                
                if result.get("citations"):
                    print("\n" + "=" * 80)
                    print("📚 ИСТОЧНИКИ:")
                    print("=" * 80)
                    for i, cit in enumerate(result["citations"], 1):
                        relevance = cit.get('relevance', 0)
                        print(f"{i}. {cit['book']} - {cit['section']} (стр. {cit['page']}) [{relevance:.1%}]")
                
                # Статистика
                if 'processing_time' in result:
                    print(f"\n⚡ Время обработки: {result['processing_time']:.2f} сек")
                if 'grounding_score' in result:
                    print(f"✅ Подтверждение источниками: {result['grounding_score']:.1%}")
                if 'question_type' in result:
                    print(f"🏷️ Тип вопроса: {result['question_type']}")
                if 'chunks_found' in result:
                    print(f"📊 Найдено фрагментов: {result['chunks_found']}")
                
                # OKPD hint если есть
                if result.get("meta", {}).get("okpd_hint"):
                    hint = result["meta"]["okpd_hint"]
                    print(f"🔍 OKPD классификатор: {hint.get('code')} ({hint.get('conf', 0):.0%})")
                    
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем.")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                continue
    
    else:
        # Одиночный запрос
        q = " ".join(args.question)
        
        print(f"\n📝 Вопрос: {q}\n")
        print("⏳ Генерация ответа...\n")
        
        if args.simple:
            result = ask_simple(q, top_k=args.top_k)
        else:
            result = chat.generate_answer(
                q,
                top_k=args.top_k,
                use_cot=not args.no_cot
            )
        
        print("=" * 80)
        print("💡 ОТВЕТ:")
        print("=" * 80)
        print(result["answer"])
        
        if result.get("citations"):
            print("\n" + "=" * 80)
            print("📚 ИСТОЧНИКИ:")
            print("=" * 80)
            for i, cit in enumerate(result["citations"], 1):
                print(f"{i}. {cit['book']} - {cit['section']} (стр. {cit['page']})")
        
        # OKPD hint если есть
        if result.get("meta", {}).get("okpd_hint"):
            hint = result["meta"]["okpd_hint"]
            print(f"\n🔍 OKPD классификатор: {hint.get('code')} ({hint.get('conf', 0):.0%})")