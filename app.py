# C:\F-ChatAI\app.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import defaultdict
import numpy as np
import os
from urllib.parse import unquote
# Импорт наших модулей
from search_windows import AdvancedHybridSearch
from ask_rag_chat import EnhancedRAGChat

from psycopg2.pool import SimpleConnectionPool
import re

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAVE_RF = True
except Exception:
    rf_process = rf_fuzz = None
    _HAVE_RF = False

MIN_CONFIDENCE = float(os.getenv("RAG_MIN_CONFIDENCE", "0.35"))  # Минимальный порог уверенности модели в ответе
MIN_GROUNDING  = float(os.getenv("RAG_MIN_GROUNDING", "0.35"))   # Минимальный порог привязки ответа к источникам

SMALLTALK_RE = re.compile(r'^\s*(привет|здравств|добро(е|й)|hi|hello|hey)\b', re.I)
RELTIME_RE = re.compile(
    r'\b('
    r'погод[ауы]?|температур|дожд|снег|ветер|облачн|солнечн|гроза|'
    r'курс|доллар|евро|биткоин|акци[ия]|'
    r'новост|пробк|трафик|'
    r'врем[яени]|который\s+час|часовой\s+пояс|'
    r'расписани[ея]|рейс|самол[её]т|поезд'
    r')\b',
    re.I
)

REL_TIME_WORDS = re.compile(
    r'\b('
    r'сегодня|завтра|послезавтра|вчера|'
    r'на\s+недел\w*|на\s+выходн\w*|'
    r'через\s+\d+\s*(?:минут[ауы]?|час(?:а|ов)?|дн(?:я|ей)?)'
    r')\b',
    re.I
)

SENSITIVE_RE = re.compile(
    r'(?P<adult_hard>\b(?:инцест\w*|лоли|loli|детск\w*\s*порн\w*|(?<!\w)cp(?!\w)|зоофил\w*|bestialit\w*)\b)'
    r'|(?P<adult>\b(?:18\+|nsfw|xxx|порно|порн\w*|эротик\w*|'
    r'гей|лесби|gay|'
    r'секс\w*|интим\w*|нюдс?|nudes?|nude|'
    r'голы[йея]|обнаж\w*|раздев\w*|фетиш\w*|'
    r'минет|оральн\w*|куни\w*|анальн\w*|вагин\w*|'
    r'проститут\w*|эскорт\w*|бордель\w*|шлюх\w*)\b)'
    r'|(?P<clinical>\b(?:медицин\w*|диагноз\w*|лечен\w*|'
    r'психиатр\w*|уролог\w*|андролог\w*|гинеколог\w*|'
    r'половы\w*|генитал\w*|эректил\w*|эякуляц\w*|сперм\w*)\b)',
    re.I
)

class TrainingPayload(BaseModel):
    title: Optional[str] = None
    text: str
    category: Optional[str] = None
    type: Optional[str] = None

class InaccuracyFeedback(BaseModel):
    inaccuracy: str
    correct_info: Optional[str] = None
    type: Optional[str] = None




def detect_sensitive_reason(q: str) -> str | None:
    if not q: 
        return None
    m = SENSITIVE_RE.search(q)
    if not m:
        return None
    # вернём имя сработавшей группы: adult_hard / adult / clinical
    for k, v in m.groupdict().items():
        if v:
            return k
    return None

# Создаем пул соединений для аналитики
DB_POOL = SimpleConnectionPool(
    1, 10,  # min и max соединений
    dbname="rag",
    user="rag", 
    password="rag",
    host="127.0.0.1",
    port=5432,
    options="-c client_encoding=UTF8"
)

# Инициализируем улучшенную аналитику вместо старой
from enhanced_analytics import EnhancedAnalytics


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Falcon AI Assistant",
    description="Интеллектуальный помощник по документации Falcon",
    version="3.1.0"
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(
        Path(__file__).with_name("favicon.ico"),
        media_type="image/x-icon",
        headers={"Cache-Control": "no-cache, max-age=0"}
    )

# единый путь к медиа (и для Windows, и для Linux через переменную окружения)
MEDIA_ROOT = Path(os.getenv(
    "MEDIA_ROOT",
    r"C:\\F-ChatAI\\media" if os.name == "nt" else "/opt/f-chatai/media"
))
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)  # на всякий случай создадим каталог
# статическая раздача картинок, на которые ссылается Markdown: /media/...
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")

DOCS_ROOT = Path(os.getenv(
    "DOCS_ROOT",
    r"C:\\F-ChatAI\\documents" if os.name == "nt" else "/opt/f-chatai/documents"
))
DOCS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/docs", StaticFiles(directory=str(DOCS_ROOT)), name="docs")
_BOOK_PATH_CACHE = {}

JS_ROOT = Path(os.getenv(
    "JS_ROOT",
    r"C:\\F-ChatAI\\js" if os.name == "nt" else "/opt/f-chatai/js"
))
app.mount("/js", StaticFiles(directory=str(JS_ROOT)), name="js")


# CORS для веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/training")
async def api_training(payload: TrainingPayload, request: Request):
    user = get_user_identifiers(request)
    conn = DB_POOL.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_actions
                    (action_type, session_id, user_domain, computer_name, metadata, created_at)
                    VALUES ('training_submit', %s, %s, %s, %s, NOW())
                """, (
                    user.get("computer_name"),
                    user.get("domain"),
                    user.get("computer_name"),
                    json.dumps(payload.dict(), ensure_ascii=False)
                ))
        return {"status": "success", "message": "Материалы получены"}
    finally:
        DB_POOL.putconn(conn)


@app.post("/api/feedback")
async def api_feedback(payload: InaccuracyFeedback, request: Request):
    user = get_user_identifiers(request)
    conn = DB_POOL.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_actions
                    (action_type, session_id, user_domain, computer_name, metadata, created_at)
                    VALUES ('feedback_inaccuracy', %s, %s, %s, %s, NOW())
                """, (
                    user.get("computer_name"),
                    user.get("domain"),
                    user.get("computer_name"),
                    json.dumps(payload.dict(), ensure_ascii=False)
                ))
        return {"status": "success", "message": "Спасибо за обратную связь!"}
    finally:
        DB_POOL.putconn(conn)


@app.post("/api/upload")
async def api_upload(
    request: Request,
    file: UploadFile = File(...),
    comment: Optional[str] = Form(None),
):
    user = get_user_identifiers(request)

    # сохраняем файл в DOCS_ROOT
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", (file.filename or "upload"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"{ts}_{safe_name}"
    dest_path = DOCS_ROOT / final_name
    content = await file.read()
    with dest_path.open("wb") as f:
        f.write(content)

    # логируем действие
    conn = DB_POOL.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_actions
                    (action_type, session_id, user_domain, computer_name, metadata, created_at)
                    VALUES ('file_upload', %s, %s, %s, %s, NOW())
                """, (
                    user.get("computer_name"),
                    user.get("domain"),
                    user.get("computer_name"),
                    json.dumps({"filename": final_name, "comment": comment}, ensure_ascii=False)
                ))
        return {
            "status": "success",
            "message": "Файл загружен",
            "filename": final_name,
            "url": f"/docs/{final_name}",
        }
    finally:
        DB_POOL.putconn(conn)


@app.post("/api/clear-chat")
async def clear_chat_history(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Скрытие истории чата (установка isHide=true)"""
    try:
        # Получаем данные запроса
        data = await request.json()
        session_id = data.get('session_id')
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")
        
        user_info = get_user_identifiers(request)
        
        # Обновляем записи в базе данных
        conn = DB_POOL.getconn()
        try:
            with conn.cursor() as cur:
                # Скрываем сообщения
                cur.execute("""
                    UPDATE qa_logs
                    SET isHide = TRUE
                    WHERE session_id = %s
                    AND (isHide IS NULL OR isHide = FALSE)
                """, (session_id,))
                hidden_count = cur.rowcount  # ← сколько строк обновлено

                # Логируем действие
                cur.execute("""
                    INSERT INTO user_actions
                        (action_type, session_id, user_domain, computer_name, metadata, created_at)
                    VALUES
                        ('clear_chat', %s, %s, %s, %s, NOW())
                """, (
                    session_id,
                    user_info['domain'],
                    user_info['computer_name'],
                    json.dumps({'hidden_count': hidden_count})
                ))

                conn.commit()

                
                logger.info(f"Chat history cleared for session {session_id}: {hidden_count} messages hidden")
                
                # Очищаем сессию в памяти
                if session_id in session_manager.sessions:
                    session_manager.clear(session_id)
                
                return {
                    "status": "success",
                    "message": f"История чата очищена ({hidden_count} сообщений)",
                    "hidden_count": hidden_count
                }
                
        finally:
            DB_POOL.putconn(conn)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/history")
async def user_history(request: Request, limit: int = 50, include_hidden: bool = False):
    """Получение истории пользователя (по умолчанию без скрытых)"""
    user = get_user_identifiers(request)
    
    conn = DB_POOL.getconn()
    try:
        with conn.cursor() as cur:
            # Добавляем фильтр по isHide
            hide_filter = "" if include_hidden else "AND (isHide IS NULL OR isHide = false)"
            
            query = f"""
                SELECT 
                    id,
                    question,
                    answer,
                    created_at as timestamp,
                    confidence_score as confidence,
                    grounding_score,
                    question_type,
                    feedback_rating as rating,
                    citations,
                    chunks,
                    processing_time,
                    isHide
                FROM qa_logs
                WHERE session_id = %s
                {hide_filter}
                ORDER BY created_at DESC
                LIMIT %s
            """
            
            # Используем computer_name как session_id
            cur.execute(query, (user['computer_name'], limit))
            
            rows = cur.fetchall()
            history = []
            
            for row in rows:
                history.append({
                    'id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'timestamp': row[3].isoformat() if row[3] else None,
                    'confidence': row[4],
                    'grounding_score': row[5],
                    'question_type': row[6],
                    'rating': row[7],
                    'citations': row[8] if row[8] else [],
                    'chunks': row[9] if row[9] else [],
                    'processing_time': row[10],
                    'is_hidden': row[11] if row[11] is not None else False
                })
            
            return {
                "user": user,
                "history": history,
                "total": len(history),
                "include_hidden": include_hidden
            }
            
    finally:
        DB_POOL.putconn(conn)

# Обновляем метод в EnhancedAnalytics для учета isHide
class EnhancedAnalytics(EnhancedAnalytics):  # Наследуем от существующего класса
    
    def get_user_history(self, domain: str, computer_name: str, limit: int = 50, include_hidden: bool = False):
        """Получение истории с учетом скрытых записей"""
        conn = DB_POOL.getconn()
        try:
            with conn.cursor() as cur:
                hide_filter = "" if include_hidden else "AND (isHide IS NULL OR isHide = false)"
                
                # Используем session_id вместо domain/computer_name если они совпадают
                cur.execute(f"""
                    SELECT 
                        question,
                        answer,
                        created_at,
                        confidence_score,
                        grounding_score,
                        feedback_rating,
                        citations
                    FROM qa_logs
                    WHERE session_id = %s
                    {hide_filter}
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (computer_name, limit))
                
                rows = cur.fetchall()
                return [{
                    'question': row[0],
                    'answer': row[1],
                    'timestamp': row[2].isoformat() if row[2] else None,
                    'confidence': row[3],
                    'grounding_score': row[4],
                    'rating': row[5],
                    'citations': row[6] if row[6] else []
                } for row in rows]
                
        finally:
            DB_POOL.putconn(conn)

@app.get("/resolve")
def resolve_doc(book: str, page: Optional[int] = None):
    # обезопасим имя (никаких .. и слэшей)
    safe_book = Path(book).name

    # 1) кэш
    rel = _BOOK_PATH_CACHE.get(safe_book)
    if not rel or not (DOCS_ROOT / rel).exists():
        # 2) ищем на диске
        exts = [".pdf", ".docx", ".doc", ".md", ".txt", ".rtf"]
        candidates = []
        for ext in exts:
            for p in DOCS_ROOT.rglob(safe_book + ext):
                candidates.append(p)

        if not candidates:
            # фоллбек: попробуем точное имя файла из стема (если вдруг уже лежит рядом)
            p = DOCS_ROOT / f"{safe_book}.docx"
            if p.exists():
                candidates = [p]

        if not candidates:
            raise HTTPException(status_code=404, detail=f"Файл для '{safe_book}' не найден в documents")

        # предпочтем более «близкий» и с лучшим расширением
        def rank(p: Path):
            ext = p.suffix.lower()
            ext_rank = exts.index(ext) if ext in exts else 99
            depth = len(p.relative_to(DOCS_ROOT).parts)
            return (ext_rank, depth, str(p).lower())

        best = sorted(candidates, key=rank)[0]
        rel = best.relative_to(DOCS_ROOT).as_posix()
        _BOOK_PATH_CACHE[safe_book] = rel

    # 3) собираем конечный URL
    url = f"/docs/{rel}"
    if rel.lower().endswith(".pdf") and page and page > 0:
        url += f"#page={page}"
    return RedirectResponse(url, status_code=307)

@app.middleware("http")
async def human_access_log(request: Request, call_next):
    path = unquote(request.url.path)                         # /suggestions
    # query_params уже декодированы Starlette’ом
    if request.query_params:
        q = "&".join(f"{k}={v}" for k, v in request.query_params.multi_items())
        full = f"{path}?{q}"
    else:
        full = path
    logger.info('GET %s', full)                              # → Понятный текст
    resp = await call_next(request)
    logger.info('-> %d %s', resp.status_code, full)
    return resp

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Инициализация компонентов с передачей searcher
DB_DSN = "dbname=rag user=rag password=rag host=127.0.0.1 port=5432"
searcher = AdvancedHybridSearch(DB_DSN)
chat_engine = EnhancedRAGChat(searcher)  # Передаем searcher в chat!

# Кэш для оптимизации
class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            if datetime.now() - self.timestamps[key] < timedelta(seconds=self.ttl):
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

response_cache = ResponseCache(ttl_seconds=1800)
analytics = EnhancedAnalytics(DB_POOL)


# Модели данных с валидацией
class Question(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(10, ge=1, le=50)
    search_methods: Optional[List[str]] = ['vector', 'fulltext']
    use_llm: Optional[bool] = True
    use_cache: Optional[bool] = True
    conversation_id: Optional[str] = None
    use_cot: Optional[bool] = True

class Answer(BaseModel):
    answer: str
    citations: List[Dict]
    chunks: List[Dict]
    processing_time: float
    question_type: Optional[str] = None
    cached: bool = False
    confidence_score: Optional[float] = None
    grounding_score: Optional[float] = None
    qa_id: Optional[str] = None
    blocked: Optional[bool] = False

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: Optional[int] = Field(10, ge=1, le=100)
    min_score: Optional[float] = Field(0.3, ge=0.0, le=1.0)
    use_query_expansion: Optional[bool] = True
    context_window: Optional[int] = Field(0, ge=0, le=3)

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=1000)
    helpful: Optional[bool] = None
    tags: Optional[List[str]] = []
    qa_id: Optional[str] = None
    
# менеджер сессий с передачей searcher
class SessionManager:
    def __init__(self, searcher: AdvancedHybridSearch):
        self.sessions = {}
        self.searcher = searcher
        self.state = {}
    
    def get_or_create(self, session_id: str) -> EnhancedRAGChat:
        if session_id not in self.sessions:
            self.sessions[session_id] = EnhancedRAGChat(self.searcher)  # Передаем searcher!
        return self.sessions[session_id]
    
    def clear(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].clear_history()
        self.state.pop(session_id, None)

     # --- фиксация последней темы для авто-разрыва ---
    def note(self, session_id: str, q: str, chunks: list[dict]):
        """Запомнить токены вопроса и первую книгу (book) из найденных чанков."""
        try:
            qtok = _tokens(q)  # используем уже определённый выше helper
            main_book = chunks[0].get('book') if chunks else None
            self.state[session_id] = {'qtok': qtok, 'book': main_book, 'last_cut_at': None}
        except Exception:
            pass
 
    def is_topic_shift(self, session_id: str, q: str, chunks: list[dict]) -> bool:
        """
        Сильный дрейф темы = изменилась книга и слабое пересечение с прошлым вопросом.
        """
        prev = self.state.get(session_id)
        if not prev:
            return False
        qtok = _tokens(q)
        if not qtok:
            return False
        overlap = len(prev['qtok'] & qtok) / max(1, len(qtok))
        book_now = chunks[0].get('book') if chunks else None
        changed_book = bool(prev.get('book') and book_now and book_now != prev['book'])
        if not (changed_book and overlap < 0.25):
            return False
        # анти-дребезг: если только что уже рвали — пропустим в течение 60с
        last_cut = prev.get('last_cut_at')
        now = datetime.now()
        if last_cut and (now - last_cut).total_seconds() < 60:
            return False
        prev['last_cut_at'] = now
        return True

session_manager = SessionManager(searcher)  # Передаем searcher в менеджер

def calculate_confidence(chunks: List[Dict], answer: str, grounding_score: float = None) -> float:
    """Расчет уверенности с учетом grounding"""
    if not chunks:
        return 0.0
    
    avg_score = np.mean([c.get('score', 0) for c in chunks[:5]])
    top_score = chunks[0].get('score', 0) if chunks else 0
    num_good_chunks = sum(1 for c in chunks if c.get('score', 0) > 0.5)
    answer_length = len(answer)
    
    # Базовая уверенность
    confidence = (
        avg_score * 0.3 +
        top_score * 0.3 +
        min(num_good_chunks / 5, 1.0) * 0.2 +
        min(answer_length / 1000, 1.0) * 0.1
    )
    
    # Учитываем grounding если есть
    if grounding_score is not None:
        confidence = confidence * 0.7 + grounding_score * 0.3
    
    return min(confidence, 1.0)

@app.get("/")
async def root():
    """Главная страница с улучшенным веб-интерфейсом"""
    return HTMLResponse(content=ENHANCED_HTML_INTERFACE, status_code=200)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)?")

def is_smalltalk(q: str) -> bool:
    """
    Smalltalk, если:
      - пустая строка, или
      - сообщение начинается с приветствия и после него 0–1 слово.
    Иначе — не smalltalk.
    """
    q = (q or "").strip()
    if not q:
        return True

    m = SMALLTALK_RE.match(q)  # приветствие строго в начале
    if not m:
        return False  # нет приветствия → обычный запрос

    tail = q[m.end():].strip(" ,.!?-–—")
    tokens = WORD_RE.findall(tail)
    return len(tokens) <= 1  # 0–1 слово после приветствия → smalltalk (блок)

TOK_RE = re.compile(r'[а-яёa-z0-9]{3,}', re.I)

 # лёгкий стемминг + стоп-слова, как в is_in_kb_domain
STOPWORDS = {"про","о","об","обо","расскажи","расскажите","подскажи","подскажите",
              "покажи","покажите","пожалуйста","дай","дайте"}
RU_SUFFIXES = ("ами","ями","ов","ев","ёв","ей","ёй","ою","ею","ом","ем","ах","ях","ам","ям","у","ю","а","я","е","и","о","ы")
def _tokens(s: str) -> set[str]:
    raw = TOK_RE.findall((s or '').lower().replace('ё','е'))
    out = set()
    for t in raw:
        if t in STOPWORDS: 
            continue
        for suf in RU_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 4:
                t = t[:-len(suf)]
                break
        out.add(t)
    return out

def calc_min_score(q: str) -> float:
    # считаем «содержательные» токены (>=3 символов)
    ntok = len(_tokens(q))
    # короткие/шумные вопросы — поднимаем порог,
    # обычные — понижаем
    if ntok <= 2:
        return 0.35
    elif ntok <= 4:
        return 0.25
    else:
        return 0.15
    
def is_in_kb_domain(q: str, chunks: list[dict], *, debug: bool = True) -> bool:
    if not chunks:
        if debug: logger.info("domain: no chunks -> False")
        return False

    q = (q or "").strip()
    if RELTIME_RE.search(q) and REL_TIME_WORDS.search(q):
        if debug: logger.info("domain: realtime intent -> False")
        return False

    STOPWORDS = {"про","о","об","обо","расскажи","расскажите","подскажи","подскажите",
                 "покажи","покажите","пожалуйста","дай","дайте"}
    RU_SUFFIXES = ("ами","ями","ов","ев","ёв","ей","ёй","ою","ею","ом","ем","ах","ях","ам","ям","у","ю","а","я","е","и","о","ы")

    def _norm_token(t: str) -> str:
        t = t.lower().replace('ё','е')
        for suf in RU_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 4:
                return t[:-len(suf)]
        return t

    TOK_RE = re.compile(r'[а-яёa-z0-9]{3,}', re.I)
    def _xtokens(s: str) -> set[str]:
        raw = TOK_RE.findall((s or '').lower().replace('ё','е'))
        return {_norm_token(t) for t in raw if t not in STOPWORDS}

    qtok = _xtokens(q)
    ntok = len(qtok)
    top = chunks[:3]

    # --- если финальный ранжировщик уверен — пропускаем (фикс вашего кейса) ---
    # searcher.search() уже даёт финальный r["score"] ~ [0..1]
    final_scores = [float(c.get('score') or 0.0) for c in top]
    max_final = max(final_scores or [0.0])
    if max_final >= 0.75:
        if debug: logger.info("domain: strong final score %.2f -> True", max_final)
        return True

    # --- скоры по компонентам (как было) ---
    rr = [float(c.get('rerank_score')   or 0.0) for c in top]
    vs = [float(c.get('vector_score')   or 0.0) for c in top]
    fs = [float(c.get('fulltext_score') or 0.0) for c in top]

    max_rr = max(rr or [0.0]);  avg_rr = sum(rr)/max(1,len(rr))
    max_vs = max(vs or [0.0]);  avg_vs = sum(vs)/max(1,len(vs))
    avg_fs = sum(fs)/max(1,len(fs))

    def overlap(c) -> float:
        head = (c.get('section','') + ' ' + (c.get('text','') or '')[:400])
        head_tok = _xtokens(head)
        inter = len(qtok & head_tok)

        # Fuzzy-оверлап для коротких запросов (ловим «моркировки»≈«маркировки»)
        if inter == 0 and ntok <= 3 and _HAVE_RF and qtok and head_tok:
            dtok_list = list(head_tok)
            fuzzy_hits = 0
            for t in qtok:
                if t in head_tok:
                    fuzzy_hits += 1
                    continue
                if len(t) >= 4:
                    m = rf_process.extractOne(t, dtok_list, scorer=rf_fuzz.WRatio)
                    if m and m[1] >= 85:
                        fuzzy_hits += 1
            inter = max(inter, fuzzy_hits)

        return inter / max(1, len(qtok))

    ov_list = [overlap(c) for c in top]
    ov_max  = max(ov_list or [0.0])
    ov_avg  = sum(ov_list)/max(1,len(ov_list))

    # --- динамические пороги + голоса (слегка смягчены для коротких запросов) ---
    if ntok <= 2:
        rr_gate, vs_gate, fs_gate = 0.08, 0.48, 0.20   # rr_gate сильно ниже: у bge-reranker значения малы
        ov_gate_max, ov_gate_avg   = 0.20, 0.18
        min_common_tokens          = 1
        required_votes             = 1
    elif ntok <= 4:
        rr_gate, vs_gate, fs_gate = 0.12, 0.46, 0.22
        ov_gate_max, ov_gate_avg   = 0.28, 0.22
        min_common_tokens          = 1
        required_votes             = 2
    else:
        rr_gate, vs_gate, fs_gate = 0.15, 0.42, 0.20
        ov_gate_max, ov_gate_avg   = 0.30, 0.25
        min_common_tokens          = 1
        required_votes             = 2

    # базовая проверка токенов (с учётом fuzzy выше через overlap)
    tokens_ok = (ov_max >= 0.20) or (ov_avg >= 0.18)

    # «голосование»
    vote_rr = (max_rr >= rr_gate and (ov_max >= ov_gate_max or ntok <= 2))
    vote_vs = (max_vs >= vs_gate and (ov_max >= ov_gate_max or ntok <= 2))
    vote_fs = (avg_fs >= fs_gate and ov_avg >= ov_gate_avg)

    votes = int(vote_rr) + int(vote_vs) + int(vote_fs)
    ok = (tokens_ok and votes >= required_votes)

    if debug:
        logger.info(
            "domain: ntok=%d | rr max/avg=%.2f/%.2f vs max/avg=%.2f/%.2f fs avg=%.2f | "
            "ov max/avg=%.2f/%.2f | tokens_ok=%s votes=%d req=%d -> %s",
            ntok, max_rr, avg_rr, max_vs, avg_vs, avg_fs, ov_max, ov_avg,
            tokens_ok, votes, required_votes, ok
        )
    return ok


@app.post("/ask", response_model=Answer)
def ask_question(
    question: Question,
    background_tasks: BackgroundTasks,
    request: Request = None,
):
    """Основной эндпоинт с исправленной передачей chunks"""
    try:
        start_time = datetime.now()

        # Проверяем кэш с улучшенным ключом
        cache_key = hashlib.md5(
            json.dumps(
                {
                    "q": question.text,
                    "k": question.top_k,
                    "llm": question.use_llm,
                    "cot": question.use_cot,
                    "methods": sorted(question.search_methods)
                    if question.search_methods
                    else [],
                    "cid": question.conversation_id,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if question.use_cache:
            cached_response = response_cache.get(cache_key)
            if cached_response:
                cached_response["cached"] = True
                return cached_response

        # Получаем или создаем сессию
        # chat = (
        #     session_manager.get_or_create(question.conversation_id)
        #     if question.conversation_id
        #     else chat_engine
        # )

        if is_smalltalk(question.text):
            return Answer(
                answer="👋 Привет! Сформулируйте вопрос по базе знаний (инструкции Falcon) — я подберу релевантные фрагменты.",
                citations=[], chunks=[], question_type="smalltalk", blocked=True,
                cached=False, confidence_score=0.0, grounding_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
            )
        ms = calc_min_score(question.text)
        # Поиск релевантных фрагментов ОДИН РАЗ
        chunks = searcher.search(
            question.text,
            top_k=question.top_k,
            search_methods=question.search_methods,
            use_query_expansion=True,
            context_window=1,
            min_score=ms,    
        )

        if not is_in_kb_domain(question.text, chunks) or detect_sensitive_reason(question.text):
            return Answer(
                answer=("Я отвечаю по внутренней базе знаний (инструкции Falcon). "
                        "По этому запросу не нашёл надёжных соответствий. "
                        "УТОЧНИТЕ ФОРМУЛИРОВКУ или спросите про процесс/функцию в Falcon."),
                citations=[], chunks=[], question_type="out_of_domain", blocked=True,
                cached=False, confidence_score=0.0, grounding_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
            )
        
        # --- Авто-разрыв темы при смене книги и низком пересечении токенов ---
        if question.conversation_id:
            try:
                if session_manager.is_topic_shift(question.conversation_id, question.text, chunks):
                    session_manager.clear(question.conversation_id)
            except Exception:
                logger.debug("topic-shift check failed", exc_info=True)

        if not chunks:
            response = Answer(
                answer=(
                    "К сожалению, по вашему запросу ничего не найдено. Попробуйте:\n"
                    "• Использовать другие ключевые слова\n"
                    "• Проверить правописание\n"
                    "• Сформулировать вопрос иначе"
                ),
                citations=[],
                chunks=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                confidence_score=0.0,
                grounding_score=0.0,
            )

            user_info = get_user_identifiers(request)
            qa_id = hashlib.md5(
                f"{question.text}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            response.qa_id = qa_id

            log_record = {
                "qa_id": qa_id,
                "session_id": question.conversation_id,
                "question": question.text,
                "answer": response.answer,
                "citations": response.citations,
                "chunks": response.chunks,
                "processing_time": response.processing_time,
                "question_type": response.question_type,
                "cached": response.cached,
                "confidence_score": response.confidence_score,
                "grounding_score": response.grounding_score,
                "model": os.getenv("LLM_MODEL", "unknown"),
                "search_methods": question.search_methods,
                "top_k": question.top_k,
                "use_llm": question.use_llm,
                "use_cot": question.use_cot,
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "domain": user_info["domain"],
                "computer_name": user_info["computer_name"],
            }

            background_tasks.add_task(analytics.log_qa_complete, log_record)
            background_tasks.add_task(
                analytics.log_query,
                question.text,
                response.processing_time,
                len(chunks),
            )

            # Запоминаем текущую тему и выходим (даже если пусто — полезно сохранить токены)
            if question.conversation_id:
                try:
                    session_manager.note(question.conversation_id, question.text, chunks)
                except Exception:
                    pass
            return response
        
        # Берём (возможно очищенную) сессию
        chat = (
            session_manager.get_or_create(question.conversation_id)
            if question.conversation_id
            else chat_engine
        )

        # --- генерация ответа (LLM или simple) ---
        if question.use_llm:
            try:
                llm_response = chat.generate_answer(
                    question.text,
                    top_k=question.top_k,
                    use_cot=question.use_cot,
                    chunks=chunks,
                )
                answer_text = llm_response.get("answer", "")
                citations = llm_response.get("citations", [])
                q_type = llm_response.get("question_type", "general")
                grounding_score = llm_response.get("grounding_score")
            except Exception as e:
                logger.error("LLM failed: %s", e, exc_info=True)
                from ask_rag_chat import ask_simple
                simple = ask_simple(question.text, chunks=chunks)
                answer_text = ("⚠️ Модель ИИ недоступна, показываю выдержки из источников.\n\n"
                               + (simple.get("answer") or ""))
                citations = simple.get("citations", [])
                q_type = "llm_unavailable"
                grounding_score = 1.0
        else:
            from ask_rag_chat import ask_simple

            simple_response = ask_simple(question.text, chunks=chunks)
            answer_text = simple_response["answer"]
            citations = simple_response["citations"]
            q_type = "simple"
            grounding_score = 1.0  # простой режим основан на источниках

        # --- метрика уверенности ---
        confidence = calculate_confidence(chunks, answer_text, grounding_score)
        processing_time = (datetime.now() - start_time).total_seconds()

        # --- фоллбек по низкой уверенности/grounding ---
        is_low = (confidence < MIN_CONFIDENCE) or (
            grounding_score is not None and grounding_score < MIN_GROUNDING
        )
        blocked = False

        if question.use_llm and is_low:
            from ask_rag_chat import ask_simple

            simple = ask_simple(question.text, chunks=chunks)
            safe_header = (
                "⚠️ Я не нашёл достаточно надёжных оснований, чтобы уверенно отвечать.\n\n"
                "Ниже — релевантные фрагменты из базы знаний."
            )
            answer_text = safe_header + "\n\n" + (simple.get("answer") or "")
            citations = simple.get("citations", [])  # берём источники simple-выдачи
            q_type = "low_confidence"
            blocked = True

        # --- итоговый ответ ---
        response = Answer(
            answer=answer_text,
            citations=citations,
            chunks=chunks[:5],
            processing_time=processing_time,
            question_type=q_type,
            cached=False,
            confidence_score=confidence,
            grounding_score=grounding_score,
            blocked=blocked,
        )

        # идентификаторы + qa_id
        user_info = get_user_identifiers(request)
        qa_id = hashlib.md5(
            f"{question.text}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        response.qa_id = qa_id

        # --- логирование (финального ответа) ---
        log_record = {
            "qa_id": qa_id,
            "session_id": question.conversation_id,
            "question": question.text,
            "answer": response.answer,
            "citations": response.citations,
            "chunks": response.chunks,
            "processing_time": response.processing_time,
            "question_type": response.question_type,
            "cached": response.cached,
            "confidence_score": response.confidence_score,
            "grounding_score": response.grounding_score,
            "model": os.getenv("LLM_MODEL", "unknown"),
            "search_methods": question.search_methods,
            "top_k": question.top_k,
            "use_llm": question.use_llm,
            "use_cot": question.use_cot,
            "ip_address": request.client.host if request and request.client else None,
            "user_agent": request.headers.get("user-agent") if request else None,
            "domain": user_info["domain"],
            "computer_name": user_info["computer_name"],
        }

        background_tasks.add_task(analytics.log_qa_complete, log_record)
        background_tasks.add_task(
            analytics.log_query,
            question.text,
            response.processing_time,
            len(chunks),
        )

        # Запоминаем текущую тему для следующего запроса
        if question.conversation_id:
            try:
                session_manager.note(question.conversation_id, question.text, chunks)
            except Exception:
                pass

        # --- кэш только для хороших ответов ---
        if (
            question.use_cache
            and (not response.blocked)
            and response.confidence_score > 0.7
            and (
                response.grounding_score is None
                or response.grounding_score > 0.8
            )
        ):
            response_cache.set(cache_key, response.dict())

        return response

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_user_identifiers(request: Optional[Request]) -> Dict:
    if request is None:
        return {"domain": "unknown", "computer_name": "unknown", "meta": {}}

    meta = {}
    try:
        meta = json.loads(request.headers.get("X-User-Meta", "{}"))
    except Exception:
        pass

    return {
        "domain": request.headers.get("X-User-Domain", "unknown"),
        "computer_name": request.headers.get("X-Computer-Name", "unknown"),
        "meta": meta,
    }


@app.get("/user/history")
async def user_history(request: Request, limit: int = 50):
    user = get_user_identifiers(request)
    hist = analytics.get_user_history(user['domain'], user['computer_name'], limit)
    return {"user": user, "history": hist, "total": len(hist)}

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Поиск документов с расширенными возможностями"""
    try:
        results = searcher.search(
            request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            use_query_expansion=request.use_query_expansion,
            context_window=request.context_window
        )
        
        grouped = defaultdict(list)
        for r in results:
            grouped[r["book"]].append(r)
        
        return {
            "results": results,
            "grouped_results": dict(grouped),
            "total_found": len(results),
            "sources": list(grouped.keys())
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar/{doc_id}")
async def find_similar(doc_id: int, top_k: int = 5):
    """Поиск похожих документов"""
    try:
        original = searcher.get_document_by_id(doc_id)
        if not original:
            raise HTTPException(status_code=404, detail="Документ не найден")
        
        similar = searcher.search_similar(doc_id, top_k)
        
        return {
            "original": original,
            "similar": similar,
            "total_found": len(similar)
        }
    except Exception as e:
        logger.error(f"Similar search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Сохранение обратной связи"""
    try:
        background_tasks.add_task(
            analytics.log_feedback,
            feedback.rating,
            feedback.comment,
            feedback.qa_id
        )
        
        if feedback.rating <= 2:
            logger.warning(f"Negative feedback: {feedback.rating} - {feedback.comment}")
        
        return {
            "status": "success",
            "message": "Спасибо за отзыв! Это поможет улучшить систему.",
            "feedback_id": hashlib.md5(
                f"{feedback.question}_{datetime.now()}".encode()
            ).hexdigest()[:8]
        }
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions")
async def get_suggestions(query: str, limit: int = 5):
    """Автодополнение и предложения"""
    try:
        suggestions = []
        
        for q in list(analytics.queries)[-100:]:
            if query.lower() in q["query"].lower():
                suggestions.append(q["query"])
        
        suggestions = list(dict.fromkeys(suggestions))[:limit]
        
        if len(suggestions) < limit:
            for topic, _ in analytics.popular_topics.items():
                if query.lower() in topic or topic.startswith(query.lower()):
                    suggestions.append(f"Как настроить {topic}")
                    if len(suggestions) >= limit:
                        break
        
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return {"suggestions": []}

@app.get("/stats")
async def get_statistics():
    """Статистика использования"""
    return analytics.get_statistics()

@app.post("/clear_cache")
async def clear_cache():
    """Очистка кэша"""
    response_cache.clear()
    return {"status": "success", "message": "Кэш очищен"}

@app.post("/reset_session/{session_id}")
async def reset_session(session_id: str):
    """Сброс сессии"""
    session_manager.clear(session_id)
    return {"status": "success", "message": f"Сессия {session_id} сброшена"}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # можно прочитать query-параметры от клиента (domain/computer)
        domain = websocket.query_params.get("domain")
        computer = websocket.query_params.get("computer")

        while True:
            try:
                # idle-таймаут + soft heartbeat
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=45)
            except asyncio.TimeoutError:
                # держим соединение живым
                await websocket.send_json({"type": "ping"})
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "bad json"})
                continue

            t = data.get("type")
            if t == "ping":
                await websocket.send_json({"type": "pong"})
            elif t == "auth":
                # тут можно сохранить user_info (domain/computer или из data)
                await websocket.send_json({"type": "status", "message": "auth ok"})
            else:
                # явный ответ на неизвестный тип
                await websocket.send_json({"type": "error", "message": f"unknown type: {t}"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# HTML интерфейс с поддержкой grounding индикатора
ENHANCED_HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
    <link rel="icon" href="/favicon.ico" sizes="any"/>
    <title>Falcon AI Assistant</title>
    <style>
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --primary: #4a5568;
        --primary-dark: #2d3748;
        --secondary: #718096;
        --bg-light: #f7fafc;
        --bg-dark: #2d3748;
        --text-light: #718096;
        --text-dark: #1a202c;
        --success: #48bb78;
        --warning: #ed8936;
        --danger: #f56565;
        --chat-bg: white;
        --input-bg: #f7fafc;
        --header-user-bg: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        --bubble-user-bg: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        --bubble-bot-bg: #f7fafc;
        --bubble-bot-text: #1a202c;
        --input-border-bg: #e2e8f0;
        --border-color: #e2e8f0;
        --body-gradient: linear-gradient(180deg, #e2e8f0 0%, #f7fafc 35%, #edf2f7 70%, #cbd5e0 100%);
  --scrollbar-track: #edf2f7;
  --scrollbar-thumb: #cbd5e0;
  --scrollbar-thumb-hover: #a0aec0;
    }
 
    body.dark-theme {
        --primary: #667eea;
        --primary-dark: #5a67d8;
        --secondary: #764ba2;
        --bg-light: #1a202c;
        --bg-dark: #0f1419;
        --text-light: #cbd5e0;
        --text-dark: #f7fafc;
        --chat-bg: #2d3748;
        --input-bg: #1a202c;
        --header-user-bg: '';
        --bubble-user-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --bubble-bot-bg: #374151;
        --bubble-bot-text: #f7fafc;
        --input-border-bg: #4a5568;
        --border-color: #4a5568;
        --header-border-color: #4a5568;
        --body-gradient: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
  --scrollbar-track: #0000003d;
  --scrollbar-thumb: #4b5563;  
  --scrollbar-thumb-hover: #6b7280;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background: var(--body-gradient);
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
        transition: background 0.3s ease;
    }
    
    .container {
        width: 100%;
        max-width: 1200px;
        background: var(--chat-bg);
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: 90vh;
        max-height: 900px;
        transition: background 0.3s ease;
    }
 
    .floating-menu {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
    
    .menu-button {
        background: var(--bubble-user-bg);
        color: white;
        border: none;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s;
        position: relative;
    }
    
    .menu-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .menu-dropdown {
        position: absolute;
        bottom: 70px;
        right: 0;
        background: var(--chat-bg);
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        opacity: 0;
        visibility: hidden;
        transform: translateY(10px);
        transition: all 0.3s ease;
        min-width: 180px;
        overflow: hidden;
    }
    
    .floating-menu:hover .menu-dropdown {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
    
    .menu-item {
        padding: 12px 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
        transition: background 0.2s;
        border: none;
        background: transparent;
        width: 100%;
        text-align: left;
        font-size: 14px;
        color: var(--text-dark);
    }
    
    .menu-item:hover {
        background: var(--bg-light);
    }
    
    .menu-item-icon {
        font-size: 18px;
    }
    
    .menu-divider {
        height: 1px;
        background: var(--border-color);
        margin: 4px 0;
    }
        
    .theme-switch {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 20px;
    }
    
    .switch {
        position: relative;
        width: 44px;
        height: 24px;
    }
    
    .switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: -3px;
        bottom: 0;
        background-color: #cbd5e0;
        transition: .4s;
        border-radius: 34px;
    }
    
    .slider:before {
        position: absolute;
        content: "";
        height: 16px;
        width: 16px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    
    input:checked + .slider {
        background-color: var(--primary);
    }
    
    input:checked + .slider:before {
        transform: translateX(20px);
    }

        .confirm-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 3000;
            align-items: center;
            justify-content: center;
        }
        
        .confirm-modal.active {
            display: flex;
        }
        
        .confirm-box {
            background: var(--chat-bg);
            border-radius: 16px;
            padding: 24px;
            max-width: 400px;
            animation: slideIn 0.3s;
        }
        
        .confirm-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-dark);
        }
        
        .confirm-message {
            color: var(--text-light);
            margin-bottom: 20px;
        }

        .confirm-buttons {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
        }
        
    .header {
        background: var(--header-user-bg);
        color: white;
        padding: 4px 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
        border-bottom: 1px solid var(--header-border-color);
    }

    .chat-container {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 24px;
        background: var(--chat-bg);
        min-height: 0;
        scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
    }

.chat-container::-webkit-scrollbar {
  width: 14px;
}
.chat-container::-webkit-scrollbar-track {
  background: var(--scrollbar-track);
}
.chat-container::-webkit-scrollbar-thumb {
  background-color: var(--scrollbar-thumb);
  border-radius: 8px;
  border: 2px solid var(--scrollbar-track);
}
.chat-container::-webkit-scrollbar-thumb:hover {
  background-color: var(--scrollbar-thumb-hover);
}
        
    .bot-message .bubble {
        background: var(--bubble-bot-bg);
        color: var(--bubble-bot-text);
        padding: 16px 20px;
        border-radius: 18px 18px 18px 4px;
        max-width: 85%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
        
    .input-container {
        padding: 20px;
        background: var(--chat-bg);
        border-top: 1px solid var(--border-color);
        flex-shrink: 0;
    }

    .input {
        background: var(--input-bg);
    }
        
        .main-content {
            flex: 1;
            display: flex;
            overflow: hidden;
            min-height: 0; 
        }
        
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0; 
        }
 
 
        .bubble {
            word-wrap: break-word;
            overflow-wrap: break-word;
            word-break: break-word;
            hyphens: auto;
        }
        
        .user-message .bubble {
            background: var(--bubble-user-bg);
            color: white;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
 
        
 
        .bot-message .bubble img,
        .bubble .md-img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 8px 0;
            border-radius: 8px;
        }
 
        .input-wrapper {
            display: flex;
            gap: 12px;
            position: relative;
            flex-wrap: wrap;  
        }
        
        #questionInput {
            flex: 1;
            min-width: 200px;
            padding: 14px 18px;
            border: 2px solid var(--border-color);
            border-radius: 12px;
            font-size: 15px;
            transition: all 0.3s;
            background: var(--input-bg);
            color: var(--text-dark);
        }
 
        .feedback-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s;
            z-index: 1000;
        }
        
        .feedback-toggle:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }
 
    .modal-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        z-index: 2000;
        animation: fadeIn 0.3s;
    }
    
    .modal-overlay.active {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal {
        background: var(--chat-bg);
        border-radius: 16px;
        padding: 32px;
        max-width: 600px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        animation: slideIn 0.3s;
    }
    
    .modal h2 {
        color: var(--primary-dark);
        margin-bottom: 24px;
    }
        
    .modal-tabs {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        border-bottom: 2px solid var(--border-color);
    }
    
    .modal-tab {
        padding: 10px 23.8px;
        background: none;
        border: none;
        color: var(--text-light);
        cursor: pointer;
        font-size: 13px;
        transition: all 0.3s;
        position: relative;
    }
    
    .modal-tab.active {
        color: var(--primary);
    }
    
    .modal-tab.active::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--primary);
    }
        
        .modal-content {
            display: none;
        }
        
        .modal-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-dark);
            font-weight: 500;
        }
        
        .form-group textarea,
        .form-group input[type="text"] {
            width: 100%;
            padding-top: 12px;
            padding-bottom: 12px;
            padding-left: 12px;
            border: 2px solid var(--input-border-bg);
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        .form-group textarea:focus,
        .form-group input[type="text"]:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        .file-upload {
            border: 2px dashed #e2e8f0;
            border-radius: 8px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .file-upload:hover {
            border-color: var(--primary);
            background: var(--bg-light);
        }
        
        .modal-buttons {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            margin-top: 24px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
    .btn-primary {
        background: var(--bubble-user-bg);
        color: white;
    }
    
    .btn-secondary {
        background: var(--bg-light);
        color: var(--text-dark);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 85, 104, 0.3);
    }
    
    .btn-secondary:hover {
        background: var(--input-border-bg);
    }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from {
                transform: translateY(-20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
 
        @media (max-width: 768px) {
            .container {
                height: 100vh;
                max-height: 100vh;
                border-radius: 0;
            }
            
            .header {
                padding: 16px 20px;
            }
            
            .header h1 {
                font-size: 18px;
            }
            
            .header .stats {
                display: none;
            }
            
            .chat-container {
                padding: 16px;
            }
            
            .user-message .bubble,
            .bot-message .bubble {
                max-width: 90%;
            }
            
            .modal {
                width: 95%;
                padding: 24px;
            }
            
            .feedback-toggle {
                width: 48px;
                height: 48px;
                font-size: 20px;
            }
        }
 
        .bubble * {
            max-width: 100%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
 
        .bubble pre {
            overflow-x: auto;
            max-width: 100%;
        }
        
        .bubble table {
            display: block;
            overflow-x: auto;
            max-width: 100%;
        }
        .hidden {display:none;}

        .bot-message .bubble img { max-width: 100%; height: auto; display: block; margin: 8px 0; }

        .bubble .md-img { max-width: 100%; height: auto; display: block; margin: 8px 0; border-radius: 8px; }
 
        .header h1 { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
        
        .header .stats { display: flex; gap: 20px; font-size: 14px; opacity: 0.9; }
        
        .main-content { flex: 1; display: flex; overflow: hidden; min-height: 0;}

.citation-link {
  color: var(--primary);
  text-decoration: none;
  border-bottom: 1px dashed rgba(102,126,234,0.5);
  transition: color .2s, border-color .2s;
}
.citation-link:hover {
  color: var(--primary-dark);
  border-color: var(--primary-dark);
}

/* блок иллюстраций из источников */
.images-from-chunks {
  margin-top: 12px;
  padding: 12px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
}
.images-from-chunks h4 {
  margin-bottom: 10px;
  font-weight: 600;
  color: var(--text-dark);
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
}

.img-wrap {
  position: relative; overflow: visible;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  cursor: zoom-in;
}
.img-frame {
  display: block;
  overflow: hidden;      
  border-radius: 12px; 
}
img.doc-img {
  display: block;
  width: 100%;
  height: 160px;
  object-fit: scale-down; 
  transition: transform .25s ease;
}
.img-wrap:hover img.doc-img { transform: scale(1.03); }

.lightbox {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.9);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
.lightbox.open { display: flex; }
.lightbox img.lb-img {
  max-width: 95vw;
  max-height: 90vh;
  border-radius: 8px;
  box-shadow: 0 10px 40px rgba(0,0,0,.5);
}

.lb-btn {
  position: absolute;
  border: none;
  background: rgba(255,255,255,.12);
  color: #fff;
  width: 44px;
  height: 44px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  font-size: 22px;
  cursor: pointer;
  transition: background .2s ease, transform .1s ease;
  user-select: none;
}
.lb-btn:hover { background: rgba(255,255,255,.2); }
.lb-btn:active { transform: scale(.96); }

.lb-close { top: 20px; right: 20px; }
.lb-prev  { left: 20px;  top: 50%; transform: translateY(-50%); }
.lb-next  { right: 20px; top: 50%; transform: translateY(-50%); }

.lb-caption {
  position: absolute;
  left: 50%;
  bottom: 24px;
  transform: translateX(-50%);
  max-width: 90vw;
  color: #e2e8f0;
  font-size: 14px;
  text-align: center;
  line-height: 1.35;
  opacity: .95;
  word-break: break-word;
}

        .sidebar {
            width: 280px;
            background: var(--bg-light);
            border-right: 1px solid #e2e8f0;
            padding: 20px;
            overflow-y: auto;
        }
        
        .sidebar h3 {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .option-group {
            margin-bottom: 24px;
        }
        
        .option-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 14px;
            color: var(--text-dark);
        }
        
        .option-item input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        
        .option-item select {
            padding: 6px 10px;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            background: white;
            font-size: 14px;
        }
        
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column; min-width: 0;
        }
 
        .message {
            margin-bottom: 24px;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .input-wrapper {
            display: flex;
            gap: 12px;
            position: relative;
            flex-wrap: wrap;
        }
        
        .user-message {
            display: flex;
            justify-content: flex-end;
        }
        
    .user-message .bubble {
        background: var(--bubble-user-bg);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
        
        .bot-message .bubble {
            background: var(--bg-light);
            color: var(--text-dark);
            padding: 16px 20px;
            border-radius: 18px 18px 18px 4px;
            max-width: 85%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .bot-message .bubble h1, 
        .bot-message .bubble h2,
        .bot-message .bubble h3 {
            color: var(--primary-dark);
            margin: 16px 0 8px 0;
        }
        
        .bot-message .bubble h1 { font-size: 20px; }
        .bot-message .bubble h2 { font-size: 18px; }
        .bot-message .bubble h3 { font-size: 16px; }
        
        .bot-message .bubble ul,
        .bot-message .bubble ol {
            margin: 8px 0;
            padding-left: 24px;
        }
        
        .bot-message .bubble li {
            margin: 4px 0;
        }
        
        .bot-message .bubble code {
            background: #2d3748;
            color: #48bb78;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 13px;
        }
        
        .bot-message .bubble pre {
            background: #2d3748;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
        }
        
        .confidence-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            font-size: 13px;
            color: var(--text-light);
        }
        
        .confidence-bar {
            width: 100px;
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .confidence-high { background: var(--success); }
        .confidence-medium { background: var(--warning); }
        .confidence-low { background: var(--danger); }
        
        .grounding-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 4px;
            font-size: 13px;
            color: var(--text-light);
        }
        
        .citations {
            margin-top: 12px;
            padding: 12px;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 13px;
        }
        
        .citations h4 {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-dark);
        }
        
        .citation {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            border-bottom: 1px solid #f7fafc;
        }
        
        .citation:last-child {
            border-bottom: none;
        }
        
        .citation-icon {
            color: var(--primary);
        }


.citation-link {
  color: var(--primary);
  text-decoration: none;
  border-bottom: 1px dashed rgba(102,126,234,0.45);
  padding: 0 2px;
  border-radius: 6px;
  transition: background .15s ease, border-color .15s ease, color .15s ease;
}
.citation-link:hover {
  background: rgba(102,126,234,0.08);
  border-color: var(--primary-dark);
  color: var(--primary-dark);
}
.ext-icon { margin-left: 4px; font-size: 12px; opacity: .7; }
 
.tooltip { position: relative; }
.tooltip::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 0;
  bottom: 130%;
  white-space: pre-line;
  padding: 6px 8px;
  font-size: 12px;
  background: #1a202c; /* var(--text-dark) */
  color: #fff;
  border-radius: 6px;
  opacity: 0;
  transform: translateY(6px);
  pointer-events: none;
  transition: opacity .15s ease, transform .15s ease;
  box-shadow: 0 6px 20px rgba(0,0,0,.15);
}
.tooltip::before {
  content: "";
  position: absolute;
  left: 10px;
  bottom: 118%;
  border: 6px solid transparent;
  border-top-color: #1a202c;
  opacity: 0;
  transition: opacity .15s ease;
}
.tooltip:hover::after,
.tooltip:focus::after { opacity: 1; transform: translateY(0); }
.tooltip:hover::before,
.tooltip:focus::before { opacity: 1; }
 
.badge {
  margin-left: auto;
  background: var(--bg-light);
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  color: var(--text-light);
}
        
        .citation-relevance {
            margin-left: auto;
            background: var(--bg-light);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            color: var(--text-light);
        }
        
    .input-wrapper {
        display: flex;
        gap: 12px;
        position: relative;
        flex-wrap: wrap;  
    }
    
    #questionInput {
        flex: 1;
        min-width: 200px;
        padding: 14px 18px;
        border: 2px solid var(--border-color);
        border-radius: 12px;
        font-size: 15px;
        transition: all 0.3s;
        background: var(--input-bg);
        color: var(--text-dark);
    }

    .text {
        color: var(--text-dark);
    }
        
    #questionInput:focus {
        outline: none;
        border-color: var(--primary);
        background: var(--input-bg);
        box-shadow: 0 0 0 3px rgba(74, 85, 104, 0.1);
    }

.app-title{
  display:flex;
  align-items:center;
  gap:.5rem;
  line-height:1.2;
  font-size:1.25rem; 
}
.app-logo{
  width:34px;
  height:34px;
  object-fit:contain;
  vertical-align:middle;
}
 
        .suggestions {
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 8px;
            box-shadow: 0 -5px 15px rgba(0,0,0,0.1);
            max-height: 200px;
            overflow-y: auto;
            display: none;
        }
        
        .suggestion-item {
            padding: 10px 16px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 14px;
        }
        
        .suggestion-item:hover {
            background: var(--bg-light);
        }
        
    #sendBtn {
        padding: 14px 28px;
        background: var(--bubble-user-bg);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        min-width: 120px;
    }
        
        #sendBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        #sendBtn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .typing {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 8px 12px;
            background: var(--bg-light);
            border-radius: 12px;
        }
        
        .typing span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary);
            animation: typing 1.4s infinite;
        }
        
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }
        
        .rating {
            display: flex;
            gap: 4px;
            margin-top: 12px;
        }
        
        .star {
            cursor: pointer;
            font-size: 18px;
            color: #cbd5e0;
            transition: color 0.2s;
        }
        
        .star:hover,
        .star.active {
            color: #f6ad55;
        }
        
        .quick-action {
            padding: 6px 12px;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-action:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            
            .container {
                height: 100vh;
                border-radius: 0;
            }
        }
    </style>
    <script src="/js/marked.min.js"></script>
    <script src="/js/purify.min.js"></script>
</head>
<body>
    <!-- Плавающее меню -->
    <div class="floating-menu">
        <div class="menu-dropdown">
            <button class="menu-item" onclick="toggleFeedbackModal()">
                <span class="menu-item-icon">📚</span>
                <span>Обучить</span>
            </button>
            <div class="menu-divider"></div>
            <div class="theme-switch">
                <span style="display: flex; align-items: center; gap: 8px;">
                    <span class="text" style="font-size: 14px;">Темная тема</span>
                </span>
                <label class="switch">
                    <input type="checkbox" id="themeToggle" onchange="toggleTheme()">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="menu-divider"></div>
            <button class="menu-item" onclick="confirmClearChat()">
                <span class="menu-item-icon">🗑️</span>
                <span>Очистить чат</span>
            </button>
        </div>
        <button class="menu-button" title="Меню">
            ⚙️
        </button>
    </div>
    
    <!-- Модальное окно подтверждения -->
    <div class="confirm-modal" id="confirmModal">
        <div class="confirm-box">
            <div class="confirm-title">Очистить историю чата?</div>
            <div class="confirm-message hidden">История будет скрыта, но сохранится в базе данных для улучшения системы.</div>
            <div class="confirm-buttons">
                <button class="btn btn-secondary" onclick="closeConfirmModal()">Отмена</button>
                <button class="btn btn-primary" onclick="clearChat()">Очистить</button>
            </div>
        </div>
    </div>
    
    <!-- Модальное окно обучения (из оригинала) -->
    <div class="modal-overlay" id="feedbackModal">
        <div class="modal">
            <h2>Обучение и обратная связь</h2>
            
            <div class="modal-tabs">
                <button class="modal-tab active" onclick="switchTab('training')">
                    📚 Добавить материалы
                </button>
                <button class="modal-tab" onclick="switchTab('inaccuracy')">
                    ⚠️ Сообщить о неточности
                </button>
                <button class="modal-tab" onclick="switchTab('file')">
                    📎 Загрузить файл
                </button>
            </div>
            
            <!-- Вкладка добавления материалов -->
            <div class="modal-content active" id="training-tab">
                <form onsubmit="submitTrainingData(event)">
                    <div class="form-group">
                        <label>Заголовок материала</label>
                        <input class="input text" type="text" id="trainingTitle" placeholder="Например: Инструкция по настройке модуля X">
                    </div>
                    <div class="form-group">
                        <label>Текст для обучения</label>
                        <textarea class="input text" id="trainingText" rows="8" placeholder="Вставьте текст, который поможет улучшить ответы системы..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label>Категория (опционально)</label>
                        <input class="input text" type="text" id="trainingCategory" placeholder="Например: Настройка, FAQ, Инструкции">
                    </div>
                    <div class="modal-buttons">
                        <button type="button" class="btn btn-secondary" onclick="toggleFeedbackModal()">Отмена</button>
                        <button type="submit" class="btn btn-primary">Отправить</button>
                    </div>
                </form>
            </div>
            
            <!-- Вкладка сообщения о неточности -->
            <div class="modal-content" id="inaccuracy-tab">
                <form onsubmit="submitInaccuracy(event)">
                    <div class="form-group">
                        <label>Описание неточности</label>
                        <textarea class="input text" id="inaccuracyText" rows="6" placeholder="Опишите, какая информация неверна и как должно быть правильно..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label>Правильная информация</label>
                        <textarea class="input text" id="correctInfo" rows="6" placeholder="Укажите правильную информацию..."></textarea>
                    </div>
                    <div class="modal-buttons">
                        <button type="button" class="btn btn-secondary" onclick="toggleFeedbackModal()">Отмена</button>
                        <button type="submit" class="btn btn-primary">Отправить</button>
                    </div>
                </form>
            </div>
            
            <!-- Вкладка загрузки файла -->
            <div class="modal-content" id="file-tab">
                <form onsubmit="submitFile(event)">
                    <div class="form-group">
                        <div class="file-upload text" onclick="document.getElementById('fileInput').click()">
                            <div id="fileInfo">
                                📁 Нажмите для выбора файла<br>
                                <small>Поддерживаются: PDF, DOCX, TXT, MD</small>
                            </div>
                            <input class="input text" type="file" id="fileInput" style="display: none;" accept=".pdf,.docx,.doc,.txt,.md" onchange="handleFileSelect(event)">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Комментарий к файлу</label>
                        <textarea class="input text" id="fileComment" rows="4" placeholder="Опишите содержание файла..."></textarea>
                    </div>
                    <div class="modal-buttons">
                        <button type="button" class="btn btn-secondary" onclick="toggleFeedbackModal()">Отмена</button>
                        <button type="submit" class="btn btn-primary">Загрузить</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="header">
           <h1 class="app-title">
  <img
    src="/js/logo.png"
    sizes="24px"
    alt="Falcon"
    class="app-logo"
    width="24"
    height="24"
    decoding="async" draggable="false"
    loading="eager"
    onerror="this.style.display='none'"  
  />
  Falcon AI Assistant
</h1>
            <div class="stats">
                <span id="queryCount">0 запросов</span>
                <span id="avgTime">0.0с среднее время</span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="sidebar hidden">
                <div class="option-group">
                    <h3>Параметры поиска</h3>
                    <div class="option-item">
                        <input type="checkbox" id="useLLM" checked>
                        <label for="useLLM">Использовать ИИ</label>
                    </div>
                    <div class="option-item">
                        <input type="checkbox" id="useCache">
                        <label for="useCache">Кэширование</label>
                    </div>
                    <div class="option-item">
                        <input type="checkbox" id="useCOT" checked>
                        <label for="useCOT">Chain of Thought</label>
                    </div>
                    <div class="option-item">
                        <label>Результатов:</label>
                        <select id="topK">
                            <option value="5">5</option>
                            <option value="10" selected>10</option>
                            <option value="15">15</option>
                            <option value="20">20</option>
                        </select>
                    </div>
                </div>
                
                <div class="option-group">
                    <h3>Методы поиска</h3>
                    <div class="option-item">
                        <input type="checkbox" id="vectorSearch" checked>
                        <label for="vectorSearch">Векторный</label>
                    </div>
                    <div class="option-item">
                        <input type="checkbox" id="fulltextSearch" checked>
                        <label for="fulltextSearch">Полнотекстовый</label>
                    </div>
                </div>
                
                <div class="option-group">
                    <h3>Популярные темы</h3>
                    <div id="popularTopics"></div>
                </div>
            </div>
            
            <div class="chat-area">
                <div class="chat-container" id="chatContainer">
                    <div class="bot-message message">
                        <div class="bubble" id="welcome"></div>
                    </div>
                </div>
                
                <div class="input-container">
                    <div class="input-wrapper">
                        <div class="suggestions" id="suggestions"></div>
                        <input 
                            type="text" 
                            id="questionInput" 
                            placeholder="Введите ваш вопрос..."
                            autocomplete="off"
                        >
                        <button id="sendBtn">Отправить</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>        
        function formatConfidence(score) {
            const percent = (score * 100).toFixed(0);
            let className = 'confidence-low';
            if (score > 0.7) className = 'confidence-high';
            else if (score > 0.4) className = 'confidence-medium';
            return {percent, className};
        }
        // Функция получения идентификаторов пользователя
        function getUserIdentifiers() {
            // Для браузера мы можем получить только базовую информацию
            const identifiers = {
                // Домен из URL
                domain: window.location.hostname || 'localhost',
                // Уникальный ID браузера (сохраняем в localStorage)
                computer_name: getOrCreateBrowserId(),
                // Дополнительная информация
                screen: `${screen.width}x${screen.height}`,
                platform: navigator.platform,
                language: navigator.language
            };
            
            return identifiers;
        }
        
        // Глобальные переменные
        let isProcessing = false;
        const user = getUserIdentifiers();
        let conversationId = user.computer_name; 
        window.queryCount = window.queryCount ?? 0;
        window.totalTime  = window.totalTime  ?? 0;
        let suggestions = [];
        
        // WebSocket для real-time
        let ws = null;
        let reconnectAttempts = 0;
        const MAX_RETRY_MS = 30000;
        let heartbeatTimer;
        
        function connectWebSocket() {
            const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${wsProto}://${location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 3000);
            };
        }

function extractImagesFromChunks(chunks) {
  const seen = new Set();   // глобальная дедупликация по src
  const items = [];
  // Матч: ![alt](/media/что-угодно_до_закрывающей_скобки)
  const mdRe = /!\[([^\]]*)\]\((?:\s*)(\/media\/[^)\s]+)(?:\s*)\)/g;

  (chunks || []).forEach(c => {
    const book    = c.book || '';
    const section = c.section || c?.meta?.section_title || c?.parent_title || '';
    const page    = Number.isInteger(c.page) && c.page > 0 ? c.page : null;
    const ctx     = [book, section, page ? `стр. ${page}` : ''].filter(Boolean).join(' • ');

    let m, t = String(c.text || "");
    while ((m = mdRe.exec(t)) !== null) {
      const src = m[2];
      if (seen.has(src)) continue;
      seen.add(src);

      const rawAlt = m[1] || src.split("/").pop();
      const alt    = String(rawAlt).trim();

      items.push({ src, alt, tooltip: ctx });
    }
  });

  return items;
}


        
        function handleWebSocketMessage(data) {
            if (data.type === 'status') {
                updateStatus(data.message);
            } else if (data.type === 'answer') {
                removeTypingIndicator();
                displayAnswer(data);
            } else if (data.type === 'error') {
                removeTypingIndicator();
                displayError(data.message);
            }
        }
        
        async function sendQuestion() { return sendQuestionWithAuth(); }
        
        function displayAnswer(data) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            
            let html = '<div class="bubble">';

            if (data.blocked) {
            html += `
                <div style="margin:-4px 0 12px 0;padding:10px 12px;border-left:4px solid #ed8936;background:#FFF7ED;color:#7B341E;border-radius:6px;">
                Ответ скрыт из-за низкой уверенности.
                </div>
            `;
            }

            html += renderMarkdownSafe(data.answer);

            // Индикатор уверенности
            const confScore = data.confidence_score ?? data.confidence;
            if (confScore !== undefined) {
                const conf = formatConfidence(confScore);
                html += `
                    <div class="confidence-indicator">
                        <span>Уверенность:</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill ${conf.className}" style="width: ${conf.percent}%"></div>
                        </div>
                        <span>${conf.percent}%</span>
                    </div>
                `;
            }
            
            // Индикатор grounding
            if (data.grounding_score !== undefined && data.grounding_score !== null) {
                const ground = formatConfidence(data.grounding_score);
                html += `
                    <div class="grounding-indicator">
                        <span>Подтверждение источниками:</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill ${ground.className}" style="width: ${ground.percent}%"></div>
                        </div>
                        <span>${ground.percent}%</span>
                    </div>
                `;
            }
            
            html += '</div>';

const imgs = extractImagesFromChunks(data.chunks);
if (imgs.length) {
  html += `
    <div class="images-from-chunks">
      <h4>🖼 Иллюстрации из источников:</h4>
      <div class="image-grid">
        ${imgs.slice(0, 8).map(({src, alt, tooltip}) => `
          <figure class="img-wrap tooltip" data-tooltip="${esc(tooltip)}">
            <span class="img-frame">
              <img
                src="${src}"
                alt="${esc(tooltip)}"
                class="doc-img"
                loading="lazy"
                referrerpolicy="no-referrer"
              />
            </span>
          </figure>
        `).join("")}
      </div>
    </div>
  `;
}

            
// экранирование текста
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Источники
if (data.citations && data.citations.length > 0) {
  html += '<div class="citations">';
  html += '<h4>📚 Источники:</h4>';

  data.citations.forEach(cit => {
    const relevance = Math.min(100,(cit.relevance * 100).toFixed(0));
    const hasPage = Number.isInteger(cit.page) && cit.page > 0;
    const url = `/resolve?book=${encodeURIComponent(cit.book)}${hasPage ? `&page=${cit.page}` : ''}`;

    html += `
      <div class="citation">
        <span class="citation-icon">📄</span>
        <span class="citation-text">
          <a href="${url}"
             target="_blank" rel="noopener"
             class="citation-link tooltip"
             aria-label="Открыть исходный документ в новой вкладке"
             data-tooltip="Открыть документ ↗">
             ${esc(cit.book)}<span class="ext-icon" aria-hidden="true">↗</span>
          </a>
          — ${esc(cit.section || 'Без раздела')}${hasPage ? ` (стр. ${cit.page})` : ''}
        </span>
        <span class="citation-relevance badge" title="Релевантность">${relevance}%</span>
      </div>
    `;
  });

  html += '</div>';
}

            
            // Рейтинг
            html += `
                <div class="rating">
                    ${[1,2,3,4,5].map(i => 
                        `<span class="star" onclick="rate(${i}, this)">☆</span>`
                    ).join('')}
                </div>
            `;
            
            messageDiv.innerHTML = html;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

          function renderMarkdownSafe(md) {
    const raw = marked.parse(md || '', { gfm: true, breaks: true, mangle: false, headerIds: false });

    if (window.DOMPurify) {
      const clean = DOMPurify.sanitize(raw, {
        // расширяем дефолтные допустимые теги/атрибуты
        ADD_TAGS: ['img'],
        ADD_ATTR: ['src','alt','title','loading','referrerpolicy','width','height','class']
      });
      return clean.replace(/<img /g, '<img loading="lazy" referrerpolicy="no-referrer" class="md-img" ');
    }

    // fallback, если по какой-то причине DOMPurify не загрузился
    return raw;
  }

  (function initLightbox() {
  // оверлей
  const lb = document.createElement('div');
  lb.className = 'lightbox';
  lb.innerHTML = `
    <button class="lb-btn lb-close" aria-label="Закрыть">×</button>
    <button class="lb-btn lb-prev"  aria-label="Предыдущая">‹</button>
    <img class="lb-img" alt="">
    <button class="lb-btn lb-next"  aria-label="Следующая">›</button>
    <div class="lb-caption"></div>
  `;
  document.body.appendChild(lb);

  const imgEl = lb.querySelector('.lb-img');
  const captionEl = lb.querySelector('.lb-caption');

  const state = { items: [], index: 0 };

  function updateLB() {
    const item = state.items[state.index];
    if (!item) return;
    imgEl.src = item.src;
    imgEl.alt = item.alt || '';
    captionEl.textContent = item.alt || '';
  }
  function openLB(items, index = 0) {
    state.items = items;
    state.index = index;
    updateLB();
    lb.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
  function closeLB() {
    lb.classList.remove('open');
    document.body.style.overflow = '';
    // небольшая задержка очистки (чтобы не мигало при быстром закрытии/открытии)
    setTimeout(() => { imgEl.src = ''; }, 150);
  }
  function prevLB() {
    state.index = (state.index - 1 + state.items.length) % state.items.length;
    updateLB();
  }
  function nextLB() {
    state.index = (state.index + 1) % state.items.length;
    updateLB();
  }

  lb.querySelector('.lb-close').addEventListener('click', closeLB);
  lb.querySelector('.lb-prev').addEventListener('click', prevLB);
  lb.querySelector('.lb-next').addEventListener('click', nextLB);

  // клик по тёмной подложке — закрыть
  lb.addEventListener('click', (e) => {
    if (e.target === lb) closeLB();
  });

  // клавиатура в открытом режиме
  document.addEventListener('keydown', (e) => {
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape')      return closeLB();
    if (e.key === 'ArrowLeft')   return prevLB();
    if (e.key === 'ArrowRight')  return nextLB();
  });

  // Делегирование кликов по превью (работает для всех сообщений)
  document.addEventListener('click', (e) => {
    const clicked = e.target.closest('img.doc-img');
    if (!clicked) return;

    // находим контейнер данной галереи и собираем все её картинки
    const container = clicked.closest('.images-from-chunks');
    if (!container) return;

    const list = Array.from(container.querySelectorAll('img.doc-img'));
    const items = list.map(img => ({
      src: img.getAttribute('data-full') || img.currentSrc || img.src,
      alt: img.getAttribute('alt') || ''
    }));
    const index = Math.max(0, list.indexOf(clicked));

    openLB(items, index);
  });
})();
        
        function displayError(message) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            messageDiv.innerHTML = `
                <div class="bubble" style="background: #fee;">
                    ❌ ${message}
                </div>
            `;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = text;
            
            messageDiv.appendChild(bubble);
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function addTypingIndicator() {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            messageDiv.id = 'typing-' + Date.now();
            
            messageDiv.innerHTML = `
                <div class="typing">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
            
            return messageDiv.id;
        }
        
        function removeTypingIndicator(id) {
            const elements = id ? [document.getElementById(id)] : 
                            document.querySelectorAll('[id^="typing-"]');
            elements.forEach(el => el?.remove());
        }
        
        function updateStatus(message) {
            console.log('Status:', message);
        }
        
        function formatTime(seconds) {
            const s = Number(seconds);
            return Number.isFinite(s) ? s.toFixed(2) + 'с' : '–';
        }

        function updateStats(processingTime) {
            const t = Number(processingTime);
            if (!Number.isFinite(t)) return;                 // не портим счётчики

            if (!Number.isFinite(window.totalTime))  window.totalTime  = 0;
            if (!Number.isFinite(window.queryCount)) window.queryCount = 0;
            window.queryCount += 1;
            window.totalTime  += t;

            const avg = window.totalTime / window.queryCount;
            document.getElementById('queryCount').textContent = `${window.queryCount} запросов`;
            document.getElementById('avgTime').textContent    = `${formatTime(avg)} среднее`;
        }
        
        async function loadSuggestions(query) {
            if (query.length < 2) {
                hideSuggestions();
                return;
            }
            
            try {
                const response = await fetch(`/suggestions?query=${encodeURIComponent(query)}&limit=5`);
                const data = await response.json();
                
                if (data.suggestions && data.suggestions.length > 0) {
                    showSuggestions(data.suggestions);
                } else {
                    hideSuggestions();
                }
            } catch (error) {
                console.error('Error loading suggestions:', error);
            }
        }
        
        function showSuggestions(items) {
        const container = document.getElementById('suggestions');
        container.innerHTML = items.map(item =>
            `<div class="suggestion-item" onclick="selectSuggestion('${item.replace(/'/g, "\\'")}')">${item}</div>`
        ).join('');
        container.style.display = 'block';
        }

        function hideSuggestions() {
            document.getElementById('suggestions').style.display = 'none';
        }
        
        function selectSuggestion(text) {
            document.getElementById('questionInput').value = text;
            hideSuggestions();
            sendQuestion();
        }
        
        function askQuestion(text) {
            document.getElementById('questionInput').value = text;
            sendQuestion();
        }
        
        function rate(rating, element) {
            const stars = element.parentElement.querySelectorAll('.star');
            stars.forEach((star, index) => {
                star.classList.toggle('active', index < rating);
                star.textContent = index < rating ? '★' : '☆';
            });
            
            fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    question: 'last_question',
                    answer: 'last_answer',
                    rating: rating,
                    qa_id: window.lastQAId || null
                })
            });
        }
        
async function loadPopularTopics() {
  try {
    const res = await fetch('/stats');
    const data = await res.json();

    const toArray = (pt) => Array.isArray(pt)
      ? pt
      : Object.entries(pt || {}).map(([topic, score]) => ({ topic, score }));

    const items = toArray(data.popular_topics).slice(0, 5);

    const container = document.getElementById('popularTopics');
    container.innerHTML = items.map(({ topic }) =>
      `<div class="quick-action" onclick="askQuestion('${(topic || '').replace(/'/g, "\\'")}')">${topic}</div>`
    ).join('') || '<div style="color:#718096;font-size:13px;"></div>';
  } catch (e) {
    console.error('Error loading topics:', e);
  }
}

        
        // Инициализация
        document.addEventListener('DOMContentLoaded', () => {
  const welcomeMd = `
👋 Здравствуйте!

Я интеллектуальный помощник по документации **Falcon**.

**Чем могу помочь:**
- Найти нужную информацию в документации
- Объяснить сложные концепции
- Предоставить пошаговые инструкции

Задайте ваш вопрос — и я найду наиболее релевантную информацию!
  `.trim();

  document.getElementById('welcome').innerHTML = renderMarkdownSafe(welcomeMd);

            // Подключаем WebSocket (если нужен real-time)
            // connectWebSocket();
            
           //  loadPopularTopics();
            
            const input = document.getElementById('questionInput');
            
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendQuestion();
                }
            });
            
            input.addEventListener('input', (e) => {
                loadSuggestions(e.target.value);
            });
            
            input.addEventListener('blur', () => {
                setTimeout(hideSuggestions, 200);
            });
            
            document.getElementById('sendBtn').addEventListener('click', sendQuestion);
            
           //  setInterval(loadPopularTopics, 60000);
        });

// Создание или получение уникального ID браузера
function getOrCreateBrowserId() {
    let browserId = localStorage.getItem('browser_id');
    if (!browserId) {
        // Генерируем уникальный ID на основе различных параметров
        const fingerprint = [
            navigator.userAgent,
            navigator.language,
            screen.width + 'x' + screen.height,
            screen.colorDepth,
            new Date().getTimezoneOffset(),
            navigator.platform
        ].join('|');

        // Простой хеш для создания ID
        browserId = 'browser_' + btoa(fingerprint)
            .replace(/[^a-zA-Z0-9]/g, '')
            .substring(0, 16);

        localStorage.setItem('browser_id', browserId);
    }
    return browserId;
}

// Обновленная функция отправки вопроса с идентификацией
async function sendQuestionWithAuth() {
    if (isProcessing) return;

    const input = document.getElementById('questionInput');
    const btn   = document.getElementById('sendBtn');
    const question = input.value.trim();
    if (!question) return;

    isProcessing = true;
    document.getElementById('sendBtn').disabled = true;

    // Получаем идентификаторы пользователя
    const userInfo = getUserIdentifiers();

    addMessage(question, 'user');
    input.value = '';
    hideSuggestions();

    const loadingId = addTypingIndicator();

    const searchMethods = [];
    if (document.getElementById('vectorSearch').checked) searchMethods.push('vector');
    if (document.getElementById('fulltextSearch').checked) searchMethods.push('fulltext');
try {
try {
  const response = await fetch('/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json', // подскажем серверу вернуть JSON
      'X-User-Domain': userInfo.domain,
      'X-Computer-Name': userInfo.computer_name,
      'X-User-Meta': JSON.stringify({
        screen: userInfo.screen,
        platform: userInfo.platform,
        language: userInfo.language
      })
    },
    body: JSON.stringify({
      text: question,
      top_k: parseInt(document.getElementById('topK').value, 10),
      use_llm: document.getElementById('useLLM').checked,
      use_cache: document.getElementById('useCache').checked,
      use_cot: document.getElementById('useCOT').checked,
      search_methods: searchMethods,
      conversation_id: userInfo.computer_name
    })
  });

  const ctype = (response.headers.get('content-type') || '').toLowerCase();
  let data = null;
  let textBody = null;

  if (ctype.includes('application/json')) {
    // Может бросить, если тело битое — ловим отдельно
    try { data = await response.json(); }
    catch (e) {
      console.error('JSON parse error for /ask:', e);
      textBody = await response.text(); // сохраним тело для логов/диагностики
    }
  } else {
    // Это типичная ситуация при 502/504/HTML-страницах ошибок
    textBody = await response.text();
  }

  if (!response.ok) {
    const serverMsg =
      (data && (data.detail || data.message || data.error)) ||
      textBody ||
      `HTTP ${response.status}`;
    console.error('Bad response from /ask', {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      body: data ?? textBody
    });
    removeTypingIndicator(loadingId);
    displayError(`Ошибка: ${serverMsg}`);
    return;
  }

  if (!data) {
    // Сервер ответил OK, но пришёл не-JSON или парсинг не удался
    console.error('Expected JSON from /ask, got:', {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      body: textBody
    });
    removeTypingIndicator(loadingId);
    displayError('Сервер вернул неожиданный формат ответа.');
    return;
  }

  removeTypingIndicator(loadingId);

  // Защитимся от падения в displayAnswer/updateStats
  try {
    displayAnswer(data);
  } catch (e) {
    console.error('displayAnswer failed:', e, data);
    displayError('Ошибка отображения ответа.');
    return;
  }

  if (typeof data.processing_time === 'number') {
    updateStats(data.processing_time);
  }
  window.lastQAId = data.qa_id;
} catch (error) {
  removeTypingIndicator(loadingId);
  // Сетевые/скриптовые ошибки сюда
  displayError(`Произошла ошибка: ${(error && error.message) || String(error)}`);
  console.error('Request to /ask failed:', error);
}
  } finally {
    // гарантия восстановления UI в любом случае
    removeTypingIndicator(loadingId);
    isProcessing = false;
    btn.disabled = false;
    input.focus();
  }
}

// Обновленная функция WebSocket с идентификацией
function connectWebSocketWithAuth() {
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    const userInfo = getUserIdentifiers();

    const params = new URLSearchParams({
        domain: userInfo.domain,
        computer: userInfo.computer_name
    });

  ws = new WebSocket(`${wsProto}://${location.host}/ws?${params}`);

  const scheduleReconnect = () => {
    const delay = Math.min(1000 * 2 ** reconnectAttempts++, MAX_RETRY_MS);
    setTimeout(connectWebSocketWithAuth, delay);
  };

  const armHeartbeat = () => {
    clearTimeout(heartbeatTimer);
    // шлём ping, если нет ответа — закрываем и переподключаемся
    heartbeatTimer = setTimeout(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        try { ws.send(JSON.stringify({ type: 'ping' })); } catch {}
        heartbeatTimer = setTimeout(() => { try { ws.close(); } catch {} }, 10000);
      }
    }, 25000);
  };

  ws.onopen = () => {
    reconnectAttempts = 0;
    try { ws.send(JSON.stringify({ type: 'auth', ...userInfo })); } catch {}
    armHeartbeat();
    console.log('WebSocket connected');
  };

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === 'pong') return armHeartbeat();
      handleWebSocketMessage(data); // у тебя уже есть эта функция
    } catch (e) {
      console.warn('WS message parse error', e);
    }
  };

  ws.onerror = (e) => {
    console.error('WebSocket error:', e);
  };

  ws.onclose = () => {
    clearTimeout(heartbeatTimer);
    console.log('WebSocket disconnected');
    scheduleReconnect();
  };
}

window.addEventListener('beforeunload', () => { try { ws?.close(); } catch {} });

// Добавляем функцию просмотра истории пользователя
async function loadUserHistory() {
    const userInfo = getUserIdentifiers();
    
    try {
        const response = await fetch('/user/history', {
            headers: {
                'X-User-Domain': userInfo.domain,
                'X-Computer-Name': userInfo.computer_name
            }
        });
        
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            displayUserHistory(data.history);
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Функция отображения истории (исправленный порядок + плашка внизу)
function displayUserHistory(history) {
  const container = document.getElementById('chatContainer');

  // Сносим предыдущий блок истории
  container.querySelectorAll('.history-divider, .history-entry').forEach(el => el.remove());

  // Хронологически: старые → новые (избегаем мутации входного массива)
  const ordered = Array.isArray(history)
    ? history.slice().sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    : [];

  const frag = document.createDocumentFragment();

  ordered.forEach(item => {
    // Вопрос
    const qDiv = document.createElement('div');
    qDiv.className = 'message user-message history-entry';
    const qHtml = `
      <div class="bubble" style="opacity: 0.85;">
        ${window.DOMPurify ? DOMPurify.sanitize(item.question || '') : (item.question || '')}
        <div style="font-size: 11px; margin-top: 5px; opacity: 0.6;">
          ${new Date(item.timestamp).toLocaleString()}
        </div>
      </div>
    `;
    qDiv.innerHTML = qHtml;
    frag.appendChild(qDiv);

    // Ответ (превью + полная версия)
    const aDiv = document.createElement('div');
    aDiv.className = 'message bot-message history-entry';

    const fullHtml = renderMarkdownSafe(item.answer || '');
    const needToggle = (item.answer || '').length > 400;
    const previewText = needToggle ? (item.answer.slice(0, 400) + '…') : (item.answer || '');
    const previewHtml = renderMarkdownSafe(previewText);
    const conf = (item.confidence !== null && item.confidence !== undefined)
      ? (item.confidence * 100).toFixed(0) + '%'
      : null;

    aDiv.innerHTML = `
      <div class="bubble" style="opacity: 0.9;">
        <div class="history-answer" data-expanded="false">
          <div class="history-answer-short">${previewHtml}</div>
          <div class="history-answer-full" style="display:none;">${fullHtml}</div>
          ${needToggle ? '<button class="toggle-history quick-action" style="margin-top:8px;">Показать полностью</button>' : ''}
        </div>
        ${conf ? `
          <div class="confidence-indicator" style="margin-top:8px;">
            <span style="font-size:12px;">Уверенность: ${conf}</span>
          </div>` : ''
        }
        ${item.rating ? `
          <div style="margin-top:6px;">${'★'.repeat(item.rating)}${'☆'.repeat(5 - item.rating)}</div>
        ` : ''}
      </div>
    `;
    frag.appendChild(aDiv);

    // Тоггл «Показать полностью / Свернуть»
    const btn = aDiv.querySelector('.toggle-history');
    if (btn) {
      btn.addEventListener('click', () => {
        const wrap = aDiv.querySelector('.history-answer');
        const shortEl = aDiv.querySelector('.history-answer-short');
        const fullEl  = aDiv.querySelector('.history-answer-full');
        const expanded = wrap.getAttribute('data-expanded') === 'true';
        if (expanded) {
          fullEl.style.display = 'none';
          shortEl.style.display = 'block';
          btn.textContent = 'Показать полностью';
        } else {
          shortEl.style.display = 'none';
          fullEl.style.display = 'block';
          btn.textContent = 'Свернуть';
        }
        wrap.setAttribute('data-expanded', String(!expanded));
      });
    }
  });

  // Плашка — ВНИЗУ
  const divider = document.createElement('div');
  divider.className = 'history-divider';
  divider.innerHTML = `
    <hr style="border: 1px dashed #cbd5e0; margin: 20px 0;">
    <div style="text-align: center; color: #718096; font-size: 12px; margin: 10px 0;">
      История предыдущих запросов
    </div>
  `;
  frag.appendChild(divider);

  container.appendChild(frag);

  // Прокрутка к низу
  container.scrollTop = container.scrollHeight;
}


// Добавляем кнопку загрузки истории в интерфейс
document.addEventListener('DOMContentLoaded', () => {
    // Добавляем кнопку в sidebar
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        const historyButton = document.createElement('button');
        historyButton.innerHTML = '📜 Загрузить историю';
        historyButton.style.cssText = `
            width: 100%;
            padding: 10px;
            margin-top: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
        `;
        historyButton.onclick = loadUserHistory;
        sidebar.appendChild(historyButton);
    }
    
    // Автоматически загружаем историю при загрузке страницы
    setTimeout(loadUserHistory, 1000);
    
    // Заменяем стандартную функцию отправки
    window.sendQuestion = sendQuestionWithAuth;
    
    // Обновляем обработчик кнопки
    document.getElementById('sendBtn').removeEventListener('click', sendQuestion);
    document.getElementById('sendBtn').addEventListener('click', sendQuestionWithAuth);
});
</script>

<script>
// Если это desktop приложение, можно получить больше информации
if (typeof window.electronAPI !== 'undefined') {
    // Electron
    window.electronAPI.getSystemInfo().then(info => {
        window.systemInfo = {
            domain: info.domain,
            computer_name: info.hostname,
            username: info.username,
            os: info.os
        };
    });
} else if (typeof window.__TAURI__ !== 'undefined') {
    // Tauri
    window.__TAURI__.invoke('get_system_info').then(info => {
        window.systemInfo = info;
    });
}

// Для Python-based GUI (через pywebview или подобные)
if (typeof window.pywebview !== 'undefined') {
    window.pywebview.api.get_system_info().then(info => {
        window.systemInfo = info;
    });
}
</script>
    <script>
        // Функции для модального окна
        function toggleFeedbackModal() {
            const modal = document.getElementById('feedbackModal');
            modal.classList.toggle('active');
            
            // Сброс формы при закрытии
            if (!modal.classList.contains('active')) {
                resetForms();
            }
        }
        
        function switchTab(tabName) {
            // Переключение вкладок
            document.querySelectorAll('.modal-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.modal-content').forEach(content => {
                content.classList.remove('active');
            });
            
            event.target.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        }
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                const fileInfo = document.getElementById('fileInfo');
                fileInfo.innerHTML = `
                    📄 ${file.name}<br>
                    <small>Размер: ${(file.size / 1024).toFixed(2)} KB</small>
                `;
            }
        }
        
        async function submitTrainingData(event) {
            event.preventDefault();
            
            const data = {
                title: document.getElementById('trainingTitle').value,
                text: document.getElementById('trainingText').value,
                category: document.getElementById('trainingCategory').value,
                type: 'training'
            };
            
            try {
                const response = await fetch('/api/training', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    alert('Материалы успешно отправлены!');
                    toggleFeedbackModal();
                } else {
                    alert('Произошла ошибка при отправке');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Произошла ошибка при отправке');
            }
        }
        
        async function submitInaccuracy(event) {
            event.preventDefault();
            
            const data = {
                inaccuracy: document.getElementById('inaccuracyText').value,
                correct_info: document.getElementById('correctInfo').value,
                type: 'inaccuracy'
            };
            
            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    alert('Спасибо за обратную связь!');
                    toggleFeedbackModal();
                } else {
                    alert('Произошла ошибка при отправке');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Произошла ошибка при отправке');
            }
        }
        
        async function submitFile(event) {
            event.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Пожалуйста, выберите файл');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('comment', document.getElementById('fileComment').value);
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    alert('Файл успешно загружен!');
                    toggleFeedbackModal();
                } else {
                    alert('Произошла ошибка при загрузке');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Произошла ошибка при загрузке');
            }
        }
        
        function resetForms() {
            document.querySelectorAll('form').forEach(form => form.reset());
            document.getElementById('fileInfo').innerHTML = `
                📁 Нажмите для выбора файла<br>
                <small>Поддерживаются: PDF, DOCX, TXT, MD</small>
            `;
        }
        
        // Закрытие модального окна при клике на оверлей
        document.getElementById('feedbackModal').addEventListener('click', function(event) {
            if (event.target === this) {
                toggleFeedbackModal();
            }
        });
    </script>
    <script>
        // Функция переключения темы
        function toggleTheme() {
            const isDark = document.getElementById('themeToggle').checked;
            if (isDark) {
                document.body.classList.add('dark-theme');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark-theme');
                localStorage.setItem('theme', 'light');
            }
        }
        
        // Загрузка сохраненной темы
        document.addEventListener('DOMContentLoaded', () => {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {
                document.body.classList.add('dark-theme');
                document.getElementById('themeToggle').checked = true;
            }
        });
        
        // Функции для очистки чата
        function confirmClearChat() {
            document.getElementById('confirmModal').classList.add('active');
        }
        
        function closeConfirmModal() {
            document.getElementById('confirmModal').classList.remove('active');
        }
        
        async function clearChat() {
            try {
                const userInfo = getUserIdentifiers();
                const response = await fetch('/api/clear-chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-User-Domain': userInfo.domain,
                        'X-Computer-Name': userInfo.computer_name
                    },
                    body: JSON.stringify({
                        session_id: userInfo.computer_name
                    })
                });
                
                if (response.ok) {
                    // Очищаем UI
                    const chatContainer = document.getElementById('chatContainer');
                    chatContainer.innerHTML = `
                        <div class="bot-message message">
                            <div class="bubble" id="welcome"></div>
                        </div>
                    `;
                    
                    // Восстанавливаем приветственное сообщение
                    const welcomeMd = `
👋 Здравствуйте!

Я интеллектуальный помощник по документации **Falcon**.

**Чем могу помочь:**
- Найти нужную информацию в документации
- Объяснить сложные концепции
- Предоставить пошаговые инструкции

Задайте ваш вопрос — и я найду наиболее релевантную информацию!
                    `.trim();
                    
                    document.getElementById('welcome').innerHTML = renderMarkdownSafe(welcomeMd);
                    
                    // Сбрасываем статистику
                    window.queryCount = 0;
                    window.totalTime = 0;
                    document.getElementById('queryCount').textContent = '0 запросов';
                    document.getElementById('avgTime').textContent = '0.0с среднее время';
                    
                    closeConfirmModal();
                } else {
                    alert('Произошла ошибка при очистке чата');
                }
            } catch (error) {
                console.error('Error clearing chat:', error);
                alert('Произошла ошибка при очистке чата');
            }
        }
        
        // Весь остальной JavaScript из оригинального кода...
        // [Здесь должен быть весь JavaScript из оригинала]
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info",
        access_log=False,
    )