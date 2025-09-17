# enhanced_analytics.py - Добавить в app.py вместо старого класса UsageAnalytics

import re
import json
import hashlib
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Стоп-слова для русского языка
RUSSIAN_STOP_WORDS = {
    'и', 'а', 'в', 'во', 'на', 'по', 'из', 'к', 'с', 'со', 'от', 'до', 'за', 'над', 'под', 'при',
    'о', 'об', 'для', 'как', 'что', 'это', 'или', 'если', 'то', 'же', 'ли', 'так', 'также',
    'у', 'бы', 'чтобы', 'где', 'когда', 'после', 'перед', 'можно', 'нужно', 'еще', 'уже',
    'все', 'всё', 'весь', 'вся', 'они', 'оно', 'она', 'он', 'мы', 'вы', 'ты', 'я',
    'который', 'которая', 'которые', 'такой', 'только', 'более', 'менее', 'через'
}

class EnhancedAnalytics:
    """Улучшенная аналитика с фразами и БД логированием"""
    
    def __init__(self, db_pool: SimpleConnectionPool):
        self.db_pool = db_pool
        self.queries = deque(maxlen=5000)
        self.feedback = deque(maxlen=2000)
        self.response_times = deque(maxlen=5000)
        self.popular_topics = defaultdict(float)  # Фразы с весами
        self.satisfaction_scores = deque(maxlen=5000)
        
        # Инициализация таблиц БД
        self._init_db_tables()
    
    def _init_db_tables(self):
        """Создание таблиц для логирования"""
        conn = self.db_pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                # Таблица пользователей
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id bigserial PRIMARY KEY,
                        user_hash text UNIQUE NOT NULL,
                        domain text,
                        computer_name text,
                        first_seen timestamptz DEFAULT now(),
                        last_seen timestamptz DEFAULT now(),
                        query_count int DEFAULT 0,
                        avg_satisfaction real,
                        meta jsonb DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS users_hash_idx ON users(user_hash);
                    CREATE INDEX IF NOT EXISTS users_domain_idx ON users(domain);
                """)
                
                # Таблица сессий
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id bigserial PRIMARY KEY,
                        session_id text UNIQUE NOT NULL,
                        user_id bigint REFERENCES users(id),
                        started_at timestamptz DEFAULT now(),
                        ended_at timestamptz,
                        message_count int DEFAULT 0,
                        avg_confidence real,
                        avg_grounding real,
                        ip_address inet,
                        user_agent text
                    );
                    CREATE INDEX IF NOT EXISTS sessions_user_idx ON chat_sessions(user_id);
                    CREATE INDEX IF NOT EXISTS sessions_sid_idx ON chat_sessions(session_id);
                """)
                
                # Основная таблица Q&A логов
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS qa_logs (
                        id bigserial PRIMARY KEY,
                        created_at timestamptz DEFAULT now(),
                        session_id text,
                        user_id bigint REFERENCES users(id),
                        question text NOT NULL,
                        answer text NOT NULL,
                        citations jsonb DEFAULT '[]',
                        chunks jsonb DEFAULT '[]',
                        processing_time real,
                        question_type text,
                        cached boolean DEFAULT false,
                        confidence_score real,
                        grounding_score real,
                        model text,
                        search_methods text[],
                        top_k int,
                        use_llm boolean DEFAULT true,
                        use_cot boolean DEFAULT true,
                        ip_address inet,
                        user_agent text,
                        feedback_rating int,
                        feedback_comment text
                    );
                    
                    CREATE INDEX IF NOT EXISTS qa_created_idx ON qa_logs(created_at DESC);
                    CREATE INDEX IF NOT EXISTS qa_session_idx ON qa_logs(session_id);
                    CREATE INDEX IF NOT EXISTS qa_user_idx ON qa_logs(user_id);
                    CREATE INDEX IF NOT EXISTS qa_type_idx ON qa_logs(question_type);
                    CREATE INDEX IF NOT EXISTS qa_confidence_idx ON qa_logs(confidence_score);
                    
                    -- Полнотекстовые индексы для поиска
                    CREATE EXTENSION IF NOT EXISTS pg_trgm;
                    CREATE INDEX IF NOT EXISTS qa_question_trgm ON qa_logs USING gin(question gin_trgm_ops);
                    CREATE INDEX IF NOT EXISTS qa_answer_trgm ON qa_logs USING gin(answer gin_trgm_ops);
                """)
                
                # Таблица популярных тем
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS popular_topics (
                        id bigserial PRIMARY KEY,
                        topic text UNIQUE NOT NULL,
                        hits int DEFAULT 1,
                        last_used timestamptz DEFAULT now(),
                        avg_confidence real,
                        avg_processing_time double precision,
                        total_weight double precision DEFAULT 0
                    );

                    CREATE INDEX IF NOT EXISTS topics_hits_idx
                    ON popular_topics (hits DESC);

                    CREATE INDEX IF NOT EXISTS topics_used_idx
                    ON popular_topics (last_used DESC);

                    CREATE INDEX IF NOT EXISTS topics_weight_used_idx
                    ON popular_topics (total_weight DESC, last_used DESC);
                """)
                
                logger.info("✓ Таблицы аналитики инициализированы")
                
        finally:
            self.db_pool.putconn(conn)
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Извлечение фраз (биграммы и триграммы) из текста"""
        # Токенизация с учетом русского языка
        tokens = re.findall(r'\b[а-яёА-ЯЁa-zA-Z0-9]+\b', text.lower())
        
        # Фильтрация стоп-слов
        filtered_tokens = [t for t in tokens if t not in RUSSIAN_STOP_WORDS and len(t) > 2]
        
        phrases = []
        
        # Триграммы (3 слова)
        for i in range(len(filtered_tokens) - 2):
            phrase = ' '.join(filtered_tokens[i:i+3])
            phrases.append((phrase, 1.5))  # Больший вес для триграмм
        
        # Биграммы (2 слова)
        for i in range(len(filtered_tokens) - 1):
            phrase = ' '.join(filtered_tokens[i:i+2])
            phrases.append((phrase, 1.0))
        
        # Важные униграммы (одиночные слова > 5 символов)
        for token in filtered_tokens:
            if len(token) > 5:
                phrases.append((token, 0.5))  # Меньший вес для одиночных слов
        
        return phrases
    
    def get_or_create_user(self, domain: str = None, computer_name: str = None, 
                          ip: str = None, user_agent: str = None, conn=None) -> int:
        """Получение или создание пользователя по идентификаторам"""
        # Создаем уникальный хеш пользователя
        user_str = f"{domain or 'unknown'}_{computer_name or 'unknown'}"
        user_hash = hashlib.md5(user_str.encode()).hexdigest()
        owned = False
        if conn is None:
            conn = self.db_pool.getconn()
            owned = True
        try:
            with conn.cursor() as cur:
                # Пытаемся найти существующего пользователя
                cur.execute("""
                    UPDATE users 
                    SET last_seen = now(), query_count = query_count + 1
                    WHERE user_hash = %s
                    RETURNING id
                """, (user_hash,))
                
                result = cur.fetchone()
                if result:
                    return result[0]
                
                # Создаем нового пользователя
                cur.execute("""
                    INSERT INTO users (user_hash, domain, computer_name, meta)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (
                    user_hash, 
                    domain, 
                    computer_name,
                    json.dumps({
                        'ip_addresses': [ip] if ip else [],
                        'user_agents': [user_agent] if user_agent else []
                    })
                ))
                
                return cur.fetchone()[0]
                
        finally:
            if owned:
                self.db_pool.putconn(conn)
    
    def log_query(self, query: str, response_time: float, chunks_found: int):
        """Логирование запроса с извлечением фраз"""
        # В памяти для быстрой статистики
        self.queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_time": response_time,
            "chunks_found": chunks_found
        })
        self.response_times.append(response_time)
        
        # Извлекаем и считаем фразы
        phrases = self._extract_phrases(query)
        for phrase, weight in phrases:
            self.popular_topics[phrase] += weight
        
        # Обновляем популярные темы в БД
        conn = self.db_pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                for phrase, weight in phrases:
                    cur.execute("""
                        INSERT INTO popular_topics (topic, hits, total_weight, avg_processing_time)
                        VALUES (%s, 1, %s, %s)
                            ON CONFLICT (topic) DO UPDATE 
                            SET hits = popular_topics.hits + 1,
                                total_weight = popular_topics.total_weight + EXCLUDED.total_weight,
                                last_used = now(),
                                avg_processing_time =
                                    (COALESCE(popular_topics.avg_processing_time, 0) * popular_topics.hits + %s)
                                    / NULLIF(popular_topics.hits + 1, 0)
                        """, (phrase, float(weight), float(response_time), float(response_time)))
        except Exception as e:
            logger.error(f"Ошибка обновления популярных тем: {e}")
        finally:
            self.db_pool.putconn(conn)
    
    
    def log_qa_complete(self, record: Dict):
        """Полное логирование Q&A в БД"""
        conn = self.db_pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cs = record.get('confidence_score')
                gs = record.get('grounding_score')
                pt = record.get('processing_time')

                cs = float(cs) if cs is not None else None
                gs = float(gs) if gs is not None else None
                pt = float(pt) if pt is not None else None
                # Получаем или создаем пользователя
                user_id = None
                if record.get('domain') or record.get('computer_name'):
                    user_id = self.get_or_create_user(
                        domain=record.get('domain'),
                        computer_name=record.get('computer_name'),
                        ip=record.get('ip_address'),
                        user_agent=record.get('user_agent'),
                        conn=conn,
                    )
                
                # Обновляем сессию если есть
                if record.get('session_id'):
                    cur.execute("""
                        INSERT INTO chat_sessions (session_id, user_id, ip_address, user_agent)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE
                        SET message_count = chat_sessions.message_count + 1,
                            avg_confidence =
                            CASE WHEN %s::float8 IS NOT NULL THEN
                                (COALESCE(chat_sessions.avg_confidence, 0)::float8 * chat_sessions.message_count
                                + %s::float8) / (chat_sessions.message_count + 1)
                            ELSE chat_sessions.avg_confidence END,
                            avg_grounding =
                            CASE WHEN %s::float8 IS NOT NULL THEN
                                (COALESCE(chat_sessions.avg_grounding, 0)::float8 * chat_sessions.message_count
                                + %s::float8) / (chat_sessions.message_count + 1)
                            ELSE chat_sessions.avg_grounding END
                    """, (
                        record['session_id'], user_id, record.get('ip_address'), record.get('user_agent'),
                        cs, cs, gs, gs
                    ))
                    
                safe = dict(record)
                safe.update({
                    'user_id': user_id,
                    'citations': json.dumps(record.get('citations', []), ensure_ascii=False),
                    'chunks': json.dumps(record.get('chunks', []), ensure_ascii=False),
                    'ip_address': record.get('ip_address'),
                    'processing_time': pt,          # ← только python float
                    'confidence_score': cs,         # ← только python float
                    'grounding_score': gs           # ← только python float
                })
                safe['public_id'] = record.get('qa_id')
                
                # Вставляем основную запись
                cur.execute("""
                    INSERT INTO qa_logs (
                        session_id, user_id, question, answer, citations, chunks,
                        processing_time, question_type, cached, confidence_score,
                        grounding_score, model, search_methods, top_k, use_llm, use_cot,
                        ip_address, user_agent, public_id
                    )
                    VALUES (
                        %(session_id)s, %(user_id)s, %(question)s, %(answer)s, 
                        %(citations)s::jsonb, %(chunks)s::jsonb,
                        %(processing_time)s, %(question_type)s, %(cached)s, 
                        %(confidence_score)s, %(grounding_score)s, %(model)s, 
                        %(search_methods)s, %(top_k)s, %(use_llm)s, %(use_cot)s,
                        %(ip_address)s::inet, %(user_agent)s, %(public_id)s
                    )
                    RETURNING id
                """, safe)
                
                qa_id = cur.fetchone()[0]
                logger.debug(f"Q&A логирован: ID={qa_id}")
                
        except Exception as e:
            logger.error(f"Ошибка логирования Q&A: {e}")
        finally:
            self.db_pool.putconn(conn)
    
    def log_feedback(self, rating: int, comment: str = None, qa_id: Optional[str] = None):
        """Логирование обратной связи. Обновляем запись по public_id (строковый),
        а при его отсутствии/несовпадении пробуем по числовому id (бэкомпат)."""
        self.feedback.append({
            "timestamp": datetime.now().isoformat(),
            "rating": rating,
            "comment": comment
        })
        self.satisfaction_scores.append(rating)

        if not qa_id:
            return  # ничего в БД не обновляем без идентификатора

        conn = self.db_pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                # 1) основной путь — по public_id
                cur.execute("""
                    UPDATE qa_logs
                    SET feedback_rating = %s,
                        feedback_comment = %s
                    WHERE public_id = %s
                """, (rating, comment, qa_id))

                # 2) бэкомпат: если запись не нашлась и qa_id похоже на число — пробуем по id
                if cur.rowcount == 0 and str(qa_id).isdigit():
                    cur.execute("""
                        UPDATE qa_logs
                        SET feedback_rating = %s,
                            feedback_comment = %s
                        WHERE id = %s::bigint
                    """, (rating, comment, qa_id))
        finally:
            self.db_pool.putconn(conn)

    
    def get_statistics(self) -> Dict:
        # 1) читаем детерминированный топ из БД
        conn = self.db_pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT topic, total_weight
                    FROM popular_topics
                    ORDER BY total_weight DESC, last_used DESC
                    LIMIT 15
                """)
                db_topics = cur.fetchall()
        finally:
            self.db_pool.putconn(conn)

        # 2) мягко домешиваем RAM-веса (если успели накопиться)
        combined = defaultdict(float)
        for t, w in db_topics:
            combined[t] += float(w or 0)
        for t, w in self.popular_topics.items():
            combined[t] += float(w or 0)

        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:15]

        return {
            "total_queries": len(self.queries),
            "avg_response_time": float(np.mean(self.response_times)) if self.response_times else 0.0,
            "median_response_time": float(np.median(self.response_times)) if self.response_times else 0.0,
            "avg_satisfaction": float(np.mean(self.satisfaction_scores)) if self.satisfaction_scores else 0.0,
            # новый стабильный формат (массив, порядок гарантирован)
            "popular_topics": [{"topic": t, "score": float(s)} for t, s in top],
            "recent_queries": list(self.queries)[-10:][::-1]
        }

    
    def get_user_history(self, domain: str, computer_name: str, limit: int = 50) -> List[Dict]:
        """Получение истории пользователя"""
        user_hash = hashlib.md5(f"{domain}_{computer_name}".encode()).hexdigest()
        
        conn = self.db_pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        q.created_at, q.question, q.answer, 
                        q.confidence_score, q.grounding_score,
                        q.feedback_rating, q.processing_time
                    FROM qa_logs q
                    JOIN users u ON q.user_id = u.id
                    WHERE u.user_hash = %s
                    ORDER BY q.created_at DESC
                    LIMIT %s
                """, (user_hash, limit))
                
                return [
                    {
                        'timestamp': row[0].isoformat(),
                        'question': row[1],
                        'answer': row[2],
                        'confidence': row[3],
                        'grounding': row[4],
                        'rating': row[5],
                        'time': row[6]
                    }
                    for row in cur.fetchall()
                ]
        finally:
            self.db_pool.putconn(conn)