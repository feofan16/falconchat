# C:\F-ChatAI\search_windows.py
from __future__ import annotations
import os, sys, psycopg2
from typing import Dict, Union, Optional, Tuple, List
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass, field
import logging
import re, unicodedata
from scripts.utils.labels import gost_norm  # для inline ГОСТ
import json, pathlib
import math
import threading
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RF = True
except Exception:
    rf_process = None
    rf_fuzz = None
    HAVE_RF = False


GLOSSARY_PATH = pathlib.Path(__file__).with_name("glossary.json")

SPACE_MAP = {
    0x00A0: ' ', 0x1680: ' ', 0x2000: ' ', 0x2001: ' ', 0x2002: ' ',
    0x2003: ' ', 0x2004: ' ', 0x2005: ' ', 0x2006: ' ', 0x2007: ' ',
    0x2008: ' ', 0x2009: ' ', 0x200A: ' ', 0x202F: ' ', 0x205F: ' ',
    0x3000: ' ',
}
INVISIBLES = {0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD}

def _load_glossary():
    try:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# Windows: кодировка клиента
if sys.platform == "win32":
    os.environ['PGCLIENTENCODING'] = 'UTF8'
    sys.stdout.reconfigure(encoding='utf-8')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт FlagReranker для русского языка
try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbedding не установлен. Установите: pip install FlagEmbedding")

# Конфигурация БД
DB_CONFIG = {
    "dbname": "rag",
    "user": "rag",
    "password": "rag",
    "host": "127.0.0.1",
    "port": 5432,
    "options": "-c client_encoding=UTF8"
}

    # 2) Универсальный фильтр терминов
REPLACE = {
    r"\bсторно\b": "отмена выдачи",
    r"\bсторнировать\b": "отменить выдачу",
    r"\bавтокаскад\b": "автоматически",
}



_SEARCHER_SINGLETON: Optional["AdvancedHybridSearch"] = None
_SEARCHER_LOCK = threading.Lock()

def get_searcher() -> "AdvancedHybridSearch":
    global _SEARCHER_SINGLETON
    if _SEARCHER_SINGLETON is None:
        with _SEARCHER_LOCK:
            if _SEARCHER_SINGLETON is None:
                _SEARCHER_SINGLETON = AdvancedHybridSearch(DB_CONFIG)
    return _SEARCHER_SINGLETON


# ====== LABELS / FILTERS / INLINE (module-level) ======
INLINE_RE = re.compile(r'(?i)\b(окпд|gost|гост|ost|ту|tu)\s*:\s*([^\s,;]+)')

def _parse_inline_filters_text(q: str) -> List[Dict]:
    out = []
    m = {'окпд':'okpd','gost':'gost','гост':'gost','ost':'ost','ту':'tu','tu':'tu'}
    for ns_raw, val in INLINE_RE.findall(q or ''):
        ns = m[ns_raw.lower()]
        if ns == 'okpd':
            out.append({'ns': ns, 'prefix': val})
        elif ns == 'gost':
            code = gost_norm('ГОСТ ' + val) or f'ГОСТ {val}'
            out.append({'ns': ns, 'code': code})
        else:
            out.append({'ns': ns, 'code': val})
    return out

def _labels_where_sql(filters: Optional[List[Dict]]):
    clauses, params = [], []
    for f in (filters or []):
        ns = f.get('ns'); code = f.get('code'); prefix = f.get('prefix')
        if not ns:
            continue
        if code:
            clauses.append("""EXISTS (
                SELECT 1 FROM docs_labels dl 
                JOIN labels l ON l.id = dl.label_id 
                WHERE dl.doc_id = docs.id AND l.ns = %s AND l.code = %s
            )""")
            params += [ns, code]
        elif prefix:
            clauses.append("""EXISTS (
                SELECT 1 FROM docs_labels dl 
                JOIN labels l ON l.id = dl.label_id 
                WHERE dl.doc_id = docs.id AND l.ns = %s AND l.path @> %s
            )""")
            params += [ns, [prefix]]  # список => text[] для psycopg2
    sql = (" AND " + " AND ".join(clauses)) if clauses else ""
    return sql, params

def _fetch_labels_for_docs(cur, doc_ids: List[int]) -> Dict[int, List[Dict]]:
    if not doc_ids:
        return {}
    cur.execute("""
        SELECT doc_id, labels
        FROM v_doc_labels
        WHERE doc_id = ANY(%s)
    """, (doc_ids,))
    return {int(did): (labels or []) for did, labels in cur.fetchall()}

def _boost_by_filters(results, wanted: Optional[List[Dict]]):
    if not results or not wanted:
        return
    need = {(w.get('ns'), w.get('code') or w.get('prefix')) for w in wanted if w.get('ns')}
    for r in results:
        tags = (r.meta or {}).get('labels') or []
        boost = 1.0
        for t in tags:
            if not t:
                continue
            crumbs = tuple(t.get('path') or [])
            key = (t.get('ns'), t.get('code'))
            if key in need or any((t.get('ns'), c) in need for c in crumbs):
                boost *= 1.05
        r.score *= boost


@dataclass
class SearchResult:
    """Результат поиска с расширенными метаданными"""
    id: int
    book: str
    section: str
    page: int
    text: str
    meta: Dict = field(default_factory=dict) 
    score: float = 0.0 
    vector_score: float = 0.0
    fulltext_score: float = 0.0
    rerank_score: float = 0.0
    importance: float = 1.0
    trgm_score: float = 0.0
    fused_score: float = 0.0

class AdvancedHybridSearch:
    """
    Улучшенный гибридный поиск с BGE-reranker для русского языка
    """
    def __init__(self, dsn_or_cfg: Union[str, dict] = DB_CONFIG):
        self.dsn_or_cfg = dsn_or_cfg
        self._domain_vocab = None
        self._domain_vocab_list = None
        self._cache_lock = threading.RLock()

        # Загружаем модели
        logger.info("Загрузка моделей...")
        self.embedder = SentenceTransformer("BAAI/bge-m3")
        
        # BGE-reranker v2-m3 - отлично работает с русским языком!
        if RERANKER_AVAILABLE:
            try:
                self.reranker = FlagReranker(
                    "BAAI/bge-reranker-v2-m3",
                    use_fp16=False  # Для CPU ставим False, для GPU можно True
                )
                self.use_reranking = True
                logger.info("✓ BGE-reranker v2-m3 загружен")
            except Exception as e:
                self.reranker = None
                self.use_reranking = False
                logger.warning(f"Не удалось загрузить reranker: {e}")
        else:
            self.reranker = None
            self.use_reranking = False
            logger.warning("Reranker отключен. Установите FlagEmbedding для улучшения качества.")
        
        # Кэш для эмбеддингов запросов
        self.query_cache = {}
        self.glossary = _load_glossary()
        # ОБНОВЛЕННЫЕ веса для русского контента с BGE-reranker
        self.vector_weight = 0.35
        self.fulltext_weight = 0.25
        self.trgm_weight    = 0.10
        self.rerank_weight  = 0.30
        # RRF и анти-шум настройки
        self.rrf_enable = True
        self.rrf_k = 60
        self.rrf_weights = {'vec': 0.50, 'fts': 0.35, 'trgm': 0.15}
        self.rrf_two_signals_gate = True
        self.rrf_or_rank_gate = {'vec': 20, 'fts': 15}   # ИЛИ-порог по рангам
        self.rrf_trgm_depth = 60                         # глубина списка trgm для RRF
        self.rrf_or_gate_pct = 0.15                      # динамика: топ-15% списка

        # динамические пороги (будут подстраиваться в _tune_trgm_thresholds)
        self.trgm_min_sim = 0.33
        self.trgm_min_overlap = 0.45

        # кеши токенизации: запросов и «section + head(text)» для документов
        self._tok_cache = {
            'query': {},     # key: normalized_query -> set(tokens)
            'doc_head': {}   # key: doc_id -> set(tokens)
        }

        # последние ранговые листы (для fail-safe доклейки)
        self._last_vec_list = []
        self._last_fts_list = []
        self._last_trgm_list = []
        # ранк-мапы для fail-safe доклейки
        self._last_vec_rankmap = {}
        self._last_fts_rankmap = {}

    # --------- токены / секция+голова ---------
    def _tokenize(self, s: str) -> set[str]:
        s = (s or '').lower().replace('ё', 'е')
        return set(re.findall(r'[а-яёa-z0-9]{3,}', s))

    def _tokenize_cached_query(self, q: str) -> set[str]:
        qn = self._normalize_ru(q)
        c = self._tok_cache['query'].get(qn)
        if c is None:
            c = self._tokenize(qn)
        with self._cache_lock:
            self._tok_cache['query'][qn] = c
        return c

    def _doc_head_text(self, r, head_chars=420) -> str:
        head = (r.meta or {}).get('_orig_head') or (r.text or '')
        return ((r.section or '') + ' ' + head[:head_chars]).lower()

    def _tokenize_cached_doc_head(self, r, head_chars=420) -> set[str]:
        if r.id in self._tok_cache['doc_head']:
            return self._tok_cache['doc_head'][r.id]
        toks = self._tokenize(self._doc_head_text(r, head_chars=head_chars))
        if not ((r.meta or {}).get('_ctx_augmented')):
            with self._cache_lock:
                self._tok_cache['doc_head'][r.id] = toks
        return toks
    

    def _overlap_ratio(self, qtok: set[str], r, head_chars=420) -> float:
        dtok = self._tokenize_cached_doc_head(r, head_chars=head_chars)
        if not qtok:
            return 0.0

        # Без rapidfuzz — старое поведение (жёсткое пересечение)
        if not HAVE_RF:
            return len(qtok & dtok) / max(1, len(qtok))

        hits = 0
        dtok_list = list(dtok)
        for t in qtok:
            if t in dtok:
                hits += 1
                continue
            if len(t) < 4:
                continue
            # Учитываем опечатки: «моркировки» ≈ «маркировки»
            m = rf_process.extractOne(t, dtok_list, scorer=rf_fuzz.WRatio)
            thr = 80
            L = len(t)
            # длинные слова совпадают реже случайно — можно чуть поднять
            if L >= 10: thr = 85
            if L >= 14: thr = 88
            # если у нас глобально включён "двухсигнальный" гейт RRF — можно быть на 2 пункта мягче
            if getattr(self, "rrf_two_signals_gate", False):
                thr -= 2

            if m and m[1] >= thr:
                hits += 1

        return hits / max(1, len(qtok))


    def _should_use_trgm(self, q: str) -> bool:
        qn = self._normalize_ru(q)
        toks = self._tokenize(qn)
        # очень короткие/однословные — трёхграммы обычно шумят
        return not (len(qn) < 6 or len(toks) <= 1)
    
    def _safe_trim_for_reranker(self, items: list, want: int) -> list:
        if len(items) >= want:
            return items[:want]
        seen = {r.id for r in items}
        tail = []
        for lst in (self._last_vec_list, self._last_fts_list):
            for r in lst:
                if r.id not in seen:
                    tail.append(r); seen.add(r.id)
                if len(items) + len(tail) >= want:
                    break
        # переранжируем хвост по мин(ранг_vec, ранг_fts)
        INF = 10**9
        def base_rank(x):
            rv = self._last_vec_rankmap.get(x.id, INF)
            rf = self._last_fts_rankmap.get(x.id, INF)
            return min(rv, rf)
        tail.sort(key=lambda r: (base_rank(r), r.id))
        need = max(0, want - len(items))
        return items + tail[:need]

    def _tune_trgm_thresholds(self, q: str):
        toks = self._tokenize_cached_query(q)
        t = len(toks)
        if t <= 2:        # короткий запрос → жестче
            self.trgm_min_sim = 0.38
            self.trgm_min_overlap = 0.50
        elif t <= 4:
            self.trgm_min_sim = 0.35
            self.trgm_min_overlap = 0.50
        else:
            self.trgm_min_sim = 0.33
            self.trgm_min_overlap = 0.45

    # --------- RRF и сбор рангов ---------
    def _rrf_fuse(self, ranks: dict[int, dict[str, int]], k: int, weights: dict[str, float]) -> dict[int, float]:
        out = {}
        for doc_id, rm in ranks.items():
            s = 0.0
            for m, r in rm.items():
                if r is None or r == float('inf'):
                    continue
                s += weights.get(m, 0.0) / (k + r)
            out[doc_id] = s
        return out

    def _dedup_rank(self, lst, key):
        seen, out = set(), []
        for r in sorted(lst, key=key, reverse=True):
            if r.id not in seen:
                out.append(r); seen.add(r.id)
        return out

    def _rrf_merge(self,
                method_lists: dict[str, list],
                *,
                query: str,
                id_to_result: dict[int, 'SearchResult'],
                want_pool: int) -> list:
        """
        Берёт «чистые» ранговые листы (vector/fulltext/trgm), применяет гейты и Weighted-RRF.
        Возвращает отсортированный по fused-скор спискок кандидатов (без жесткой обрезки).
        """
        vec_list = method_lists.get('vector') or []
        fts_list = method_lists.get('fulltext') or []
        trgm_list = method_lists.get('trgm') or []

        # динамические пороги: макс(абсолютный, перцентиль)
        vec_gate_dyn = max(self.rrf_or_rank_gate['vec'], math.ceil(self.rrf_or_gate_pct * len(vec_list)) or 0)
        fts_gate_dyn = max(self.rrf_or_rank_gate['fts'], math.ceil(self.rrf_or_gate_pct * len(fts_list)) or 0)

        # 1) построим ранги (1-based) по каждому листу
        ranks: dict[int, dict[str, int]] = {}
        def _acc(lst, label):
            for pos, r in enumerate(lst, 1):
                ranks.setdefault(r.id, {})
                if pos < ranks[r.id].get(label, float('inf')):
                    ranks[r.id][label] = pos

        _acc(vec_list, 'vec')
        _acc(fts_list, 'fts')
        _acc(trgm_list, 'trgm')

        # 2) гейты
        qtok = self._tokenize_cached_query(query)

        def pass_gates(doc_id: int, rm: dict[str, int]) -> tuple[bool, float, bool]:
            """
            Возвращает (ok, overlap, two_signals):
            ok          — прошёл ли документ гейты
            overlap     — метрика пересечения токенов запроса с документом (для логов)
            two_signals — было ли >=2 сигналов (vec/fts/trgm), для метрик
            """
            # 0) Сначала обрабатываем "чисто trigram"-кейс: rm содержит только 'trgm'
            only_trgm = ('trgm' in rm) and ('vec' not in rm) and ('fts' not in rm)
            if only_trgm:
                r = id_to_result.get(doc_id)
                if not r:
                    return False, 0.0, False

                tscore = float(getattr(r, 'trgm_score', 0.0))
                ov = self._overlap_ratio(qtok, r, head_chars=420)

                # Адаптивные пороги для коротких запросов (1–2 токена)
                min_sim = self.trgm_min_sim
                min_ovl = self.trgm_min_overlap
                if len(qtok) <= 2:
                    # чуть мягче, чтобы ловить «моркировки»≈«маркировки» и т.п.
                    min_sim = max(0.34, min_sim - 0.02)
                    min_ovl = max(0.50, min_ovl - 0.05)

                ok = (tscore >= min_sim) and (ov >= min_ovl)
                return ok, ov, False

            # 1) Общий путь: консенсус по сигналам или попадание по порогам рангов vec/fts
            signals = len(rm)
            vec_ok = rm.get('vec', float('inf')) <= vec_gate_dyn
            fts_ok = rm.get('fts', float('inf')) <= fts_gate_dyn
            consensus_ok = (signals >= 2) if self.rrf_two_signals_gate else (signals >= 1)
            pass_basic = consensus_ok or vec_ok or fts_ok

            if not pass_basic:
                return False, 0.0, (signals >= 2)

            # 2) Для не-чисто trgm считаем overlap лишь как метрику (на гейт не влияет)
            r = id_to_result.get(doc_id)
            ov = self._overlap_ratio(qtok, r, head_chars=420) if r else 0.0
            return True, ov, (signals >= 2)
            
        # 3) базовый проход гейтов
        accepted, overlaps, two_sig_marks = {}, [], []
        for doc_id, rm in ranks.items():
            ok, ov, two = pass_gates(doc_id, rm)
            if ok:
                accepted[doc_id] = rm
                overlaps.append(ov)
            two_sig_marks.append(1 if two else 0)

        # fail-safe: если кандидатов мало — ослабим консенсус
        if len(accepted) < max(30, want_pool // 2):
            bak_two = self.rrf_two_signals_gate
            self.rrf_two_signals_gate = False
            accepted, overlaps = {}, []

            # чуть ослабим динамические пороги
            vec_gate2 = vec_gate_dyn + 5
            fts_gate2 = fts_gate_dyn + 5

            for doc_id, rm in ranks.items():
                vec_ok = rm.get('vec', float('inf')) <= vec_gate2
                fts_ok = rm.get('fts', float('inf')) <= fts_gate2
                signals = len(rm)
                pass_basic = (signals >= 1) or vec_ok or fts_ok
                if not pass_basic:
                    continue

                only_trgm = (('trgm' in rm) and ('vec' not in rm) and ('fts' not in rm))
                if only_trgm:
                    r = id_to_result.get(doc_id)
                    if not r:
                        continue
                    tscore = float(getattr(r, 'trgm_score', 0.0))
                    ov = self._overlap_ratio(qtok, r, head_chars=420)
                    if (tscore < self.trgm_min_sim) or (ov < self.trgm_min_overlap):
                        continue
                    accepted[doc_id] = rm
                    overlaps.append(ov)
                else:
                    r = id_to_result.get(doc_id)
                    ov = self._overlap_ratio(qtok, r, head_chars=420) if r else 0.0
                    accepted[doc_id] = rm
                    overlaps.append(ov)

            self.rrf_two_signals_gate = bak_two

        # 4) RRF-слияние
        fused = self._rrf_fuse(accepted, k=self.rrf_k, weights=self.rrf_weights)
        # метрики в логи
        two_ratio = (sum(two_sig_marks) / max(1, len(two_sig_marks))) if two_sig_marks else 0.0
        avg_ov = (sum(overlaps) / max(1, len(overlaps))) if overlaps else 0.0
        logger.info("RRF: sizes vec/fts/trgm raw=%d/%d/%d | accepted=%d | two-signal=%.1f%% | avg-overlap=%.2f",
                    len(vec_list), len(fts_list), len(trgm_list), len(accepted), 100*two_ratio, avg_ov)

        # 5) отсортируем doc_id по fused-скор и вернём объекты
        ordered_ids = [doc_id for doc_id, _ in sorted(fused.items(), key=lambda t: t[1], reverse=True)]
        out = []
        for did in ordered_ids:
            if did in id_to_result:
                r = id_to_result[did]
                r.fused_score = float(fused.get(did, 0.0))  # пригодится для tiebreaker
                out.append(r)
        return out
        
    def _connect(self):
        """Создание подключения к БД"""
        if isinstance(self.dsn_or_cfg, str):
            conn = psycopg2.connect(self.dsn_or_cfg)
        else:
            conn = psycopg2.connect(**self.dsn_or_cfg)
        conn.set_client_encoding('UTF8')
        return conn
    
    def _vec_literal(self, vec) -> str:
        """Преобразование вектора в строку для PostgreSQL"""
        return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"
    
    @staticmethod
    def _unaccent_basic(s: str) -> str:
        # приближённый unaccent: NFKD + выкинуть combining marks
        s = unicodedata.normalize('NFKD', s)
        return ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

    def _normalize_ru(self, s: str) -> str:
        s = (s or '').lower().replace('ё', 'е')
        s = self._unaccent_basic(s)
        # заменить «редкие» пробелы на обычные
        s = s.translate({cp: ' ' for cp in SPACE_MAP})
        # удалить невидимые символы
        s = ''.join(ch for ch in s if ord(ch) not in INVISIBLES)
        # схлопнуть пробелы
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _normalize_query(self, q: str) -> str:
        # базовая нормализация + словарные замены
        s = self._normalize_ru(q)
        try:
            for pat, repl in REPLACE.items():
                s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        except Exception:
            pass
        return s
 
    def _build_domain_vocab(self) -> set[str]:
        """Лениво собираем доменный словарь из glossary + (book, section)."""
        if self._domain_vocab is not None:
            return self._domain_vocab
        vocab: set[str] = set()

        # 1) из glossary.json
        for base, syns in (self.glossary or {}).items():
            for tok in re.findall(r'[а-яёa-z0-9]{3,}', str(base).lower()):
                vocab.add(tok.replace('ё', 'е'))
            for s in (syns or []):
                for tok in re.findall(r'[а-яёa-z0-9]{3,}', str(s).lower()):
                    vocab.add(tok.replace('ё', 'е'))

        # 2) из БД: отличные носители терминов — book и section (дешёво и достаточно)
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT book, section FROM docs;")
            for book, section in cur.fetchall():
                if book:
                    for tok in re.findall(r'[а-яёa-z0-9]{3,}', book.lower()):
                        vocab.add(tok.replace('ё', 'е'))
                if section:
                    for tok in re.findall(r'[а-яёa-z0-9]{3,}', section.lower()):
                        vocab.add(tok.replace('ё', 'е'))
        except Exception as e:
            logger.warning("build_domain_vocab: %s", e)
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass

        self._domain_vocab = {w for w in vocab if len(w) >= 3}
        self._domain_vocab_list = list(self._domain_vocab)
        return self._domain_vocab

    def _typo_fuzzy_normalize(self, q: str) -> str:
        """Поправляем опечатки по доменному словарю (WRatio) без агрессии."""
        if not HAVE_RF:
            return q
        vocab = self._build_domain_vocab()
        if not vocab:
            return q

        words = re.findall(r'[а-яёa-z0-9]+', q.lower())
        out = []
        for w in words:
            wn = w.replace('ё', 'е')
            # короткие/цифры/уже в словаре — не трогаем
            if wn in vocab or len(wn) < 5 or wn.isdigit():
                out.append(w); continue
            m = rf_process.extractOne(wn, self._domain_vocab_list, scorer=rf_fuzz.WRatio)
            if m and m[1] >= 88:       # устойчиво к «моркировки»→«маркировки»
                out.append(m[0])
            else:
                out.append(w)
        return " ".join(out)

    # внутри AdvancedHybridSearch._apply_glossary
    def _apply_glossary(self, q: str, max_variants: int = 3) -> list[str]:
        q_norm = self._normalize_query(q).lower()

        out  = [q_norm]
        seen = {q_norm}
        glossary = self.glossary or {}

        # длинные ключи – первыми
        for base in sorted(glossary.keys(), key=len, reverse=True):
            if len(out) >= max_variants:
                break

            syns = glossary.get(base) or []

            # 1) точное слово (границы)
            if not base.endswith('*'):
                pat = re.compile(rf'(?<!\w){re.escape(base)}(?!\w)')
            else:
                # 2) стем: w*
                stem = re.escape(base[:-1])
                pat = re.compile(rf'(?<!\w){stem}\w*')

            if not pat.search(q_norm):
                continue

            for s in syns:
                s_norm = self._normalize_query(str(s)).lower()
                v = pat.sub(s_norm, q_norm)
                if v not in seen:
                    out.append(v)
                    seen.add(v)
                    if len(out) >= max_variants:
                        break

        # лёгкие перефразы — как было
        if len(out) < max_variants and q_norm.startswith('как '):
            v = 'каким образом ' + q_norm[4:]
            if v not in seen:
                out.append(v); seen.add(v)

        if len(out) < max_variants and 'что делать' in q_norm:
            v = q_norm.replace('что делать', 'пошагово')
            if v not in seen:
                out.append(v); seen.add(v)

        return out[:max_variants]

    def _expand_query(self, q: str, max_variants: int = 3) -> Tuple[list[str], list[str]]:
        variants = self._apply_glossary(q, max_variants=max_variants)
        # keywords просто для логов/подсветки
        kw = re.findall(r'\b[а-яёА-ЯЁa-zA-Z0-9_\-]{4,}\b', " ".join(variants))
        return variants, list({k.lower() for k in kw})

    def _trgm_search(self, cur, query: str, limit: int = 150, use_knn: bool = True,
                    filters: Optional[List[Dict]] = None) -> list[SearchResult]:
        if not query or len(query) < 3:
            return []

        qn = self._normalize_ru(query)
        labels_sql, labels_params = _labels_where_sql(filters)

        if use_knn:
            sql = f"""
            SELECT id, book, section, page, text, meta,
                is_parent, parent_group, child_index, parent_title,
                1 - (trgm_all_norm <-> %s) AS tscore,
                COALESCE(importance_score, 1.0) AS importance
            FROM docs
            WHERE TRUE {labels_sql}
            ORDER BY trgm_all_norm <-> %s
            LIMIT %s;
            """
            # порядок: SELECT(%s), WHERE(labels...), ORDER BY(%s), LIMIT
            cur.execute(sql, (qn, *labels_params, qn, int(limit)))
        else:
            cur.execute("SELECT set_limit(%s);", (0.3,))
            sql = f"""
            SELECT id, book, section, page, text, meta,
                is_parent, parent_group, child_index, parent_title,
                similarity(trgm_all_norm, %s) AS tscore,
                COALESCE(importance_score, 1.0) AS importance
            FROM docs
            WHERE trgm_all_norm % %s {labels_sql}
            ORDER BY tscore DESC, COALESCE(importance_score, 1.0) DESC, id ASC
            LIMIT %s;
            """
            cur.execute(sql, (qn, qn, *labels_params, int(limit)))

        rows = cur.fetchall()
        out: list[SearchResult] = []
        for r in rows:
            m = (r[5] or {}).copy()
            extra = {
                "is_parent": bool(r[6]) if r[6] is not None else False,
                "parent_group": r[7],
                "child_index": int(r[8]) if r[8] is not None else None,
                "parent_title": r[9],
            }
            m.update({k: v for k, v in extra.items() if v is not None})
            tscore = float(r[10])
            importance = float(r[11]) if r[11] is not None else 1.0
            out.append(SearchResult(
                id=r[0], book=r[1], section=r[2], page=r[3],
                text=r[4], meta=m,
                trgm_score=tscore, importance=importance, score=0.0
            ))
        return out



    def _mmr_diversify(
        self,
        query: str,
        results: list[SearchResult],
        k: int = 10,
        lam: float = 0.70,           # 0..1: важность близости к запросу
        max_pool: int = 50,          # сколько верхних кандидатов диверсифицируем
        group_penalty: float = 0.2,  # штраф за повтор того же раздела/родителя
        base_score_weight: float = 0.10,  # небольшой приоритет исходного r.score
        qvec: Optional[np.ndarray] = None) -> list[SearchResult]:
        """
        Vectorized MMR + анти-дубликаты по book/section/parent и лёгкий учёт исходного score.
        Предполагается normalize_embeddings=True у self.embedder.
        """
        if not results or k <= 0:
            return []
        if len(results) <= k:
            return results

        pool = results[:min(max_pool, len(results))]
        # собираем валидные пары (индекс в pool + текст)
        valid: list[tuple[int, str]] = []
        for i, r in enumerate(pool):
            title = (r.meta or {}).get("parent_title") or (r.section or "")
            body  = (r.text or "")
            h = " ".join(x.strip() for x in (title, body) if x and x.strip()).strip()
            if not h:
                h = (title or body or r.book or "").strip()
            if h:
                valid.append((i, f"passage: {h[:512]}"))

        if not valid:
            return results[:k]

        idxs, texts = zip(*valid)
        embs = self.embedder.encode(
            list(texts),
            batch_size=min(64, max(8, len(texts))),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embs = np.ascontiguousarray(
            np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float32
        )

        if qvec is None:
            qvec = self._encode_query_vec(query)

        # Схожесть с запросом и попарная схожесть документов
        sim_q = embs @ qvec                    # shape: [n]
        S = embs @ embs.T                      # shape: [n, n], косинусы (т.к. нормированы)
        S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0) # защита от NaN/Inf
        S = np.clip(S, -1.0, 1.0)

        n = len(idxs)
        selected: list[int] = []
        candidates = set(range(n))

        # Нормируем исходные скоры для мягкого приоритета
        base_scores = np.array(
            [max(0.0, float(getattr(pool[i], "score", 0.0))) for i in idxs],
            dtype=np.float32
        )
        if base_scores.max() > 1e-9:
            base_scores = base_scores / base_scores.max()
        else:
            base_scores = np.zeros_like(base_scores)

        # Инициализация — лучший по близости к запросу (с учётом base_score_weight)
        mmr0 = sim_q + base_score_weight * base_scores
        i0 = int(np.argmax(mmr0))
        selected.append(i0)
        candidates.remove(i0)

        # Предзаготовим групповые ключи для штрафов
        def _norm(s: Optional[str]) -> str:
            if not s:
                return ""
            s = s.lower().replace('ё', 'е')
            return re.sub(r'\s+', ' ', s.strip())

        def _group_key(r) -> str:
            m = r.meta or {}
            # 1) явная групповая метка из БД
            pg = m.get("parent_group")
            if pg:
                return f"pg::{pg}"
            # 2) лёгкий вариант через parent_id в meta
            pid = m.get("parent_id")
            if pid is not None:
                return f"pid::{pid}"
            # 3) фоллбек на (book, section)
            return f"bs::{_norm(r.book)}|{_norm(r.section)}"

        # Предзаготовим ключи для пула
        group_keys = [_group_key(pool[i]) for i in idxs]

        def same_group(i: int, j: int) -> bool:
            return group_keys[i] == group_keys[j]

        # Основной цикл MMR
        while len(selected) < min(k, n) and candidates:
            # max редундантности каждого кандидата к уже выбранным
            red = np.max(S[:, selected], axis=1)  # shape: [n]

            # Базовый MMR + маленький приоритет исходного скoра
            mmr = lam * sim_q - (1.0 - lam) * red + base_score_weight * base_scores

            # Штрафы за дубликаты по группам
            if group_penalty > 0.0:
                for idx in list(candidates):
                    if any(same_group(idx, j) for j in selected):
                        mmr[idx] -= group_penalty

            # Выбираем лучший только из оставшихся кандидатов
            best_i = max(((idx, mmr[idx]) for idx in candidates), key=lambda t: t[1])[0]
            selected.append(best_i)
            candidates.remove(best_i)

        # Возвращаем выбранные (если нужно добрать до k — доклеиваем по исходному рангу)
        picked = [pool[idxs[i]] for i in selected]
        if len(picked) < k and len(idxs) > len(selected):
            rest = [pool[idxs[i]] for i in range(len(idxs)) if i not in selected]
            # более стабильный фолбэк-порядок
            rest.sort(key=lambda r: (
                getattr(r, "rerank_score", 0.0),
                getattr(r, "fused_score", 0.0),
                getattr(r, "vector_score", 0.0),
                -r.id
            ), reverse=True)
            picked += rest[:(k - len(picked))]
        return picked

    def _fulltext_search(self, cur, query: str, expanded_query: str, limit: int = 100,
                        filters: Optional[List[Dict]] = None) -> list[SearchResult]:
        labels_sql, labels_params = _labels_where_sql(filters)
        sql = f"""
        WITH ranked_docs AS (
            SELECT 
                id, book, section, page, text, meta,
                is_parent, parent_group, child_index, parent_title,
                ts_rank_cd(ts, q, 32) AS rank_cd,
                ts_rank(ts, q, 1)      AS rank_simple,
                COALESCE(importance_score, 1.0) AS importance
            FROM docs, websearch_to_tsquery('russian', %s) AS q
            WHERE ts @@ q {labels_sql}

            UNION

            SELECT 
                id, book, section, page, text, meta,
                is_parent, parent_group, child_index, parent_title,
                ts_rank_cd(ts, q, 32) * 0.9 AS rank_cd,
                ts_rank(ts, q, 1)      * 0.9 AS rank_simple,
                COALESCE(importance_score, 1.0) AS importance
            FROM docs, plainto_tsquery('russian', %s) AS q
            WHERE ts @@ q {labels_sql}
        )
        SELECT DISTINCT ON (id)
            id, book, section, page, text, meta,
            is_parent, parent_group, child_index, parent_title,
            GREATEST(rank_cd, rank_simple) AS fscore,
            importance
        FROM ranked_docs
        ORDER BY id, fscore DESC
        LIMIT %s;
        """
        # Порядок параметров ВАЖЕН:
        params = [query, *labels_params, expanded_query, *labels_params, limit]
        cur.execute(sql, params)
        rows = cur.fetchall()

        results: Dict[int, SearchResult] = {}
        for row in rows:
            doc_id = row[0]
            fscore = float(row[10])
            m = (row[5] or {}).copy()
            extra = {
                "is_parent": bool(row[6]) if row[6] is not None else False,
                "parent_group": row[7],
                "child_index": int(row[8]) if row[8] is not None else None,
                "parent_title": row[9],
            }
            m.update({k: v for k, v in extra.items() if v is not None})

            # храним лучший fulltext_score на документ
            if doc_id in results and results[doc_id].fulltext_score >= fscore:
                continue
            results[doc_id] = SearchResult(
                id=row[0], book=row[1], section=row[2], page=row[3], text=row[4],
                meta=m, fulltext_score=fscore,
                importance=float(row[11]) if row[11] is not None else 1.0,
                score=0.0
            )
        return list(results.values())


    
    def _vector_search(self, cur, query_vec, limit: int = 100, filters: Optional[List[Dict]] = None):
        qv = self._vec_literal(query_vec)
        labels_sql, labels_params = _labels_where_sql(filters)
        cur.execute(f"""
            SELECT 
                id, book, section, page, text, meta,
                is_parent, parent_group, child_index, parent_title,
                1 - (emb <=> %s::vector) AS vscore,
                COALESCE(importance_score, 1.0) AS importance
            FROM docs 
            WHERE emb IS NOT NULL {labels_sql}
            ORDER BY emb <=> %s::vector
            LIMIT %s;
        """, (qv, *labels_params, qv, limit))
        rows = cur.fetchall()

        results = []
        for row in rows:
            # индексы:
            # 0=id,1=book,2=section,3=page,4=text,5=meta,
            # 6=is_parent,7=parent_group,8=child_index,9=parent_title,
            # 10=vscore, 11=importance

            # аккуратно объединяем meta + parent-поля
            m = (row[5] or {}).copy()
            extra = {
                "is_parent": bool(row[6]) if row[6] is not None else False,
                "parent_group": row[7],
                "child_index": int(row[8]) if row[8] is not None else None,
                "parent_title": row[9],
            }
            # не пишем None-ы в meta
            m.update({k: v for k, v in extra.items() if v is not None})

            vscore = float(row[10])
            importance = float(row[11]) if row[11] is not None else 1.0

            # лёгкий буст по важности (помни: ещё будет множитель в _calculate_final_scores)
            adjusted_score = vscore * (0.8 + 0.2 * importance)

            results.append(SearchResult(
                id=row[0],
                book=row[1],
                section=row[2],
                page=row[3],
                text=row[4],
                meta=m,
                vector_score=adjusted_score,
                importance=importance,
                score=0.0
            ))

        return results

    
    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 20,
        batch_size: int = 32,
    ) -> list[SearchResult]:
        """Переранжирование BGE-reranker: сортируем по rerank_score и отдаем top_k."""
        if not self.use_reranking or not self.reranker or not results:
            return results

        # Берем ограниченный пул кандидатов для качества/скорости
        candidates = results[:min(max(80, top_k*6), len(results))]
        if top_k <= 0:
            top_k = len(candidates)

        # Готовим пары [query, doc]
        pairs = []
        for r in candidates:
            title = (r.meta or {}).get("parent_title") or (r.section or "")
            body  = (r.text or "")
            body  = self._snippet_for_reranker(query, body, max_chars=700)
            pairs.append([query, f"{title}\n{body}"])


        try:
            # У FlagReranker compute_score поддерживает normalize и batch_size
            scores = self.reranker.compute_score(pairs, normalize=True, batch_size=batch_size)

            # Приводим к list[float] для совместимости (list/np/torch)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            else:
                scores = [float(s) for s in scores]

            for r, s in zip(candidates, scores):
                r.rerank_score = float(s)

            # Сортируем и возвращаем top_k по rerank_score
            candidates.sort(key=lambda x: x.rerank_score, reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.warning(f"Ошибка при reranking: {e}")
            return results

    
    def _calculate_final_scores(self, results: Dict[int, SearchResult]) -> list[SearchResult]:
        """Расчет финальных скоров с нормализацией, отсечкой шума и глобальной перенормировкой весов."""
        if not results:
            return []

        eps = 1e-9
        min_threshold = 1e-2  # если максимум по компоненте ниже — считаем, что сигнала нет

        # 1) максимумы по компонентам
        max_vector = max((r.vector_score   for r in results.values()), default=0.0)
        max_full   = max((r.fulltext_score for r in results.values()), default=0.0)
        max_trgm   = max((getattr(r, "trgm_score", 0.0) for r in results.values()), default=0.0)
        max_rerank = max((r.rerank_score   for r in results.values()), default=0.0)

        # 2) какие сигналы вообще активны в этом запросе
        has_v  = max_vector > min_threshold
        has_f  = max_full   > min_threshold
        has_t  = max_trgm   > min_threshold
        has_rr = self.use_reranking and (max_rerank > min_threshold)

        # 3) переносим веса на активные сигналы и нормируем их сумму к 1
        wv = self.vector_weight   if has_v  else 0.0
        wf = self.fulltext_weight if has_f  else 0.0
        wt = self.trgm_weight     if has_t  else 0.0
        wr = self.rerank_weight   if has_rr else 0.0

        sumw = wv + wf + wt + wr
        if sumw <= eps:
            # нет ни одного сигнала — все скоры будут 0, просто вернём стабильную сортировку
            norm = (0.0, 0.0, 0.0, 0.0)
        else:
            norm = (wv / sumw, wf / sumw, wt / sumw, wr / sumw)

        wv, wf, wt, wr = norm

        # 4) считаем нормированные компоненты и итоговый скор
        for r in results.values():
            nv = (r.vector_score   / (max_vector + eps)) if has_v  else 0.0
            nf = (r.fulltext_score / (max_full   + eps)) if has_f  else 0.0
            nt = (getattr(r, "trgm_score", 0.0) / (max_trgm + eps)) if has_t else 0.0
            nr = (r.rerank_score   / (max_rerank + eps)) if has_rr else 0.0

            r.score = (wv * nv) + (wf * nf) + (wt * nt) + (wr * nr)

            # мягкий бонус за importance
            r.score *= (0.9 + 0.1 * float(getattr(r, "importance", 1.0)))

        return sorted(results.values(), key=lambda x: x.score, reverse=True)

    def _encode_query_vec(self, q: str) -> np.ndarray:
        v = self.embedder.encode(
            [f"query: {q}"],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        # строгая типизация + защита от NaN/Inf
        return np.ascontiguousarray(
            np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float32
        )

    def _snippet_for_reranker(self, query: str, text: str, max_chars: int = 700) -> str:
        # простая «BM25-подобная» эвристика: по токенному оверлапу
        sents = re.split(r'(?<=[.!?…])\s+', (text or '').strip())
        qtok = self._tokenize(query)
        scored = []
        for s in sents:
            stok = self._tokenize(s)
            if not stok:
                continue
            overlap = len(qtok & stok) / max(1, len(qtok))
            scored.append((overlap, s))
        scored.sort(key=lambda t: t[0], reverse=True)
        picked = " ".join(s for _, s in scored[:3])[:max_chars]
        return picked if picked else (text or "")[:max_chars]

    
    def search(self, 
            query: str, 
            top_k: int = 8,
            search_methods: Optional[List[str]] = None,
            min_score: float = 0.1,
            use_query_expansion: bool = True,
            context_window: int = 0,
            filters: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Weighted-RRF поверх (vector/fulltext/trgm) с гейтами и fail-safe,
        затем BGE-reranker → MMR → контекст. Сортировка после reranker — по rerank_score.
        """
        inline = _parse_inline_filters_text(query)
        if inline:
            filters = (filters or []) + inline

        if not query.strip():
            return []
        
        # базовая нормализация
        q_norm = self._normalize_query(query)
        # типо-фикс (если RF есть)
        q_fixed = self._typo_fuzzy_normalize(q_norm) if HAVE_RF else q_norm

        # методы
        if search_methods is None:
            search_methods = ['vector', 'fulltext']
        if 'trgm' not in search_methods and len(self._normalize_query(query)) >= 3:
            search_methods = [*search_methods, 'trgm']

        # динамика порогов для trgm
        self._tune_trgm_thresholds(query)

        # (опц.) расширение запроса
        if use_query_expansion:
            variants, keywords = self._expand_query(q_fixed, max_variants=3)
        else:
            variants, keywords = [q_fixed], []
        # добавим исходник для надёжности и уберём дубликаты, максимум 3
        variants = list(dict.fromkeys([q_fixed, *variants]))[:3]

        logger.debug(f"Варианты запроса: {variants}")
        if keywords:
            logger.debug(f"Ключевые слова: {keywords}")

        # эмбеддинг запроса для MMR/переранжировки
        q_ck = q_fixed.lower().strip()
        query_vec = self.query_cache.get(q_ck)
        if query_vec is None:
            query_vec = self._encode_query_vec(q_fixed)
            with self._cache_lock:
                self.query_cache[q_ck] = query_vec

        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute("SET LOCAL hnsw.ef_search = 256;")
            except:
                pass

            # аккумулируем «чистые» ранговые листы для RRF
            vec_list_all, fts_list_all, trgm_list_all = [], [], []
            # и агрегат для метрик/вывода (максимум по компонентам)
            all_results: Dict[int, SearchResult] = {}

            vec_found = fts_found = trgm_found = 0
            vec_ids: set[int] = set()
            fts_ids: set[int] = set()
            trgm_ids: set[int] = set()

            for qv in variants:
                ck = qv.lower().strip()
                qvec = self.query_cache.get(ck)
                if qvec is None:
                    qvec = self._encode_query_vec(qv)
                    with self._cache_lock:
                        self.query_cache[ck] = qvec

                # vector
                if 'vector' in search_methods:
                    vres = list(self._vector_search(cur, qvec, limit=100, filters=filters))
                    vec_list_all.extend(vres)
                    vec_found += len(vres); vec_ids.update(r.id for r in vres)
                    for r in vres:
                        all_results.setdefault(r.id, r)
                        all_results[r.id].vector_score = max(all_results[r.id].vector_score, r.vector_score)

                # fulltext
                if 'fulltext' in search_methods:
                    fres = list(self._fulltext_search(cur, query, qv, limit=100, filters=filters))
                    fts_list_all.extend(fres)
                    fts_found += len(fres); fts_ids.update(r.id for r in fres)
                    for r in fres:
                        if r.id in all_results:
                            all_results[r.id].fulltext_score = max(all_results[r.id].fulltext_score, r.fulltext_score)
                        else:
                            all_results[r.id] = r

                # trgm (сразу берем ограниченную глубину)
                if ('trgm' in (search_methods or [])) and self._should_use_trgm(query):
                    tres = list(self._trgm_search(cur, qv, limit=self.rrf_trgm_depth, use_knn=True, filters=filters))
                    trgm_list_all.extend(tres)
                    trgm_found += len(tres); trgm_ids.update(r.id for r in tres)
                    for r in tres:
                        if r.id in all_results:
                            all_results[r.id].trgm_score = max(getattr(all_results[r.id], 'trgm_score', 0.0), r.trgm_score)
                        else:
                            all_results[r.id] = r

            logger.info(
                "Кандидатов всего (сырых): %d | vector: %d (уник.%d) | fulltext: %d (уник.%d) | trgm: %d (уник.%d)",
                len(all_results), vec_found, len(vec_ids), fts_found, len(fts_ids), trgm_found, len(trgm_ids)
            )

            # дедуп ранговых листов по id в каждом методе
            self._last_vec_list = self._dedup_rank(vec_list_all, key=lambda x: x.vector_score)
            self._last_fts_list = self._dedup_rank(fts_list_all, key=lambda x: x.fulltext_score)
            self._last_trgm_list = self._dedup_rank(trgm_list_all, key=lambda x: getattr(x, 'trgm_score', 0.0))

            self._last_vec_rankmap = {r.id: i+1 for i, r in enumerate(self._last_vec_list)}
            self._last_fts_rankmap = {r.id: i+1 for i, r in enumerate(self._last_fts_list)}

            # RRF-слияние с гейтами → пул для reranker
            want_L = max(60, 4 * top_k)
            pre_raw = self._rrf_merge(
                {'vector': self._last_vec_list, 'fulltext': self._last_fts_list,
                'trgm': self._last_trgm_list if self._should_use_trgm(query) else []},
                query=q_fixed,
                id_to_result=all_results,
                want_pool=want_L
            )

            # fail-safe на консенсус: если мало — доклеим из vec/fts
            pre = self._safe_trim_for_reranker(pre_raw, want=max(80, top_k * 6))

            # Reranking
            if self.use_reranking and pre:
                reranked = self._rerank_results(q_fixed, pre, top_k=len(pre))
                # Чистый порядок по rerank_score (без повторного суммирования базовых сигналов)
                reranked.sort(key=lambda r: (r.rerank_score, getattr(r, "fused_score", 0.0)), reverse=True)
                candidates = reranked
            else:
                # Fallback: без reranker оставим RRF-порядок
                candidates = pre

            # MMR-диверсификация и контекст
            final_results = candidates
            if top_k >= 3:
                final_results = self._mmr_diversify(q_fixed, final_results, k=min(top_k, 12), qvec=query_vec)

            if context_window > 0:
                final_results = self._add_context(cur, final_results, context_window)

            # обрезка
            final_results = final_results[:top_k]

            # нормализуем финальный score:
            # если был reranker — приводим к [0,1] по rerank_score, иначе — по вашей схеме
            if self.use_reranking and any(r.rerank_score > 0 for r in final_results):
                max_rr = max(r.rerank_score for r in final_results) or 1.0
                for r in final_results:
                    base = (r.rerank_score / max_rr) if max_rr > 0 else 0.0
                    # лёгкий бонус за importance (как было)
                    r.score = base * (0.9 + 0.1 * float(getattr(r, "importance", 1.0)))
            else:
                # когда reranker нет — используем вашу нормализацию
                tmp = {r.id: r for r in final_results}
                final_results = self._calculate_final_scores(tmp)

            try:
                ids = [r.id for r in final_results if isinstance(r.id, int) and r.id > 0]
                lab_map = _fetch_labels_for_docs(cur, ids)
                for r in final_results:
                    if r.id in lab_map:
                        if r.meta is None:
                            r.meta = {}
                        r.meta['labels'] = lab_map[r.id]
                _boost_by_filters(final_results, filters)
            except Exception as e:
                logger.warning("labels attach/boost: %s", e)

            # порог по min_score
            filtered = [r for r in final_results if float(getattr(r, 'score', 0.0)) >= min_score]

            logger.info("Финально (после RRF→rerank→MMR): %d документов (min_score=%.2f)", len(filtered), min_score)

            # keywords в meta
            if keywords:
                for r in filtered:
                    if r.meta is None:
                        r.meta = {}
                    r.meta.setdefault('query_keywords', keywords)

            # возврат формата, как у вас
            return [
                {
                    "id": r.id,
                    "book": r.book,
                    "section": r.section,
                    "page": r.page,
                    "text": r.text,
                    "score": float(r.score),
                    "vector_score": float(getattr(r, 'vector_score', 0.0)),
                    "fulltext_score": float(getattr(r, 'fulltext_score', 0.0)),
                    "trgm_score": float(getattr(r, 'trgm_score', 0.0)),
                    "rerank_score": float(getattr(r, 'rerank_score', 0.0)),
                    "importance": float(getattr(r, 'importance', 1.0)),
                    "meta": r.meta
                }
                for r in filtered
            ]

        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass
    
    def _add_context(self, cur, results: list[SearchResult], window: int = 1) -> list[SearchResult]:
        """
        Добавляем логический контекст: parent-сниппет и ближайших соседей.
        Приоритет:
        1) Если есть parent_group/child_index — соседи по child_index.
        2) Если результат — родитель is_parent=True — подтянуть первые N детей.
        3) Иначе, если есть page — ближайшие по page в том же (book, section).
        4) Иначе — ближайшие по id в том же (book, section).
        """
        if window <= 0 or not results:
            return results

        # Настройки контекста
        PREFIX_CHARS = 220   # из предыдущего куска (хвост)
        SUFFIX_CHARS = 220   # из следующего куска (начало)
        PARENT_CHARS = 400   # сниппет из родителя (для детей)
        MAX_ENHANCED = 10    # обогащаем только первые N

        def _smart_tail(s: str, max_len: int) -> str:
            if not s:
                return ""
            s = s[-(max_len + 200):]
            m = re.search(r'([.!?…]\s+)([^.!?…]*)$', s)
            frag = (s[m.end():] if m else s)[-max_len:]
            return frag.lstrip()

        def _smart_head(s: str, max_len: int) -> str:
            if not s:
                return ""
            s = s[:(max_len + 200)]
            m = re.search(r'([.!?…]\s+)', s)
            start = m.end() if m else 0
            frag = s[start:start + max_len]
            return frag.rstrip()
        
        def _fetch_concat(cur, sql: str, params: tuple, mode: str, max_len: int) -> Optional[str]:
            assert mode in ('before', 'after'), f"bad mode: {mode}"
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                return None
            texts = [str(x[0]) for x in rows if x and x[0]]

            if mode == 'before':
                texts = list(reversed(texts))  # читаемо слева-направо
                return _smart_tail("\n\n".join(texts), max_len)
            else:
                return _smart_head("\n\n".join(texts), max_len)

        def _norm_ws(s: str) -> str:
            return re.sub(r'\n{3,}', '\n\n', s).strip()

        enhanced_results: list[SearchResult] = []

        for r in results[:MAX_ENHANCED]:
            if (r.meta or {}).get("_ctx_augmented"):
                enhanced_results.append(r)
                continue

            pg = (r.meta or {}).get("parent_group")
            ci = (r.meta or {}).get("child_index")
            is_parent = bool((r.meta or {}).get("is_parent"))

            added_before = None
            added_after  = None
            parent_snip  = None
            child_preview = None

            if pg is not None and ci is not None:
                # --- 1) Соседи внутри раздела по child_index (только для детей)
                added_before = _fetch_concat(cur, """
                    SELECT text FROM docs
                    WHERE parent_group = %s AND child_index < %s
                    ORDER BY child_index DESC
                    LIMIT %s
                """, (pg, int(ci), window), mode='before', max_len=PREFIX_CHARS)

                added_after = _fetch_concat(cur, """
                    SELECT text FROM docs
                    WHERE parent_group = %s AND child_index > %s
                    ORDER BY child_index ASC
                    LIMIT %s
                """, (pg, int(ci), window), mode='after', max_len=SUFFIX_CHARS)

                # краткий сниппет родителя
                try:
                    cur.execute("""
                        SELECT text
                        FROM docs
                        WHERE is_parent = TRUE AND parent_group = %s
                        LIMIT 1
                    """, (pg,))
                    row_parent = cur.fetchone()
                    if row_parent and row_parent[0]:
                        parent_text = str(row_parent[0])
                        para = parent_text.split("\n\n", 1)[0] or parent_text
                        parent_snip = para[:PARENT_CHARS].rstrip()
                except Exception:
                    pass

            elif is_parent and pg is not None:
                # --- 2) Результат — сам родитель: берём первые N детей как превью
                try:
                    cur.execute("""
                        SELECT text
                        FROM docs
                        WHERE parent_group = %s AND child_index IS NOT NULL
                        ORDER BY child_index ASC
                        LIMIT %s
                    """, (pg, max(1, window)))
                    rows = cur.fetchall()
                    if rows:
                        joined = "\n\n".join([str(x[0]) for x in rows if x and x[0]])
                        # используем "head", чтобы ровно начать на границе предложения
                        child_preview = _smart_head(joined, SUFFIX_CHARS)
                except Exception:
                    pass

            else:
                # --- 3) Фоллбек по page
                if r.page is not None:
                    added_before = _fetch_concat(cur, """
                        SELECT text FROM docs
                        WHERE book = %s AND section = %s AND page < %s
                        ORDER BY page DESC
                        LIMIT %s
                    """, (r.book, r.section, int(r.page), window), mode='before', max_len=PREFIX_CHARS)

                    added_after = _fetch_concat(cur, """
                        SELECT text FROM docs
                        WHERE book = %s AND section = %s AND page > %s
                        ORDER BY page ASC
                        LIMIT %s
                    """, (r.book, r.section, int(r.page), window), mode='after', max_len=SUFFIX_CHARS)

                else:
                    # --- 4) Фоллбек по id
                    added_before = _fetch_concat(cur, """
                        SELECT text FROM docs
                        WHERE book = %s AND section = %s AND id < %s
                        ORDER BY id DESC
                        LIMIT %s
                    """, (r.book, r.section, r.id, window), mode='before', max_len=PREFIX_CHARS)

                    added_after = _fetch_concat(cur, """
                        SELECT text FROM docs
                        WHERE book = %s AND section = %s AND id > %s
                        ORDER BY id ASC
                        LIMIT %s
                    """, (r.book, r.section, r.id, window), mode='after', max_len=SUFFIX_CHARS)

            if added_before:
                added_before = added_before.strip() or None
            if added_after:
                added_after = added_after.strip() or None
            if child_preview:
                child_preview = child_preview.strip() or None
                
            # --- сборка текста с контекстом ---
            body = r.text or ""
            pieces: list[str] = []

            if parent_snip:
                title = (r.meta or {}).get("parent_title") or (r.section or "")
                head = f"«{title.strip()}» — краткий контекст:\n{parent_snip}" if title and title.strip() else parent_snip
                pieces.append(head)

            # защита от дубликатов: не добавляем, если совпадает «стык»
            if added_before and not body.lstrip().startswith(added_before[-40:].lstrip()):
                pieces.append("...\n" + added_before)

            pieces.append(body)

            if child_preview and not body.rstrip().endswith(child_preview[:40].rstrip()):
                # для родителя добавим превью детей после основного текста
                pieces.append(child_preview + "\n...")

            if added_after and not body.rstrip().endswith(added_after[:40].rstrip()):
                pieces.append(added_after + "\n...")

            if r.meta is None:
                r.meta = {}
            if r.meta.get('_orig_head') is None:
                r.meta['_orig_head'] = (r.text or '')[:420]

            r.text = _norm_ws("\n\n".join(pieces))

            if r.meta is None:
                r.meta = {}
            r.meta["_ctx_augmented"] = True

            enhanced_results.append(r)

        # Остальные без изменений
        enhanced_results.extend(results[MAX_ENHANCED:])
        return enhanced_results
    
    def search_similar(self, doc_id: int, top_k: int = 5) -> list[Dict]:
        """Поиск похожих документов с косинусным расстоянием"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            
            # Оптимизация HNSW
            try:
                cur.execute("SET LOCAL hnsw.ef_search = 128;")
            except:
                pass
            
            # Получаем информацию об исходном документе
            cur.execute("""
                SELECT emb, book, section, text, meta
                FROM docs WHERE id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            if not row:
                return []
            
            base_emb, base_book, base_section, base_text, base_meta = row

            cur.execute("""
                SELECT 
                    id, book, section, page, text, meta,
                    1 - (emb <=> %s) AS score,
                    CASE 
                        WHEN book = %s THEN 0.1 
                        ELSE 0 
                    END AS same_book_bonus
                FROM docs
                WHERE id <> %s AND emb IS NOT NULL
                ORDER BY (emb <=> %s) + 
                         CASE WHEN book = %s THEN -0.05 ELSE 0 END
                LIMIT %s
            """, (base_emb, base_book, doc_id, base_emb, base_book, top_k * 2))
            
            rows = cur.fetchall()
            results = []
            
            for row in rows:
                similarity_score = float(row[6])
                same_book_bonus = float(row[7])
                
                # Проверяем схожесть разделов
                section_similarity = 0
                if row[2] and base_section:
                    if row[2] == base_section:
                        section_similarity = 0.15
                    elif row[2].lower() in base_section.lower() or base_section.lower() in row[2].lower():
                        section_similarity = 0.05
                
                final_score = similarity_score + same_book_bonus + section_similarity
                
                results.append({
                    "id": row[0],
                    "book": row[1],
                    "section": row[2],
                    "page": row[3],
                    "text": row[4],
                    "score": final_score,
                    "similarity": similarity_score,
                    "same_book": row[1] == base_book,
                    "meta": row[5] or {}
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:top_k]
            
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Получение документа по ID"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, book, section, page, text, meta
                FROM docs WHERE id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "book": row[1],
                    "section": row[2],
                    "page": row[3],
                    "text": row[4],
                    "meta": row[5] or {}
                }
            return None
            
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass


# Класс для обратной совместимости
class HybridSearch(AdvancedHybridSearch):
    """Обертка для обратной совместимости"""
    pass


# Функция для обратной совместимости
def search(question: str, top: int = 8) -> list[Dict]:
    """Функция-обертка для обратной совместимости"""
    searcher = get_searcher()
    results = searcher.search(question, top_k=top)
    # Упрощаем вывод для обратной совместимости
    return [
        {
            "id": r["id"],
            "book": r["book"],
            "section": r["section"],
            "page": r["page"],
            "text": r["text"],
            "score": r["score"]
        }
        for r in results
    ]


if __name__ == "__main__":
    # Тестирование
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Улучшенный поиск по русским документам')
    parser.add_argument('query', nargs='*', help='Поисковый запрос')
    parser.add_argument('--top-k', type=int, default=8, help='Количество результатов')
    parser.add_argument('--no-rerank', action='store_true', help='Отключить reranking')
    parser.add_argument('--context', type=int, default=0, help='Размер контекстного окна')
    parser.add_argument('--min-score', type=float, default=0.2, help='Минимальный скор')
    args = parser.parse_args()
    
    query = " ".join(args.query) if args.query else "Как настроить систему?"
    
    searcher = get_searcher()
    if args.no_rerank:
        searcher.use_reranking = False
    
    # Выполняем поиск
    print(f"\n🔍 Поиск: '{query}'")
    print("=" * 80)
    
    if searcher.use_reranking:
        print("✓ BGE-reranker v2-m3 активен (отличная поддержка русского языка)")
    else:
        print("⚠ Reranker отключен (установите FlagEmbedding для улучшения качества)")
    
    print("-" * 80)
    
    results = searcher.search(
        query, 
        top_k=args.top_k,
        context_window=args.context,
        min_score=args.min_score
    )
    
    if not results:
        print("❌ Ничего не найдено")
    else:
        print(f"✓ Найдено {len(results)} результатов:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['book']}] {r['section']} (стр. {r['page']})")
            print(f"   📊 Общий скор: {r['score']:.3f}")
            print(f"   📍 Векторный: {r['vector_score']:.3f}")
            print(f"   📝 Полнотекст: {r['fulltext_score']:.3f}")
            if r['rerank_score'] > 0:
                print(f"   🎯 Rerank: {r['rerank_score']:.3f}")
            print(f"   ⭐ Важность: {r['importance']:.1f}")
            print(f"   📄 Текст: {r['text'][:150]}...")
            
            if r.get('meta'):
                kw = r['meta'].get('query_keywords') or r['meta'].get('keywords')
                if kw:
                    print(f"   🔑 Ключевые слова: {', '.join(kw[:5])}")
            print()