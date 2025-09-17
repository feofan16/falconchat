# scripts/utils/okpd_safe.py
import os, re
from typing import Optional, Tuple, List
from scripts.utils.labels import okpd_breadcrumb, attach_label

SAFE_OKPD_CONF_LEAF   = float(os.getenv("SAFE_OKPD_CONF_LEAF",   "0.93"))
SAFE_OKPD_CONF_ANCHOR = float(os.getenv("SAFE_OKPD_CONF_ANCHOR", "0.97"))
SAFE_OKPD_MAX_LEVEL   = int(os.getenv("SAFE_OKPD_MAX_LEVEL", "3"))  # до какого уровня укрупняем

def _okpd_trim(code: str, level: int) -> str:
    parts = (code or "").split(".")
    level = max(1, min(len(parts), level))
    return ".".join(parts[:level])

def _okpd_exists(cur, code: str) -> Optional[int]:
    cur.execute("SELECT level FROM okpd2 WHERE code=%s", (code,))
    row = cur.fetchone()
    return int(row[0]) if row else None

def _okpd_nearest_valid(cur, code: str) -> Tuple[str, Optional[int]]:
    """Возвращает ближайший валидный код вверх по иерархии (включая сам)."""
    c = code
    while c:
        lvl = _okpd_exists(cur, c)
        if lvl:
            return c, lvl
        c = ".".join(c.split(".")[:-1])
    return "", None

def attach_okpd_safely(cur, doc_id: int, code: str, conf: float, why: str, *, source: str = "pred") -> bool:
    """
    Возвращает True, если боевой код (ns='okpd') установлен на полном уровне,
    False — если отправлено на ревью/укрупнено.
    """
    code = (code or "").strip()
    if not code:
        return False

    # 1) валидируем по справочнику, поднимаясь вверх если нужно
    valid_code, level = _okpd_nearest_valid(cur, code)
    if not valid_code:
        # ничего валидного — только гипотеза + ревью
        _queue_review(cur, doc_id, code, conf, why)
        _attach_pred(cur, doc_id, code, conf)
        return False

    # 2) сверхнадежный "якорь" (ГОСТ) — можно ставить полный код
    is_anchor = (source == "gost_anchor") or why.lower().startswith("гост ")
    if is_anchor and conf >= SAFE_OKPD_CONF_ANCHOR:
        attach_label(cur, doc_id, 'okpd', valid_code, conf, None, okpd_breadcrumb(valid_code))
        return True

    # 3) высокое доверие — лист разрешён
    if level >= 4 and conf >= SAFE_OKPD_CONF_LEAF:
        attach_label(cur, doc_id, 'okpd', valid_code, conf, None, okpd_breadcrumb(valid_code))
        return True

    # 4) иначе укрупняем до безопасного уровня и ставим гипотезу
    safe_code = _okpd_trim(valid_code, min(level or 1, SAFE_OKPD_MAX_LEVEL))
    attach_label(cur, doc_id, 'okpd', safe_code, min(conf, 0.85), None, okpd_breadcrumb(safe_code))
    _attach_pred(cur, doc_id, valid_code, conf)
    _queue_review(cur, doc_id, valid_code, conf, why)
    return False

def _attach_pred(cur, doc_id: int, code: str, conf: float):
    attach_label(cur, doc_id, 'okpd_pred', code, conf, None, okpd_breadcrumb(code))

def _queue_review(cur, doc_id: int, code: str, conf: float, why: str):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS okpd_review_queue(
          id bigserial PRIMARY KEY,
          doc_id bigint NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
          predicted_code text NOT NULL,
          conf real,
          reason text,
          created_at timestamptz DEFAULT now(),
          status text DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
          reviewer text
        );
    """)
    cur.execute("""
        INSERT INTO okpd_review_queue(doc_id, predicted_code, conf, reason)
        VALUES (%s,%s,%s,%s)
    """, (doc_id, code, conf, why))
