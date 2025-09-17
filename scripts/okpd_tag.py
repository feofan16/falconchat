# scripts/okpd_tag.py
from __future__ import annotations
import argparse
import logging
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import sys

from scripts.utils.labels import (
    extract_labels_from_text, classify_factory_item,
    okpd_canon, okpd_breadcrumb, extract_material_info
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB = dict(dbname="rag", user="rag", password="rag", host="127.0.0.1", port=5432)

class OKPDTagger:
    """Класс для тегирования документов ОКПД кодами"""
    
    def __init__(self, conn, dry_run: bool = False):
        self.conn = conn
        self.dry_run = dry_run
        self.stats = {
            'processed': 0,
            'tagged': 0,
            'okpd_found': 0,
            'okpd_guessed': 0,
            'gost_found': 0,
            'materials_found': 0,
            'unsure': 0,
            'errors': 0
        }
    
    def upsert_label(self, cur, ns: str, code: str, title: Optional[str] = None, 
                    path: Optional[List[str]] = None) -> int:
        """Вставка или обновление метки - устойчива к типу курсора"""
        try:
            cur.execute("""
                INSERT INTO labels(ns, code, title, path)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ns, code) DO UPDATE SET
                  title = COALESCE(EXCLUDED.title, labels.title),
                  path  = COALESCE(EXCLUDED.path,  labels.path)
                RETURNING id
            """, (ns, code, title, path))
            row = cur.fetchone()
            # Поддержка и dict, и tuple
            return row['id'] if isinstance(row, dict) else row[0]
        except Exception as e:
            logger.error(f"Error upserting label {ns}:{code}: {e}")
            raise
    
    def attach_label(self, cur, doc_id: int, ns: str, code: str, 
                    conf: Optional[float] = None, title: Optional[str] = None,
                    path: Optional[List[str]] = None):
        """Привязка метки к документу с защитой от NULL в conf"""
        if ns == 'okpd':
            code = okpd_canon(code) or code
            path = path or okpd_breadcrumb(code)
        
        lid = self.upsert_label(cur, ns, code, title, path)
        
        # КРИТИЧНО: защита от NULL в conf
        cur.execute("""
            INSERT INTO docs_labels(doc_id, label_id, conf)
            VALUES (%s, %s, %s)
            ON CONFLICT (doc_id, label_id) DO UPDATE SET
                conf = GREATEST(
                    docs_labels.conf,
                    COALESCE(EXCLUDED.conf, docs_labels.conf)
                )
        """, (doc_id, lid, conf))
    
    def has_okpd(self, cur, doc_id: int) -> bool:
        """Проверка наличия ОКПД метки у документа"""
        cur.execute("""
            SELECT 1
            FROM docs_labels dl
            JOIN labels l ON l.id = dl.label_id
            WHERE dl.doc_id = %s AND l.ns = 'okpd'
            LIMIT 1
        """, (doc_id,))
        return cur.fetchone() is not None
    
    def save_material_info(self, cur, doc_id: int, info: Dict[str, Any]):
        """Сохранение расширенной информации о материале в meta"""
        if not info:
            return
        
        # Сохраняем максимум полезной информации
        meta_update = {
            'material_info': {
                'type':     info.get('type'),
                'material': info.get('material'),
                'material_code': info.get('material_code'),  # Новое
                'size':     info.get('size'),
                'gost':     info.get('gost', []),
                'tu':       info.get('tu'),      # Новое
                'ost':      info.get('ost'),     # Новое
                'din':      info.get('din'),     # Новое
                'iso':      info.get('iso'),     # Новое
                'features': info.get('features', {}),
                'extracted_at': datetime.now().isoformat()
            }
        }
        
        cur.execute("""
            UPDATE docs
            SET meta = COALESCE(meta, '{}'::jsonb) || %s
            WHERE id = %s
        """, (Json(meta_update), doc_id))
    
    def process_document(self, cur, doc_id: int, section: str, text: str, 
                         min_conf: float = 0.90, save_materials: bool = True,
                         max_analyze_chars: int = 3000) -> Dict:
        """Обработка одного документа"""
        result = {
            'doc_id': doc_id,
            'labels_added': [],
            'okpd_guess': None,
            'material_info': None
        }
        
        try:
            # Используем настраиваемый лимит для анализа
            body = f"{section or ''}\n{text or ''}"[:max_analyze_chars]
            
            # 1) Извлечение явных меток из текста
            labels = extract_labels_from_text(body)
            for lb in labels:
                self.attach_label(cur, doc_id, lb['ns'], lb['code'], 
                                lb.get('conf'), None, lb.get('path'))
                result['labels_added'].append(f"{lb['ns']}:{lb['code']}")
                
                if lb['ns'] == 'okpd':
                    self.stats['okpd_found'] += 1
                elif lb['ns'] == 'gost':
                    self.stats['gost_found'] += 1
            
            # 2) Классификация если ОКПД не найден
            if not self.has_okpd(cur, doc_id):
                code, conf, desc, info = classify_factory_item(body)
                
                # Дополнительная канонизация для надёжности
                code = okpd_canon(code) or code
                
                if code != 'UNSURE' and conf >= min_conf:
                    # Присваиваем ОКПД с высокой уверенностью
                    self.attach_label(cur, doc_id, 'okpd', code, conf)
                    result['labels_added'].append(f"okpd:{code}")
                    self.stats['okpd_guessed'] += 1
                    logger.debug(f"Doc {doc_id}: assigned OKPD {code} (conf={conf:.2f})")
                else:
                    # Сохраняем догадку в meta для ручной проверки
                    result['okpd_guess'] = {
                        'code': code,
                        'conf': conf,
                        'desc': desc
                    }
                    
                    if code == 'UNSURE':
                        self.stats['unsure'] += 1
                    
                    cur.execute("""
                        UPDATE docs
                        SET meta = COALESCE(meta, '{}'::jsonb) || %s
                        WHERE id = %s
                    """, (Json({'okpd_guess': result['okpd_guess']}), doc_id))
                
                # 3) Сохранение информации о материале
                if save_materials and info:
                    result['material_info'] = info
                    self.save_material_info(cur, doc_id, info)
                    if info.get('material'):
                        self.stats['materials_found'] += 1
            
            if result['labels_added']:
                self.stats['tagged'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing doc {doc_id}: {e}")
            self.stats['errors'] += 1
            return result
    
    def process_batch(self, limit: int = 1000, offset: int = 0, 
                      min_conf: float = 0.90, save_materials: bool = True,
                      filter_untagged: bool = False, include_parents: bool = True,
                      max_analyze_chars: int = 3000):
        """Обработка пакета документов с фильтрацией родителей"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Формируем запрос с учётом фильтрации родителей
            query = """
                SELECT d.id, d.section, d.text
                FROM docs d
            """
            
            # Фильтр родительских документов
            where_clauses = []
            if not include_parents:
                where_clauses.append("COALESCE(d.is_parent, false) = false")
            
            # Фильтр непромаркированных
            if filter_untagged:
                query += """
                    LEFT JOIN docs_labels dl ON d.id = dl.doc_id
                    LEFT JOIN labels l ON dl.label_id = l.id AND l.ns = 'okpd'
                """
                where_clauses.append("l.id IS NULL")
            
            # Добавляем WHERE если есть условия
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += """
                ORDER BY d.id
                LIMIT %s OFFSET %s
            """
            
            cur.execute(query, (limit, offset))
            rows = cur.fetchall()
            
            logger.info(f"Processing {len(rows)} documents (offset={offset})")
            
            for row in rows:
                self.process_document(
                    cur, row['id'], row['section'], row['text'],
                    min_conf, save_materials, max_analyze_chars
                )
                self.stats['processed'] += 1
                
                # Периодический коммит (если не dry-run)
                if not self.dry_run and self.stats['processed'] % 100 == 0:
                    self.conn.commit()
                    logger.info(f"Processed: {self.stats['processed']}, "
                              f"Tagged: {self.stats['tagged']}, "
                              f"OKPD found: {self.stats['okpd_found']}, "
                              f"OKPD guessed: {self.stats['okpd_guessed']}")
            
            if not self.dry_run:
                self.conn.commit()
            else:
                self.conn.rollback()
                logger.info("Dry run - changes rolled back")
    
    def generate_report(self) -> str:
        """Генерация отчета о работе с процентом автоприсвоений"""
        auto_rate = self.stats['okpd_guessed'] / max(self.stats['processed'], 1) * 100
        success_rate = self.stats['tagged'] / max(self.stats['processed'], 1) * 100
        
        report = f"""
=== OKPD Tagging Report ===
Processed documents: {self.stats['processed']}
Tagged documents: {self.stats['tagged']}

Labels found:
- OKPD (explicit): {self.stats['okpd_found']}
- OKPD (guessed): {self.stats['okpd_guessed']}
- GOST: {self.stats['gost_found']}
- Materials: {self.stats['materials_found']}

Unsure classifications: {self.stats['unsure']}
Errors: {self.stats['errors']}

Success rate: {success_rate:.1f}%
Auto-OKPD rate: {auto_rate:.1f}%
        """
        return report

def analyze_untagged(conn):
    """Анализ непромаркированных документов"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Документы без ОКПД
        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN meta->>'okpd_guess' IS NOT NULL THEN 1 END) as with_guess
            FROM docs d
            LEFT JOIN docs_labels dl ON d.id = dl.doc_id
            LEFT JOIN labels l ON dl.label_id = l.id AND l.ns = 'okpd'
            WHERE l.id IS NULL
        """)
        stats = cur.fetchone()
        
        logger.info(f"Untagged documents: {stats['total']} "
                   f"(with guess: {stats['with_guess']})")
        
        # Топ неуверенных догадок
        cur.execute("""
            SELECT 
                meta->'okpd_guess'->>'code' as code,
                meta->'okpd_guess'->>'desc' as description,
                AVG((meta->'okpd_guess'->>'conf')::float) as avg_conf,
                COUNT(*) as count
            FROM docs
            WHERE meta->'okpd_guess'->>'code' IS NOT NULL
              AND meta->'okpd_guess'->>'code' != 'UNSURE'
            GROUP BY 1, 2
            ORDER BY count DESC
            LIMIT 20
        """)
        
        logger.info("\nTop uncertain classifications:")
        for row in cur.fetchall():
            logger.info(f"  {row['code']}: {row['count']} docs "
                       f"(avg conf={row['avg_conf']:.2f}) - {row['description']}")

def retag_low_confidence(conn, old_threshold: float = 0.85, new_threshold: float = 0.90,
                         dry_run: bool = False):
    """Перетегирование документов с низкой уверенностью с заменой старых слабых меток"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT dl.doc_id, d.section, d.text, dl.conf, l.code as old_code
            FROM docs_labels dl
            JOIN docs d ON d.id = dl.doc_id
            JOIN labels l ON l.id = dl.label_id
            WHERE l.ns = 'okpd' 
              AND dl.conf < %s
              AND dl.conf >= %s
            ORDER BY dl.conf
        """, (new_threshold, old_threshold))
        
        rows = cur.fetchall()
        logger.info(f"Found {len(rows)} documents for retagging")
        
        tagger = OKPDTagger(conn, dry_run)
        improved = 0
        
        for row in rows:
            code, conf, desc, info = classify_factory_item(
                f"{row['section'] or ''}\n{row['text'] or ''}"[:3000]
            )
            
            # Канонизация для надёжности
            code = okpd_canon(code) or code
            
            if conf > row['conf'] and conf >= new_threshold:
                # Удаляем старые слабые OKPD-метки перед вставкой новой
                cur.execute("""
                    DELETE FROM docs_labels dl
                    USING labels l
                    WHERE dl.doc_id = %s
                      AND dl.label_id = l.id
                      AND l.ns = 'okpd'
                      AND dl.conf < %s
                """, (row['doc_id'], conf))
                
                tagger.attach_label(cur, row['doc_id'], 'okpd', code, conf)
                improved += 1
                logger.debug(f"Doc {row['doc_id']}: improved {row['old_code']} "
                           f"({row['conf']:.2f}) -> {code} ({conf:.2f})")
        
        if not dry_run:
            conn.commit()
            logger.info(f"Improved {improved} classifications")
        else:
            conn.rollback()
            logger.info(f"Dry run - would improve {improved} classifications")

def main():
    ap = argparse.ArgumentParser(description="OKPD tagging for factory documents")
    ap.add_argument("--limit", type=int, default=1000,
                   help="Number of documents to process")
    ap.add_argument("--offset", type=int, default=0,
                   help="Offset for batch processing")
    ap.add_argument("--min-conf", type=float, default=0.90,
                   help="Minimum confidence for auto-assignment")
    ap.add_argument("--save-materials", action="store_true",
                   help="Save material information to meta")
    ap.add_argument("--filter-untagged", action="store_true",
                   help="Process only untagged documents")
    ap.add_argument("--include-parents", action="store_true",
                   help="Process parent docs as well")
    ap.add_argument("--max-analyze-chars", type=int, default=3000,
                   help="Max characters to analyze per doc")
    ap.add_argument("--analyze", action="store_true",
                   help="Analyze untagged documents")
    ap.add_argument("--retag", action="store_true",
                   help="Retag low confidence documents")
    ap.add_argument("--batch-size", type=int, default=1000,
                   help="Batch size for processing")
    ap.add_argument("--dry-run", action="store_true",
                   help="Do not write changes to database")
    ap.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging")
    
    args = ap.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        with psycopg2.connect(**DB) as conn:
            if args.analyze:
                analyze_untagged(conn)
            elif args.retag:
                retag_low_confidence(conn, old_threshold=0.85, 
                                   new_threshold=args.min_conf,
                                   dry_run=args.dry_run)
            else:
                tagger = OKPDTagger(conn, dry_run=args.dry_run)
                
                # Обработка в батчах для больших объемов
                if args.limit > args.batch_size:
                    total_processed = 0
                    offset = args.offset
                    
                    while total_processed < args.limit:
                        batch_limit = min(args.batch_size, args.limit - total_processed)
                        tagger.process_batch(
                            limit=batch_limit,
                            offset=offset,
                            min_conf=args.min_conf,
                            save_materials=args.save_materials,
                            filter_untagged=args.filter_untagged,
                            include_parents=args.include_parents,
                            max_analyze_chars=args.max_analyze_chars
                        )
                        total_processed += batch_limit
                        offset += batch_limit
                else:
                    tagger.process_batch(
                        limit=args.limit,
                        offset=args.offset,
                        min_conf=args.min_conf,
                        save_materials=args.save_materials,
                        filter_untagged=args.filter_untagged,
                        include_parents=args.include_parents,
                        max_analyze_chars=args.max_analyze_chars
                    )
                
                print(tagger.generate_report())
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()