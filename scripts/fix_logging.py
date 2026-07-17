"""Batch replace print() with logger calls across all app modules."""
import re

for fname in ['app/retriever.py', 'app/rag.py', 'app/ntes_client.py', 'app/pnr_client.py']:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'from app.logger import' in content:
        print(f'{fname}: already has logger import, skipping')
        continue

    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i
            break

    for i in range(insert_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('import ') and not line.startswith('from ') and not line.startswith('#'):
            insert_idx = i
            break

    module_name = fname.replace('/', '.').replace('.py', '')
    logger_lines = ['from app.logger import get_logger', f'logger = get_logger("{module_name}")', '']
    lines = lines[:insert_idx] + logger_lines + lines[insert_idx:]
    content = '\n'.join(lines)

    # Replace prints with appropriate log levels
    content = re.sub(r'print\(f"\[WARN\]', 'logger.warning(f"[WARN]', content)
    content = re.sub(r'print\("\[WARN\]', 'logger.warning("[WARN]', content)
    content = re.sub(r'print\(f"\[ERROR\]', 'logger.error(f"[ERROR]', content)
    content = re.sub(r'print\("\[ERROR\]', 'logger.error("[ERROR]', content)
    content = re.sub(r'print\(f"\[EXACT\]', 'logger.debug(f"[EXACT]', content)
    content = re.sub(r'print\(f"\[KEYWORD\]', 'logger.debug(f"[KEYWORD]', content)
    content = re.sub(r'print\(f"\[REWRITE\]', 'logger.debug(f"[REWRITE]', content)
    content = re.sub(r'print\(f"\[INTENT\]', 'logger.debug(f"[INTENT]', content)
    # Remaining bracketed prints -> info
    content = re.sub(r'(?<!\.)print\(f"\[', 'logger.info(f"[', content)
    content = re.sub(r'(?<!\.)print\("\[', 'logger.info("[', content)
    # Generic prints -> info
    content = re.sub(r'(?<!\.)print\(f"', 'logger.info(f"', content)
    content = re.sub(r'(?<!\.)print\("', 'logger.info("', content)

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

    count = content.count('logger.')
    print(f'{fname}: {count} logger calls added')
