import os, json
from core.config import PROJECT_ROOT

VECTOR_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector')
print(f"✅ 修正路径: {VECTOR_DIR}")


def fix_folder1(sub):
    folder = os.path.join(VECTOR_DIR, sub)
    for name in ('metadata.json',):
        path = os.path.join(folder, name)
        if not os.path.exists(path): continue
        data = json.load(open(path, 'r', encoding='utf-8'))
        for m in data:
            for key in ('source_file', 'image_path'):
                val = m.get(key)
                if val and os.path.isabs(val):
                    m[key] = os.path.relpath(val, PROJECT_ROOT)
        json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def fix_folder2(sub):
    folder = os.path.join(VECTOR_DIR, sub)
    for name in ('metadata.json',):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        data = json.load(open(path, 'r', encoding='utf-8'))
        for m in data:
            for key in ('source_file', 'image_path'):
                val = m.get(key)
                if not val:
                    continue
                # 转换为相对路径（如果原路径为绝对路径）
                if os.path.isabs(val):
                    val = os.path.relpath(val, PROJECT_ROOT)
                # 处理路径中的'database'前添加'data'
                normalized_val = os.path.normpath(val)
                parts = normalized_val.split(os.sep)
                new_parts = []
                for part in parts:
                    if part == 'database':
                        new_parts.append('data')
                    new_parts.append(part)
                # 使用反斜杠作为路径分隔符
                new_val = '\\'.join(new_parts)
                # 确保替换所有正斜杠为反斜杠
                new_val = new_val.replace('/', '\\')
                m[key] = new_val
        json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


