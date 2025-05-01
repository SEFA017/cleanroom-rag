import os
import json
from core.config import PROJECT_ROOT

VECTOR_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector')


def process_metadata(sub_folder, fix_data=True, remove_dots=True):
    """
    元数据处理总入口
    :param sub_folder: 子文件夹名称 ('text'/'img')
    :param fix_data: 是否修复data目录
    :param remove_dots: 是否移除前导..
    """
    folder = os.path.join(VECTOR_DIR, sub_folder)
    metadata_path = os.path.join(folder, 'metadata.json')

    if not os.path.exists(metadata_path):
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        for key in ('source_file', 'image_path'):
            val = item.get(key)
            if not val:
                continue

            # 处理绝对路径
            if os.path.isabs(val):
                val = os.path.relpath(val, PROJECT_ROOT)

            # 路径标准化
            val = val.replace('/', '\\')

            # 功能1: 添加data目录
            if fix_data:
                val = val.replace('database', 'data\\database', 1)

            # 功能2: 移除前导..
            if remove_dots:
                while val.startswith('..\\'):
                    val = val[3:]

            item[key] = val

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 独立函数封装
def fix_data_directory():
    """只添加data目录不处理前导.."""
    process_metadata('text', fix_data=True, remove_dots=False)
    process_metadata('img', fix_data=True, remove_dots=False)
    print("✅ 数据目录修正完成")


def remove_leading_dots():
    """只移除前导..不修改data目录"""
    process_metadata('text', fix_data=False, remove_dots=True)
    process_metadata('img', fix_data=False, remove_dots=True)
    print("✅ 前导点修正完成")


def full_fix():
    """同时执行两种修正"""
    process_metadata('text')
    process_metadata('img')
    print("✅ 完整路径修正完成")


if __name__ == '__main__':
    # 示例调用方式（根据实际需求选择）：
    # fix_data_directory()    # 只执行添加data目录
    # remove_leading_dots()   # 只移除前导..
    remove_leading_dots()  # 执行完整修正