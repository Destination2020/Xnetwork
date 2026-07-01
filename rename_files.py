"""
重命名文件为英文名称 - 使用目录列表方法
"""
import os
from pathlib import Path

base_dir = Path(r"d:\Work_files\EUV-OCT\XRR\XRR_Software\DeepLearning\XRR_FCNN\AEExperiment")

# 获取所有 docx 文件
docx_files = list(base_dir.glob("*.docx"))

print("当前目录的 docx 文件:")
print("="*60)
for i, f in enumerate(docx_files, 1):
    print(f"{i}. {f.name}")

print("\n准备重命名:")
print("="*60)

# 创建重命名计划
rename_plan = []

for f in docx_files:
    name = f.name
    if "llj" in name and "四稿" in name:
        new_name = "llj_patent_v4.docx"
        rename_plan.append((f, new_name))
    elif "五稿" in name and "物理信息" in name:
        new_name = "patent_v5.docx"
        rename_plan.append((f, new_name))
    elif "专利草稿" in name and "我" in name:
        new_name = "my_patent_draft.docx"
        rename_plan.append((f, new_name))

# 执行重命名
for old_path, new_name in rename_plan:
    new_path = old_path.parent / new_name
    try:
        old_path.rename(new_path)
        print(f"✓ {old_path.name} -> {new_name}")
    except Exception as e:
        print(f"✗ 重命名失败 {old_path.name}: {e}")

print("="*60)

# 验证结果
print("\n重命名后的文件:")
docx_files = list(base_dir.glob("*.docx"))
for f in docx_files:
    print(f"  - {f.name}")
