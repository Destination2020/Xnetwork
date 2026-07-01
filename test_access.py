"""
直接测试文件访问
"""
import os
import sys

# 测试不同的路径格式
base_dir = "d:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/AEExperiment"

print("测试文件访问")
print("="*60)

# 列出目录内容
print(f"\n目录：{base_dir}")
print("目录内容:")
try:
    items = os.listdir(base_dir)
    docx_files = [f for f in items if f.endswith('.docx')]
    print(f"  找到 {len(docx_files)} 个 docx 文件:")
    for f in docx_files:
        fpath = os.path.join(base_dir, f)
        size = os.path.getsize(fpath)
        print(f"    - {f} ({size:,} 字节)")
except Exception as e:
    print(f"  错误：{e}")

print("\n" + "="*60)
