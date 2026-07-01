"""
专利文档格式转换脚本
使用 pathlib 处理路径
"""

import zipfile
import re
from pathlib import Path

def extract_text_from_docx(file_path):
    """从 docx 文件中提取所有文本段落"""
    try:
        print(f"  尝试打开：{file_path}")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            if 'word/document.xml' not in zip_ref.namelist():
                print(f"  ✗ 不是标准的 DOCX 文件")
                return None
            
            with zip_ref.open('word/document.xml') as doc_file:
                content = doc_file.read().decode('utf-8')
                # 提取<w:t>标签中的文本
                texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', content)
                print(f"  ✓ 成功提取 {len(texts)} 个段落")
                return texts
    except Exception as e:
        print(f"  ✗ 读取错误：{e}")
        import traceback
        traceback.print_exc()
        return None

def get_paragraph_structure(texts):
    """分析段落结构，识别标题和正文"""
    structure = []
    for i, text in enumerate(texts):
        text_stripped = text.strip()
        if not text_stripped:
            continue
        
        # 判断是否是标题
        is_heading = False
        level = 0
        
        # 检查常见标题模式
        heading_patterns = [
            r'^第 [一二三四五六七八九十\d]+部分',
            r'^第 [一二三四五六七八九十\d]+章',
            r'^第 [一二三四五六七八九十\d]+节',
            r'^[\d]+[、.．]',
            r'^[一二三四五六七八九十]+[、.．]',
            r'^(权利要求书 | 说明书 | 摘要 | 附图说明)',
            r'^(技术领域 | 背景技术 | 发明内容 | 附图说明 | 具体实施方式)',
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text_stripped):
                is_heading = True
                level = 1
                break
        
        structure.append({
            'index': i,
            'text': text_stripped,
            'is_heading': is_heading,
            'level': level,
            'length': len(text_stripped)
        })
    
    return structure

def print_structure(name, structure, max_lines=30):
    """打印文档结构"""
    print(f"\n{'='*60}")
    print(f"{name} 的文档结构 (前{max_lines}行):")
    print(f"{'='*60}")
    
    for i, item in enumerate(structure[:max_lines], 1):
        marker = "【标题】" if item['is_heading'] else "【正文】"
        text_preview = item['text'][:50] + '...' if len(item['text']) > 50 else item['text']
        print(f"{i:3d}. {marker} {text_preview}")
    
    if len(structure) > max_lines:
        print(f"... 共 {len(structure)} 行")

def save_to_text(structure, output_path):
    """将提取的内容保存为文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in structure:
            f.write(item['text'] + '\n')
    print(f"  ✓ 已保存到：{output_path}")

def main():
    # 使用 pathlib 处理路径
    base_dir = Path(r"d:\Work_files\EUV-OCT\XRR\XRR_Software\DeepLearning\XRR_FCNN\AEExperiment")
    
    files = {
        'llj': "llj 专利 0226-四稿.docx",
        'v5': "1.8-五稿 - 基于物理信息神经网络的 X 射线小角散射图样重构方法.docx",
        'my': "专利草稿 - 我.docx"
    }
    
    print("="*60)
    print("专利文档格式分析")
    print("="*60)
    
    all_structures = {}
    
    # 处理每个文件
    for key, fname in files.items():
        fpath = base_dir / fname
        print(f"\n处理：{fname}")
        
        texts = extract_text_from_docx(str(fpath))
        if texts:
            structure = get_paragraph_structure(texts)
            all_structures[key] = {
                'name': fname,
                'structure': structure,
                'raw_texts': texts
            }
            print_structure(fname, structure)
        else:
            print(f"  ✗ 无法提取文本")
    
    # 输出分析摘要
    print("\n" + "="*60)
    print("格式分析摘要:")
    print("="*60)
    
    for key, data in all_structures.items():
        structure = data['structure']
        headings = [item for item in structure if item['is_heading']]
        
        print(f"\n{data['name']}:")
        print(f"  - 总段落数：{len(structure)}")
        print(f"  - 标题数：{len(headings)}")
        print(f"  - 主要章节:")
        for h in headings[:10]:
            print(f"    * {h.text[:60]}")
    
    # 保存提取的文本
    print("\n" + "="*60)
    print("保存提取的文本:")
    print("="*60)
    
    for key, data in all_structures.items():
        output_name = data['name'].replace('.docx', '.txt')
        output_path = base_dir / output_name
        save_to_text(data['structure'], str(output_path))
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 查看生成的.txt 文件以了解完整内容")
    print("2. 根据参考文档的格式，重新组织您的专利内容")
    print("3. 创建符合新格式要求的专利文档")

if __name__ == "__main__":
    main()
