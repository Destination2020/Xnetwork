"""
检查并提取 docx 文件内容
"""
import zipfile
import os

def check_and_extract_docx(file_path):
    """检查并提取 docx 文件的文本内容"""
    print(f"\n{'='*60}")
    print(f"处理：{os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 文件不存在")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"文件大小：{file_size:,} 字节")
    
    # 尝试作为 zip 文件打开
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            
            # 检查是否包含 document.xml
            if 'word/document.xml' not in files:
                print(f"✗ 不是标准的 DOCX 文件 (缺少 word/document.xml)")
                return None
            
            print(f"✓ 是有效的 DOCX 文件")
            
            # 读取 document.xml
            with zip_ref.open('word/document.xml') as doc_file:
                content = doc_file.read().decode('utf-8')
                
                # 简单的 XML 文本提取
                import re
                # 提取<w:t>标签中的文本
                texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', content)
                
                print(f"提取到 {len(texts)} 个文本段落")
                
                # 返回前 20 行
                return texts[:20]
                
    except zipfile.BadZipFile:
        print(f"✗ 不是一个有效的 ZIP 文件")
        return None
    except Exception as e:
        print(f"✗ 读取错误：{e}")
        return None

def main():
    base_dir = r"d:\Work_files\EUV-OCT\XRR\XRR_Software\DeepLearning\XRR_FCNN\AEExperiment"
    
    files = [
        "llj 专利 0226-四稿.docx",
        "1.8-五稿 - 基于物理信息神经网络的 X 射线小角散射图样重构方法.docx",
        "专利草稿 - 我.docx"
    ]
    
    print("="*60)
    print("检查并提取 DOCX 文件内容")
    print("="*60)
    
    for fname in files:
        fpath = os.path.join(base_dir, fname)
        texts = check_and_extract_docx(fpath)
        
        if texts:
            print(f"\n前 20 行内容预览:")
            for i, text in enumerate(texts[:10], 1):
                if text.strip():
                    print(f"  {i}. {text.strip()[:80]}")
    
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)

if __name__ == "__main__":
    main()
