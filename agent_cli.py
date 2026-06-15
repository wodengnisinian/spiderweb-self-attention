# agent_cli.py - Interactive SpiderWeb Reading Agent
# Paste an article, ask questions, get relevant passages.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_encoder import SpiderWebReader

def main():
    print("=" * 60)
    print("  SpiderWeb 阅读智能体 - v0.1")
    print("  基于 SpiderWeb Self-Attention 文章理解")
    print("=" * 60)
    print()

    # Init reader
    print("初始化模型...")
    reader = SpiderWebReader(max_len=512)
    print("模型就绪。")
    print()

    while True:
        print("=" * 60)
        print("  操作菜单")
        print("  1. 加载文章")
        print("  2. 提问")
        print("  3. 查看文章摘要")
        print("  4. 查看文章段落")
        print("  5. 退出")
        print("=" * 60)

        choice = input("\n请选择 (1-5): ").strip()

        if choice == "1":
            print("\n请粘贴文章内容（输入 END 结束）:")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == "END":
                    break
                lines.append(line)
            article = "\n".join(lines)
            if article.strip():
                reader.load_article(article)
                print(reader.summarize())
            else:
                print("文章为空。")

        elif choice == "2":
            if reader.paragraph_embeddings is None:
                print("请先加载文章（选择 1）。")
                continue
            question = input("\n请输入问题: ").strip()
            if not question:
                continue
            print(f"\n正在检索相关段落...")
            results = reader.ask(question, top_k=3)
            print(f"\n{'='*60}")
            print(f"  Q: {question}")
            print(f"{'='*60}")
            for i, (para, score) in enumerate(results):
                print(f"\n--- 第 {i+1} 段落 (相似度: {score:.3f}) ---")
                print(para.strip())

        elif choice == "3":
            if reader.paragraph_embeddings is None:
                print("请先加载文章（选择 1）。")
                continue
            print(reader.summarize())

        elif choice == "4":
            if reader.paragraph_embeddings is None:
                print("请先加载文章（选择 1）。")
                continue
            print(f"\n文章共 {len(reader.paragraphs)} 个段落:")
            for i, para in enumerate(reader.paragraphs):
                print(f"\n--- 段落 {i+1} ---")
                print(para[:200] + ("..." if len(para) > 200 else ""))

        elif choice == "5":
            print("再见！")
            break

        else:
            print("无效选择，请输入 1-5。")

        print()

if __name__ == "__main__":
    main()
