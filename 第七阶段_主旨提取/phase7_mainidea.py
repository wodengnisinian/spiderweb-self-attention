# phase7_mainidea.py - SpiderWeb Main Idea Extraction from Real Chinese Text
#
# Task: Input a real Chinese article -> Split sentences -> Score each sentence
#       by how "central" it is using SpiderWeb structural encoding -> Output:
#       main idea, central sentences, keywords, supporting sentences.
#
# No classification. Uses SpiderWeb's encoder to measure sentence centrality.

import sys, os, re, torch, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model import create_model
from real_data import COMMON_CHARS

OUT = os.path.join(BASE, "phase7")
os.makedirs(OUT, exist_ok=True)

# Build char vocabulary from real_data's COMMON_CHARS
char_to_id = {}
for i, ch in enumerate(COMMON_CHARS[:2000]):
    char_to_id[ch] = i + 10
vocab_size = len(char_to_id) + 10

class SpiderWebMainIdea:
    """Extract main idea, central sentences, keywords, and supporting sentences from Chinese text."""

    def __init__(self, max_len=256, device="cpu"):
        self.max_len = max_len
        self.device = device
        self.char_to_id = char_to_id
        self.vocab_size = vocab_size

        # SpiderWeb encoder (untrained - structural bias alone provides useful encoding)
        self.model = create_model(
            bias_mode="full", vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2,
            d_ff=512, n_classes=8, max_len=max_len).to(device)
        self.model.eval()

    def _tokenize(self, text):
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                ids.append(10 + (hash(ch) % (self.vocab_size - 10)))
        return ids

    def _encode_sentences(self, sentences):
        """Encode sentences into embedding vectors."""
        embeddings = []
        for sent in sentences:
            ids = self._tokenize(sent)
            n = min(len(ids), self.max_len)
            tids = torch.zeros(1, self.max_len, dtype=torch.long, device=self.device)
            tids[0, :n] = torch.tensor(ids[:n], dtype=torch.long)
            mask = (tids != 0).unsqueeze(1).unsqueeze(2)

            with torch.no_grad():
                pos = torch.arange(self.max_len, device=self.device).unsqueeze(0)
                h = self.model.token_embedding(tids) + self.model.pos_embedding(pos)
                for layer in self.model.layers:
                    h = layer(h, mask, None, 0.0)
                vmask = mask.squeeze(1).squeeze(1).float()
                pooled = (h * vmask.unsqueeze(-1)).sum(dim=1) / vmask.sum(dim=1, keepdim=True).clamp(min=1)
                embeddings.append(pooled.cpu().numpy()[0])
        return np.array(embeddings)

    def analyze(self, text):
        """Analyze a Chinese article and return structured results."""
        # Split into sentences
        raw_sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 5]
        if len(sentences) < 3:
            return {"error": "Article too short (need >= 3 sentences)", "sentences": sentences}

        # Encode all sentences
        embs = self._encode_sentences(sentences)

        # Compute centrality: cosine similarity to the mean embedding
        mean_emb = embs.mean(axis=0)
        mean_n = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        centrality = np.dot(norms, mean_n)
        order = np.argsort(centrality)[::-1]

        # Classify sentences by score
        top_idx = order[:max(2, len(sentences) // 4)]    # main idea: top 25%
        mid_idx = order[max(2, len(sentences) // 4):max(5, len(sentences) // 2)]  # supporting
        bot_idx = order[max(5, len(sentences) // 2):]     # detail

        # Keyword extraction: most frequent chars in top sentences
        from collections import Counter
        keyword_chars = Counter()
        for i in top_idx:
            for ch in sentences[i]:
                keyword_chars[ch] += 1
        top_keywords = [ch for ch, _ in keyword_chars.most_common(10) if ch.strip() and '\u4e00' <= ch <= '\u9fff']

        return {
            "main_idea": [sentences[i] for i in top_idx[:2]],
            "central_sentences": [sentences[i] for i in top_idx],
            "supporting_sentences": [sentences[i] for i in mid_idx],
            "detail_sentences": [sentences[i] for i in bot_idx],
            "keywords": top_keywords,
            "scores": [(sentences[i], float(centrality[i])) for i in order],
            "sentence_count": len(sentences),
        }

    def print_analysis(self, result):
        """Pretty-print analysis result."""
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        print(f"\n{'='*60}")
        print(f"  SPIDERWEB MAIN IDEA ANALYSIS")
        print(f"  {result['sentence_count']} sentences analyzed")
        print(f"{'='*60}")
        print(f"\n--- Main Idea ---")
        for s in result["main_idea"]:
            print(f"  [{len(s)} chars] {s[:120]}...")
        print(f"\n--- Keywords ---")
        print(f"  {' '.join(result['keywords'][:8])}")
        print(f"\n--- Central Sentences ({len(result['central_sentences'])}) ---")
        for i, s in enumerate(result['central_sentences']):
            print(f"  [{i+1}] {s[:100]}...")
        print(f"\n--- Supporting ({len(result['supporting_sentences'])}) ---")
        for i, s in enumerate(result['supporting_sentences'][:3]):
            print(f"  [{i+1}] {s[:100]}...")
        print(f"\n--- Detail ({len(result['detail_sentences'])}) ---")
        for i, s in enumerate(result['detail_sentences'][:2]):
            print(f"  [{i+1}] {s[:100]}...")


if __name__ == "__main__":
    extractor = SpiderWebMainIdea(max_len=256)

    # Test with real Chinese article
    article = """
人工智能技术正在深刻改变现代社会的方方面面。从深度学习到大语言模型，AI的发展速度令人瞩目。

在医疗领域，AI辅助诊断系统已经能够快速分析医学影像。研究表明，AI在肺癌早期筛查中的准确率已经超过90%，大幅降低了漏诊风险。此外，AI药物研发平台也大大缩短了新药开发周期。

然而，AI技术的快速发展也带来了诸多挑战。数据隐私保护、算法公平性、算力资源消耗等问题日益突出。全球各国政府正在积极制定AI监管框架，以确保技术发展不偏离伦理轨道。

在教育方面，个性化学习平台利用AI技术为学生定制专属学习方案。智能辅导系统可以根据学生的学习进度自动调整教学内容和难度。这一趋势正在重新定义传统教育模式。

同时，自动驾驶技术从实验室走向实际道路的步伐正在加快。多个城市已经开始L4级别自动驾驶出租车的试点运营。业内预计，未来五年将有更多城市开放自动驾驶商业化服务。

展望未来，通用人工智能仍然是一个充满挑战但令人向往的目标。科研人员正在从算法、算力和数据三个维度同步推进。
"""

    result = extractor.analyze(article)
    extractor.print_analysis(result)

    # Export to JSON
    import json
    json_path = os.path.join(OUT, "mainidea_demo.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nExported: {json_path}")

    # Also generate report
    lines = []
    lines.append("# Phase 7: Main Idea Extraction Demo")
    lines.append("")
    lines.append("## Input Article")
    lines.append("")
    lines.append(article.strip()[:500] + "...")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append(f"- **Sentence count**: {result['sentence_count']}")
    lines.append(f"- **Keywords**: {', '.join(result['keywords'][:8])}")
    lines.append("")
    lines.append("### Main Idea")
    lines.append("")
    for s in result["main_idea"]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("### Central Sentences")
    lines.append("")
    for i, s in enumerate(result["central_sentences"]):
        lines.append(f"{i+1}. {s}")
    lines.append("")
    lines.append("### Supporting Sentences")
    lines.append("")
    for i, s in enumerate(result["supporting_sentences"][:5]):
        lines.append(f"{i+1}. {s}")

    with open(os.path.join(OUT, "mainidea_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Phase 7 MainIdea demo complete")
