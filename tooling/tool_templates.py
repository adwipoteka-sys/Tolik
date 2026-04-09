from __future__ import annotations

from textwrap import dedent, indent

from tooling.tool_spec import ToolSpec


def render_text_summarizer(spec: ToolSpec) -> str:
    max_sentences = int(spec.parameters.get("max_sentences", 3))
    variant = str(spec.parameters.get("variant", "stable"))
    if variant == "counts_raw_inputs":
        limit_block = f"limit = {max_sentences}"
        source_count_line = "source_count = len(texts)"
    elif variant == "blank_input_guard":
        limit_block = dedent(
            f"""
            runtime_limit = payload.get(\"max_sentences\", {max_sentences})
            if isinstance(runtime_limit, bool):
                runtime_limit = {max_sentences}
            elif isinstance(runtime_limit, (int, float)):
                runtime_limit = int(runtime_limit)
            elif isinstance(runtime_limit, str):
                cleaned_limit = runtime_limit.strip()
                if cleaned_limit.lstrip("-").isdigit():
                    runtime_limit = int(cleaned_limit)
                else:
                    runtime_limit = {max_sentences}
            else:
                runtime_limit = {max_sentences}
            limit = max(0, min({max_sentences}, runtime_limit))
            """
        ).strip()
        source_count_line = "source_count = len(normalized)"
    else:
        limit_block = f"limit = {max_sentences}"
        source_count_line = "source_count = len(normalized)"

    limit_block = indent(limit_block, "    ")
    source_count_line = indent(source_count_line, "    ")
    body = f'''
def run_tool(payload):
    texts = payload.get("texts", [])
    if not isinstance(texts, list):
        raise ValueError("texts must be a list")
    normalized = []
    for item in texts:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    if not normalized:
        return {{"summary": "", "source_count": 0, "sentences_used": 0}}
    text_blob = " ".join(normalized)
    prepared = text_blob.replace("!", ".").replace("?", ".")
    chunks = [chunk.strip() for chunk in prepared.split(".") if chunk.strip()]
{limit_block}
    selected = chunks[:limit]
    summary = ". ".join(selected)
    if summary and not summary.endswith("."):
        summary = summary + "."
{source_count_line}
    return {{
        "summary": summary,
        "source_count": source_count,
        "sentences_used": len(selected),
    }}
'''
    return dedent(body).strip() + "\n"


def render_keyword_extractor(spec: ToolSpec) -> str:
    top_k = int(spec.parameters.get("top_k", 5))
    stopwords = [
        "and", "the", "with", "that", "this", "from", "into", "for", "you", "your",
        "как", "это", "что", "для", "или", "при", "так", "она", "они", "его",
    ]
    return dedent(
        f'''
        def run_tool(payload):
            text = str(payload.get("text", ""))
            prepared = []
            for symbol in text.lower():
                prepared.append(symbol if symbol.isalnum() or symbol == " " else " ")
            words = [word for word in "".join(prepared).split() if len(word) > 2]
            stopwords = {stopwords!r}
            counts = {{}}
            for word in words:
                if word in stopwords:
                    continue
                counts[word] = counts.get(word, 0) + 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            keywords = [word for word, _count in ranked[:{top_k}]]
            return {{"keywords": keywords, "unique_terms": len(counts)}}
        '''
    ).strip() + "\n"


def render_numeric_stats(spec: ToolSpec) -> str:
    return dedent(
        '''
        def run_tool(payload):
            values = payload.get("values", [])
            if not isinstance(values, list):
                raise ValueError("values must be a list")
            numbers = [float(value) for value in values]
            if not numbers:
                return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
            total = sum(numbers)
            return {
                "count": len(numbers),
                "mean": total / len(numbers),
                "min": min(numbers),
                "max": max(numbers),
            }
        '''
    ).strip() + "\n"


TEMPLATE_RENDERERS = {
    "text_summarizer": render_text_summarizer,
    "keyword_extractor": render_keyword_extractor,
    "numeric_stats": render_numeric_stats,
}
