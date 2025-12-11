def generate_topic_background(topic_title: str, llm_client) -> str:
    """
    将简短的标题扩展为较为丰富的事件背景；LLM 失败时返回简短兜底。
    """
    system_prompt = "你是一个舆情分析师。请根据话题标题构思一个逼真的社交媒体热点事件背景。"
    user_prompt = f"""
话题标题：{topic_title}
请生成一段约 150 字的事件背景描述，包含：
1) 事件起因（谁、在哪里、做了什么）
2) 核心争议点（为什么会吵）
3) 当前舆论情绪（震惊/愤怒/感动/嘲讽等）
直接输出纯文本描述。
"""
    try:
        return llm_client.chat(system_prompt, user_prompt)
    except Exception:
        return f"关于 {topic_title} 的热门讨论，引发了广泛关注。"
