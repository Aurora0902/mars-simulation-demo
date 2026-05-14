import os
from typing import List
from openai import OpenAI

class CanvasManager:
    """
    任务进展记录管理器：追踪讨论中已确认、未定和被否决的内容，
    为生成器提供上下文，不在 UI 中展示。
    """
    def __init__(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=30.0)
        self.model_name = "deepseek-chat"
        self.canvas_state: List[str] = []

    def get_canvas_state(self) -> List[str]:
        return self.canvas_state

    def update_canvas(self, batch_dialogue: List[dict]):
        if not batch_dialogue:
            return

        old_canvas_text = (
            "\n".join(f"- {item}" for item in self.canvas_state)
            if self.canvas_state else "（尚无记录）"
        )
        dialogue_text = "\n".join(
            f"{t['speaker']}：{t['utterance']}" for t in batch_dialogue
        )

        prompt = (
            "你是一个记录任务讨论状态的助手。根据已有线索和最新讨论，"
            "更新一版简洁的讨论进展记录。\n\n"
            "【记录规则】\n"
            "1. 只保留四类信息：\n"
            "   - [主题] 当前小组大概想做什么\n"
            "   - [已定] 讨论中明确被接受的方案或元素\n"
            "   - [未定] 还在犹豫或争议中的内容\n"
            "   - [已否决] 被明确拒绝或放弃的想法（仅在有明确否决时才写）\n"
            "2. 若最新讨论推翻了已定内容，将其移入已否决。\n"
            "3. 不追踪分工、步骤顺序、完成状态。\n"
            "4. 忽略纯闲聊内容。\n"
            "5. 每条 10 字以内，粗粒度即可。\n"
            "6. 输出格式固定：[标签] 内容，每条一行。\n\n"
            f"【已有线索】\n{old_canvas_text}\n\n"
            f"【最新讨论】\n{dialogue_text}\n\n"
            "【更新后的线索（直接输出，不加解释）】"
        )

        system_msg = "你是讨论状态记录助手，严格按格式输出线索，不输出任何解释。"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            new_canvas_text = response.choices[0].message.content
            self.canvas_state = [
                line.strip() for line in new_canvas_text.split("\n") if line.strip()
            ]
        except Exception as e:
            print(f"\n更新任务进展时调用 LLM 出错: {e}")
