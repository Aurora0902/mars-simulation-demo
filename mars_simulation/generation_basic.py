
import os
import re
import random
from typing import List, Tuple
import pandas as pd
from openai import OpenAI
from mars_simulation.data_structures import Student
from mars_simulation.fallbacks import fallback_for_role
# 复用 generation.py 的全局客户端池
from mars_simulation.generation import _get_next_client

ROLE_HINTS = {
    "知识贡献者": "说出一于任务的具体事实或信息。",
    "想法提出者": "提出新点子或新方法。",
    "协调组织者": "安排分工、整合意见，或把话题拉回任务。",
    "反思评价者": "对刚才的想法或内容表达困惑、质疑或追问。",
    "社交支持者": "表示肯定或赞同。",
    "消极参与者": "表达不想参与或觉得自己没用。",
    "争吵促进者": "反驳或嘲讽刚才某人说的话。",
    "干扰者": "说跟任务完全无关的闲话或搞笑内容。",
}

class BatchDialogueGenerator:
    """
    基础提示词组的对话生成器（对照组）。
    """

    def __init__(self, agents_data: pd.DataFrame):
        self.model_name  = "deepseek-chat"
        self.agents_data = agents_data

    def _next_client(self) -> OpenAI:
        """从全局池轮换取客户端，复用 TCP 连接。"""
        return _get_next_client()

    def _build_member_profiles(self, group_names: List[str]) -> str:
        profiles = [Student(self.agents_data.loc[name]).profile_text for name in group_names]
        return "\n".join(f"- {name}：{profile}" for name, profile in zip(group_names, profiles))

    @staticmethod
    def _fallback(role: str) -> str:
        return fallback_for_role(role)

    def _parse_line(self, line: str) -> str:
        line  = line.strip()
        line  = re.sub(r"^[\s\-*]*(?:第?\d+\s*[轮.、:：)]\s*)?", "", line)
        match = re.match(r"[^:：]+[:：]\s*(.+)", line)
        if match:
            return match.group(1).strip().strip('"""')
        return line.strip().strip('"""')

    def _calc_max_tokens(self, turns: List) -> int:
        return 350 + len(turns) * 90



    def generate_batch(
        self,
        group_names: List[str],
        turn_sequence: List[Tuple[int, str, str]],
        recent_history: List[dict],
        group_facts: List[str],
        task_name: str,
        stage_name: str,
        stage_desc: str,
    ):
        member_profiles = self._build_member_profiles(group_names)
        history_text = "（基础提示词组不提供最近对话）"
        facts_text  = "（基础提示词组不提供画面记录）"
        seq_text    = "\n".join(f"第{g}轮 {spk}（{role}）→ {ROLE_HINTS.get(role, '自然参与')}" for g, spk, role in turn_sequence)
        max_tokens  = self._calc_max_tokens(turn_sequence)

        DEFAULT_TASK = "设计一个创意生态角"
        task_note = '【注意】只是讨论"画出一张设计图"，不是真实制作。' if task_name == DEFAULT_TASK else ''

        prompt = f"""你是小学五年级课堂小组讨论的对话生成器。根据指定的发言顺序生成对话。

【任务】{task_name}
【当前阶段】{stage_name}：{stage_desc}
{task_note}

【小组成员档案】
{member_profiles}

【当前画面记录】
{facts_text}

【最近对话】
{history_text}

【接下来 {len(turn_sequence)} 轮发言（请按顺序生成）】
{seq_text}

【严格规则】
1. 每行只输出"姓名：发言内容"，不加任何其他多余信息。

直接输出："""


        system_msg = (
            "你是一名参与课堂小组讨论的小学五年级学生。"
            "严格每行输出一条发言，格式为「姓名：内容」。"
        )

        client = self._next_client()
        try:
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.75,
                stream=True,
            )
        except Exception as e:
            print(f"\n[basic] 调用 DeepSeek API 时出错: {e}")
            for g, spk, role in turn_sequence:
                yield g, spk, role, self._fallback(role)
            return

        buffer   = ""
        turn_idx = 0
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                buffer += chunk.choices[0].delta.content or ""
                while "\n" in buffer and turn_idx < len(turn_sequence):
                    line, buffer = buffer.split("\n", 1)
                    utterance = self._parse_line(line)
                    if utterance:
                        g, spk, role = turn_sequence[turn_idx]
                        yield g, spk, role, utterance
                        turn_idx += 1
        except Exception as e:
            print(f"\n[basic stream] 流式传输中断: {e}，已收到 {turn_idx} 轮，剩余用 fallback")

        if buffer.strip() and turn_idx < len(turn_sequence):
            utterance = self._parse_line(buffer.strip())
            if utterance:
                g, spk, role = turn_sequence[turn_idx]
                yield g, spk, role, utterance
                turn_idx += 1

        # fallback 填充剩余
        for i in range(turn_idx, len(turn_sequence)):
            g, spk, role = turn_sequence[i]
            yield g, spk, role, self._fallback(role)
