
import os
import re
import random
from typing import List, Tuple
import pandas as pd
from openai import OpenAI
from mars_simulation.data_structures import Student
from mars_simulation.fallbacks import fallback_for_role

ROLE_HINTS = {
    "知识贡献者": "说出一个关于任务的具体事实或信息，别解释太完整。",
    "想法提出者": "提出一个新点子或新方法，一句话就行。",
    "协调组织者": "简单安排分工、整合意见，或把话题拉回任务。",
    "反思评价者": "对刚才的想法或内容表达困惑、质疑或追问。",
    "社交支持者": "极短地表示肯定或赞同。",
    "消极参与者": "极短地表达不想参与或觉得自己没用。",
    "争吵促进者": "反驳或嘲讽刚才某人说的话。",
    "干扰者": "说一句跟任务完全无关的闲话或搞笑内容。",
}

ROLE_MAX_CHARS = {
    "知识贡献者": 30,
    "想法提出者": 30,
    "协调组织者": 30,
    "反思评价者": 30,
    "社交支持者": 10,
    "消极参与者": 15,
    "争吵促进者": 20,
    "干扰者": 20,
}

# 真实五年级课堂发言示例
ROLE_STYLE_EXAMPLES = {
    "知识贡献者": [
        "夹竹桃是有毒的", "每天给它浇点水", "水的多层的",
        "潮湿吧 因为雨林比较潮湿的呀", "有的鱼可以有两三年的好吗",
        "红色花瓣是非常吉祥的", "蝙蝠鱼会把这些鱼吃掉的",
    ],
    "想法提出者": [
        "加个水", "画一个大大的池塘", "我要画一个鱼缸",
        "咱们搞个生态盒怎么样", "在瀑布里面放小虾",
        "要不再画一层吧", "未来主义咋样",
    ],
    "协调组织者": [
        "先画", "你们选好了没", "我们来投票吧",
        "咱们商量一下吧", "每个人都要画",
        "谁画的好啊 你们两个谁来", "先讨论好吧 先讨论",
    ],
    "反思评价者": [
        "这是什么", "不同意", "印象派啥意思我都不知道",
        "那跟未来主义有什么关系啊", "星云那我们做不到啊",
        "草原太大众了", "你这个貌似不咋现实",
    ],
    "社交支持者": [
        "好", "行", "可以",
        "对啊 肯定是啊", "有道理", "可以 这个可以",
    ],
    "消极参与者": [
        "我摆烂", "随便", "我不画你们画就行",
        "画不出来", "不用问了 我都同意",
        "那你们俩画吧", "我画画很拉的",
    ],
    "争吵促进者": [
        "你有病啊", "你烦死了", "不会说就不要说",
        "那你就别来我们组了", "闭嘴 就这些吗",
        "你不要提 不会提就别瞎提", "你画的乱七八糟的",
    ],
    "干扰者": [
        "我觉得我画个厕所吧", "咱们画蛋仔派对吧",
        "我在画太极八卦图", "警报 植物袭击人类",
        "这好像个烤面包机啊", "我上次画了两大坨奥利给",
        "我要画一个小猪佩奇的房子",
    ],
}

# 真实课堂连贯对话片段
REAL_DIALOGUE_SAMPLE = [
    "张卓：来个赛博 来一个未来主义 它们是赛博",
    "张懿恬：我也未来主义 未来主义",
    "刘书航：我比较喜欢清爽一点的",
    "宓轩磊：未来主义",
    "张懿恬：我也是",
    "张卓：我比较喜欢科技与狠活的",
    "张懿恬：那要不就定未来主义吧",
    "刘书航：行",
    "张卓：高楼大厦 有一点偏蓝色系",
    "张懿恬：那关键词是什么",
    "宓轩磊：豪华大床",
    "张懿恬：什么玩意啊",
]


class BatchDialogueGenerator:
    """
    完整系统的对话生成器（batch=3，链式回应提示词）。

    核心改进：每批发言中，第 N 轮明确被提示"直接回应第 N-1 轮"，
    消除自问自答和自我反驳问题。
    """

    def __init__(self, agents_data: pd.DataFrame):
        keys = [k.strip() for k in os.environ.get("DEEPSEEK_API_KEYS", "").split(",") if k.strip()]
        if not keys:
            # 兼容旧的单 key 环境变量
            single = os.environ.get("DEEPSEEK_API_KEY", "")
            if not single:
                raise ValueError("DEEPSEEK_API_KEY 或 DEEPSEEK_API_KEYS 环境变量未设置。")
            keys = [single]
        self._api_keys = keys
        # 预创建固定客户端池，复用 TCP 连接，不再每批新建
        from httpx import Timeout
        self._timeout = Timeout(connect=15.0, read=45.0, write=15.0, pool=15.0)
        self._clients = [
            OpenAI(api_key=k, base_url="https://api.deepseek.com/v1", timeout=self._timeout)
            for k in keys
        ]
        self._client_idx = 0
        self.model_name  = "deepseek-chat"
        self.agents_data = agents_data

    def _next_client(self) -> OpenAI:
        """轮换复用预创建的客户端，分散限流压力。"""
        client = self._clients[self._client_idx % len(self._clients)]
        self._client_idx += 1
        return client

    def _build_member_profiles(self, group_names: List[str]) -> str:
        profiles = [Student(self.agents_data.loc[name]).profile_text for name in group_names]
        return "\n".join(f"- {name}：{profile}" for name, profile in zip(group_names, profiles))

    @staticmethod
    def _fallback(role: str) -> str:
        return fallback_for_role(role)

    def _retry_remaining(
        self,
        remaining: List[Tuple[int, str, str]],
        group_names: List[str],
        recent_history: List[dict],
        group_facts: List[str],
        task_name: str,
        stage_name: str,
        stage_desc: str,
    ):
        """对未能生成的轮次补一次，仍缺失时才用 fallback 兜底。"""
        MAX_RETRIES = 1
        pending = list(remaining)  # 还未填充的轮次

        for attempt in range(1, MAX_RETRIES + 1):
            if not pending:
                return

            seq_text = self._build_chained_seq_text(pending, recent_history)
            history_text = "\n".join(
                f"{t['speaker']}：{t['utterance']}" for t in recent_history
            ) if recent_history else "（对话刚开始）"

            prompt = f"""你是小学五年级课堂小组讨论的对话生成器。
只生成【需要生成的发言】里列出的这些轮次。每一行只写这一行指定学生当前要说的话。

【任务】{task_name}
【当前阶段】{stage_name}：{stage_desc}
【最近对话】
{history_text}
【需要生成的发言（严格按顺序，每行一条，格式：姓名：内容）】
{seq_text}
【严格规则】
只输出当前指定发言人的当前一句话。
直接输出："""

            try:
                resp = self._next_client().chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你只输出本次指定轮次的发言。每行只能有一个学生姓名和一句话。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self._calc_max_tokens(pending),
                    temperature=0.7,
                    stream=False,
                )
                lines = (resp.choices[0].message.content or "").strip().split("\n")
                still_pending = []
                fill_idx = 0
                for line in lines:
                    if fill_idx >= len(pending):
                        break
                    utt = self._parse_line(line)
                    if utt:
                        g, spk, role = pending[fill_idx]
                        yield g, spk, role, self._repair_utterance(
                            utt, spk, role, recent_history, task_name, stage_name, stage_desc,
                        )
                        fill_idx += 1
                still_pending = pending[fill_idx:]
                pending = still_pending
                if pending:
                    print(f"  [retry {attempt}] 还剩 {len(pending)} 轮未生成，继续重试…")
            except Exception as e:
                print(f"\n[retry {attempt}] API 调用失败: {e}，继续重试…")

        # 超过最大重试次数才用 fallback
        for g, spk, role in pending:
            print(f"  [fallback] 第{g}轮 {spk}（{role}）所有重试耗尽")
            yield g, spk, role, self._fallback(role)

    def _parse_line(self, line: str) -> str:
        line  = line.strip()
        line  = re.sub(r"^[\s\-*]*(?:第?\d+\s*[轮.、:：)]\s*)?", "", line)
        match = re.match(r"[^:：]+[:：]\s*(.+)", line)
        if match:
            return match.group(1).strip().strip('"""')
        return line.strip().strip('"""')

    def _calc_max_tokens(self, turns: List) -> int:
        return 280 + len(turns) * 80

    def _max_chars_for(self, speaker: str, role: str) -> int:
        role_cap = ROLE_MAX_CHARS.get(role, 22)
        try:
            mean_len = float(self.agents_data.loc[speaker].get("avg_utterance_length", role_cap))
            speaker_cap = max(16, int(round(mean_len * 2.5)))
            return min(role_cap, speaker_cap)
        except Exception:
            return role_cap

    def _length_hint_for(self, speaker: str, role: str) -> str:
        max_chars = self._max_chars_for(speaker, role)
        if max_chars <= 8:
            return f"最多{max_chars}字，像真实课堂里随口应一声"
        if max_chars <= 16:
            return f"最多{max_chars}字，半句话也可以"
        return f"尽量{max_chars}字以内，说完整一句短话"

    def _role_style_hint(self, role: str) -> str:
        hint = ROLE_HINTS.get(role, "自然接一句")
        examples = ROLE_STYLE_EXAMPLES.get(role, [])[:7]
        example_text = " / ".join(f"「{e}」" for e in examples)
        if example_text:
            return f"{hint}输出要模仿这些句子的长度和风格：{example_text}。不要解释原因"
        return hint

    def _clean_utterance(self, utterance: str) -> str:
        text = re.sub(r"\s+", " ", utterance).strip()
        text = re.sub(r"\bresponse\b", "", text, flags=re.IGNORECASE).strip()
        names = [re.escape(str(n)) for n in getattr(self.agents_data, "index", [])]
        if names:
            # Sometimes the model packs several "姓名：内容" turns into one line.
            text = re.split(rf"(?:{'|'.join(names)})[:：]", text, maxsplit=1)[0].strip()
        text = text.strip('"""').strip()
        text = text.replace("！", "").replace("!", "")
        return text

    def _looks_like_name_only(self, utterance: str) -> bool:
        text = self._clean_utterance(utterance).strip("，,。！？!? ")
        if len(text) < 2:
            return False
        names = [str(n) for n in getattr(self.agents_data, "index", [])]
        return any(text == name or (len(text) >= 2 and text in name) for name in names)

    def _too_vague_for_role(self, utterance: str, role: str) -> bool:
        text = self._clean_utterance(utterance).strip("，,。！？!? ")
        if role == "社交支持者":
            return False
        vague = {
            "试试这个", "这个可以", "这样也行", "可以吧", "行吧",
            "就这样", "先这样", "也对", "对啊", "好吧",
        }
        return text in vague

    def _repair_utterance(
        self,
        utterance: str,
        speaker: str,
        role: str,
        recent_history: List[dict],
        task_name: str,
        stage_name: str,
        stage_desc: str,
    ) -> str:
        text = self._clean_utterance(utterance)
        if not text or self._looks_like_name_only(text) or self._too_vague_for_role(text, role):
            return self._fallback(role)
        if len(text) > 60:
            return self._fallback(role)
        return text

    def _build_chained_seq_text(
        self,
        turn_sequence: List[Tuple[int, str, str]],
        recent_history: List[dict],
    ) -> str:
        lines = []
        for i, (g, spk, role) in enumerate(turn_sequence):
            role_hint = self._role_style_hint(role)
            length_hint = self._length_hint_for(spk, role)
            if i == 0:
                if recent_history:
                    prev_spk = recent_history[-1]['speaker']
                    chain_hint = f"接续 {prev_spk} 的话，"
                else:
                    chain_hint = ""
            else:
                prev_spk = turn_sequence[i - 1][1]
                if prev_spk == spk:
                    prev_spk = turn_sequence[i - 2][1] if i >= 2 else (
                        recent_history[-1]['speaker'] if recent_history else "上一位同学"
                    )
                chain_hint = f"直接回应 {prev_spk}，"
            lines.append(
                f"第{g}轮 {spk}（{role}）→ {chain_hint}{role_hint}。{length_hint}"
            )
        return "\n".join(lines)

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
        history_text = "\n".join(
            f"{t['speaker']}：{t['utterance']}" for t in recent_history[-8:]
        ) if recent_history else "（对话刚开始）"
        facts_text  = "\n".join(f"- {f}" for f in group_facts) if group_facts else "（暂无）"
        seq_text    = self._build_chained_seq_text(turn_sequence, recent_history)
        max_tokens  = self._calc_max_tokens(turn_sequence)

        DEFAULT_TASK = "设计一个创意生态角"
        task_note = (
            '【注意】只是讨论"画出一张设计图"，不是真实制作。'
        ) if task_name == DEFAULT_TASK else ''

        prompt = f"""你是小学五年级课堂小组讨论的对话生成器。
只生成【接下来...轮发言】里列出的这些轮次。
每一行只写这一行指定学生当前要说的话。重点是模仿真实课堂发言的口吻：短、碎、随口、不完整，不像写作文。

【任务】{task_name}
【当前阶段】{stage_name}：{stage_desc}
{task_note}

【小组成员档案】
{member_profiles}

【当前画面记录（只用来避免重复，已定/已完成的不要继续聊）】
{facts_text}

【最近对话】
{history_text}

【接下来 {len(turn_sequence)} 轮发言（请按顺序生成）】
{seq_text}

【规则】
1. 每行只输出一个人的当前一句话，格式"姓名：发言内容"。
2. 每一轮都优先模仿该角色后面给出的真实例句口吻和长度；可以就像示例一样很短，不完整。
3. 不要反复解释已定内容，上一句已经定了就换到下一个小点、分工或检查。
4. 不要用感叹号。不要每句话都解释原因。

直接输出："""

        system_msg = (
            "你只输出本次指定轮次的发言。"
            "每行只能有一个学生姓名和一句话，格式为「姓名：内容」。"
            "每句自然接续前面，但只说当前指定学生自己的话。"
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
            print(f"\n调用 DeepSeek API 时出错: {e}")
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
                        yield g, spk, role, self._repair_utterance(
                            utterance, spk, role, recent_history, task_name, stage_name, stage_desc
                        )
                        turn_idx += 1
        except Exception as e:
            print(f"\n[stream] 流式传输中断: {e}，已收到 {turn_idx} 轮，剩余走 retry")

        if buffer.strip() and turn_idx < len(turn_sequence):
            utterance = self._parse_line(buffer.strip())
            if utterance:
                g, spk, role = turn_sequence[turn_idx]
                yield g, spk, role, self._repair_utterance(
                    utterance, spk, role, recent_history, task_name, stage_name, stage_desc
                )
                turn_idx += 1

        # 如果还有未填充的轮次，先 retry 一次，再 fallback
        if turn_idx < len(turn_sequence):
            remaining = turn_sequence[turn_idx:]
            for g, spk, role, utt in self._retry_remaining(
                remaining, group_names, recent_history, group_facts,
                task_name, stage_name, stage_desc,
            ):
                yield g, spk, role, utt
