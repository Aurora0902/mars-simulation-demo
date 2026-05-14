import pandas as pd

class Student:
    """学生的数据容器，只存静态数据。"""
    def __init__(self, series: pd.Series):
        self.name: str = series.name
        self.gender: str = str(series.get("gender", "M")).strip()
        self.score_percentile: float = float(series.get("score_percentile", 50.0))
        self.friend: str = str(series.get("friend_in_group", "")).strip()
        self.mean_utterance_length: float = float(series.get("avg_utterance_length", 15.0))

    def _base_profile(self) -> str:
        """性别 + 成绩 + 好友描述（共用部分）。"""
        gender_desc = "你是男生。" if self.gender == "M" else "你是女生。"
        if self.score_percentile > 70:
            score_desc = "你成绩比较好。"
        elif self.score_percentile < 30:
            score_desc = "你成绩不太好。"
        else:
            score_desc = "你成绩中等。"
        has_friend = self.friend and self.friend not in ("/", "nan", "None")
        friend_desc = f"你和组里的{self.friend}是好朋友。" if has_friend else ""
        return " ".join(filter(None, [gender_desc, score_desc, friend_desc]))

    @property
    def profile_text(self) -> str:
        """生成用于 LLM Prompt 的个人简介（实验组/对照组共用，不含发言长度约束）。"""
        return self._base_profile()

    # 保留别名兼容旧调用
    @property
    def short_profile_text(self) -> str:
        return self._base_profile()
