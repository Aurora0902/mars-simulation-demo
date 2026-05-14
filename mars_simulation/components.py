import pandas as pd
import numpy as np
from typing import List, Optional

class RoleSampler:
    """
    角色采样器：根据马尔可夫链（角色转换矩阵）决定下一轮的目标角色。
    支持 role_bias 参数，对转换矩阵列进行缩放，实现不同组类型的角色分布差异。
    """
    def __init__(self, transition_matrix_df: pd.DataFrame, initial_dist_series: pd.Series,
                 role_bias: dict = None):
        base_matrix = self._normalize_matrix(transition_matrix_df)
        base_dist   = self._normalize_series(initial_dist_series)

        if role_bias:
            base_matrix = self._apply_col_bias(base_matrix, role_bias)
            base_dist   = self._apply_series_bias(base_dist, role_bias)

        self.transition_matrix   = self._normalize_matrix(base_matrix)
        self.roles               = self.transition_matrix.index.tolist()
        self.initial_distribution = self._normalize_series(base_dist)

    # ── normalizers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_matrix(matrix_df: pd.DataFrame) -> pd.DataFrame:
        return matrix_df.div(matrix_df.sum(axis=1).replace(0, 1), axis=0)

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        return series / series.sum()

    # ── bias helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_col_bias(matrix_df: pd.DataFrame, bias: dict) -> pd.DataFrame:
        """Scale columns of the transition matrix by bias factors, then each row will be re-normalised."""
        result = matrix_df.copy().astype(float)
        for role, factor in bias.items():
            if role in result.columns:
                result[role] = result[role] * factor
        return result

    @staticmethod
    def _apply_series_bias(series: pd.Series, bias: dict) -> pd.Series:
        result = series.copy().astype(float)
        for role, factor in bias.items():
            if role in result.index:
                result[role] = result[role] * factor
        return result

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample_next_role(self, previous_role: Optional[str], excluded_roles: set = set()) -> str:
        if previous_role not in self.roles:
            probs = self.initial_distribution.copy()
            for role in excluded_roles:
                if role in probs:
                    probs[role] = 0
            probs = self._normalize_series(probs)
            return np.random.choice(probs.index, p=probs.values)

        probs = self.transition_matrix.loc[previous_role].copy()
        for role in excluded_roles:
            if role in probs:
                probs[role] = 0
        probs = self._normalize_series(probs)
        return np.random.choice(self.roles, p=probs.values)

    def get_initial_role(self) -> str:
        return np.random.choice(self.initial_distribution.index, p=self.initial_distribution.values)

    def reset(self):
        pass


class SpeakerSelector:
    """
    发言者选择器：根据加权公式决定下一个发言人。
    S_i = (RoleFit_i + SilenceBoost_i) * SaturationPenalty_i

    agents_data_source 可以是：
      - str  : Excel 文件路径（原始用法，向后兼容）
      - DataFrame : 已经过预处理的 DataFrame（索引为 name，列含角色概率中文名）
    """
    ROLE_COLS = {
        'r1_prob': '知识贡献者', 'r2_prob': '想法提出者', 'r3_prob': '协调组织者',
        'r4_prob': '反思评价者', 'r5_prob': '社交支持者', 'r6_prob': '消极参与者',
        'r7_prob': '争吵促进者', 'r8_prob': '干扰者',
    }

    def __init__(self, agents_data_source):
        if isinstance(agents_data_source, pd.DataFrame):
            self.all_agents_data = agents_data_source
        else:
            self.all_agents_data = self._load_and_preprocess(agents_data_source)

    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.rename(columns=self.ROLE_COLS, inplace=True)
        df['score_percentile'] = df.groupby('group')['score'].rank(pct=True) * 100
        df.set_index('name', inplace=True)
        return df

    def select_speaker(
        self,
        target_role: str,
        agents_states: dict,
        turn_caps: dict,
        excluded: list = [],
    ) -> str:
        current_group_data = self.all_agents_data.loc[list(agents_states.keys())]
        candidates = [n for n in current_group_data.index if n not in excluded]
        if not candidates:
            return np.random.choice(list(agents_states.keys()))

        scores = []
        for name in candidates:
            role_fit = float(current_group_data.loc[name, target_role])

            BOOST_WEIGHT = 0.5
            turns_silent  = agents_states[name]['turns_since_last_spoke']
            silence_boost = BOOST_WEIGHT * 0.049 * (turns_silent ** 0.55) if agents_states[name]['has_spoken'] else 0.0

            PENALTY_WEIGHT  = 0.5
            current_turns   = agents_states[name]['total_turns']
            target_turns    = turn_caps.get(name, 1)
            saturation      = current_turns / target_turns if target_turns > 0 else float(current_turns)
            saturation_penalty = 1 / (1 + saturation**(2 * PENALTY_WEIGHT))

            scores.append((role_fit + silence_boost) * saturation_penalty)

        total_score  = sum(scores)
        probabilities = [s / total_score for s in scores] if total_score > 0 else [1 / len(candidates)] * len(candidates)
        return str(np.random.choice(candidates, p=probabilities))


class TeacherAgent:
    """教师智能体：开场白与阶段转换（纯模板，不调用 LLM）。"""

    _DEFAULT_TASK = "设计一个创意生态角"

    def opening(self, task_name: str) -> str:
        if task_name == self._DEFAULT_TASK:
            return (f"同学们，今天我们要一起合作完成「{task_name}」，最后要画出一张完整的作品。"
                    f"先讨论一下，你们想画什么主题、什么风格，把大方向定下来。")
        return (f"同学们，今天我们要一起合作完成「{task_name}」。"
                f"先讨论一下你们的想法，把大方向定下来。")

    def phase_transition(self, stage_name: str, stage_desc: str) -> str:
        return f"好，上一个环节结束了。接下来进入「{stage_name}」阶段——{stage_desc}"
