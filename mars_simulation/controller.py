
import json
import pandas as pd
from typing import List, Tuple, Optional

from .components import SpeakerSelector, RoleSampler, TeacherAgent
from .generation import BatchDialogueGenerator
from .memory import CanvasManager

BATCH_SIZE = 3  # 减小批次，提升对话连贯性

# 默认阶段定义
DEFAULT_STAGES = [
    ("主题确定", "讨论并确定创作的主题、风格。"),
    ("内容扩充", "讨论并确定画面中的具体元素、构图、色彩。"),
    ("分工合作", "讨论并确定每个人的绘画任务。"),
    ("检查与润色", "检查画面，进行最后的修改与润色。"),
]


class MarsController:
    """
    主控制器。
    agents_data 可传入 Excel 路径（str）或已预处理的 DataFrame。
    role_bias   可传入角色偏置字典（用于不同典型组）。
    stage_defs  可传入自定义阶段列表，每项 (stage_name, stage_desc)。
    """
    MAX_CONSECUTIVE = 1  # 实验组：同一人不允许连续发言

    def __init__(
        self,
        matrix_path,               # str | pd.DataFrame（兼容旧用法）
        agents_data,               # str | pd.DataFrame
        initial_dist_path=None,    # str | pd.Series（兼容旧用法）
        role_bias: dict = None,
        stage_defs: list = None,
        transition_matrix_df: pd.DataFrame = None,
        initial_dist_series: pd.Series = None,
    ):
        # 兼容：如果传了预读的 DataFrame 就直接用，否则从文件读
        if transition_matrix_df is None:
            transition_matrix_df = pd.read_excel(matrix_path, index_col=0, sheet_name=1, engine='openpyxl')
        if initial_dist_series is None:
            initial_dist_df     = pd.read_csv(initial_dist_path, index_col='role')
            initial_dist_series = initial_dist_df['frequency']

        self.speaker_selector = SpeakerSelector(agents_data)
        self.role_sampler     = RoleSampler(transition_matrix_df, initial_dist_series, role_bias=role_bias)
        self.teacher_agent    = TeacherAgent()
        self.batch_gen        = BatchDialogueGenerator(self.speaker_selector.all_agents_data)

        self.stage_defs = stage_defs if stage_defs else DEFAULT_STAGES

    # ── stage helpers ────────────────────────────────────────────────────────

    def _get_stage(self, current_turn: int, total_turns: int) -> Tuple[str, str, str]:
        ratio = current_turn / total_turns
        if ratio <= 0.30:
            idx = 0
        elif ratio <= 0.60:
            idx = 1
        elif ratio <= 0.90:
            idx = 2
        else:
            idx = 3
        stage_code = f"S{idx+1}"
        name, desc = self.stage_defs[idx]
        return stage_code, name, desc

    # ── sequence pre-generation ──────────────────────────────────────────────

    def _pregenerate_sequence(self, group_agent_names: List[str], dialogue_turns: int) -> List[Tuple[int, str, str]]:
        full_sequence = []
        turn_caps = self._calc_turn_caps(group_agent_names, dialogue_turns)
        agent_states = {
            name: {'turns_since_last_spoke': 0, 'total_turns': 0, 'has_spoken': False}
            for name in group_agent_names
        }
        consecutive_counts = {name: 0 for name in group_agent_names}
        current_role = None

        for i in range(1, dialogue_turns + 1):
            excluded_roles = set()
            if i == 1:
                excluded_roles.update({'社交支持者', '反思评价者', '争吵促进者'})
            target_role = self.role_sampler.sample_next_role(current_role, excluded_roles)

            excluded_speakers = {name for name, cnt in consecutive_counts.items() if cnt >= self.MAX_CONSECUTIVE}
            speaker = self.speaker_selector.select_speaker(
                target_role, agent_states, turn_caps, excluded_speakers,
            )
            full_sequence.append((i, speaker, target_role))

            for name in consecutive_counts:
                consecutive_counts[name] = consecutive_counts[name] + 1 if name == speaker else 0
            for name in agent_states:
                agent_states[name]['turns_since_last_spoke'] = (
                    0 if name == speaker else agent_states[name]['turns_since_last_spoke'] + 1
                )
            agent_states[speaker]['total_turns'] += 1
            agent_states[speaker]['has_spoken'] = True
            current_role = target_role

        return full_sequence

    def _calc_turn_caps(self, group_agent_names: List[str], dialogue_turns: int) -> dict:
        group_data = self.speaker_selector.all_agents_data.loc[group_agent_names]
        if 'speaking_count' in group_data.columns and group_data['speaking_count'].sum() > 0:
            counts      = group_data['speaking_count']
            proportions = counts / counts.sum()
            return (proportions * dialogue_turns).round().astype(int).to_dict()
        return {name: round(dialogue_turns / len(group_agent_names)) for name in group_agent_names}

    def _split_into_plan(self, full_sequence: list, dialogue_turns: int) -> list:
        plan  = []
        batch = []
        batch_phase = None

        def flush_batch():
            nonlocal batch
            if batch:
                plan.append(("batch", batch_phase, batch))
                batch = []

        for g_turn, speaker, role in full_sequence:
            current_phase, _, _ = self._get_stage(g_turn, dialogue_turns)
            if batch_phase is None:
                batch_phase = current_phase
            elif current_phase != batch_phase:
                flush_batch()
                plan.append(("transition", batch_phase, current_phase))
                batch_phase = current_phase
            batch.append((g_turn, speaker, role))
            if len(batch) >= BATCH_SIZE:
                flush_batch()
        flush_batch()
        return plan

    # ── main simulation ──────────────────────────────────────────────────────

    def run_simulation(
        self,
        group_agent_names: List[str],
        dialogue_turns: int,
        output_file_path: str,
        task_name: str = "设计一个创意生态角",
        should_stop=None,
    ):
        print(f"--- 正在为小组: {', '.join(group_agent_names)} 进行模拟 ({dialogue_turns} 轮) ---")
        full_seq = self._pregenerate_sequence(group_agent_names, dialogue_turns)
        plan     = self._split_into_plan(full_seq, dialogue_turns)

        SUBSTANTIVE_ROLES = {'知识贡献者', '想法提出者', '协调组织者', '反思评价者', '争吵促进者', '干扰者'}
        conversation_history: list = []
        output_lines: list         = []
        canvas_manager             = CanvasManager()
        cancelled                  = False

        # 开场白
        opening      = self.teacher_agent.opening(task_name)
        opening_line = f"[教师] 老师：{opening}"
        print(opening_line)
        output_lines.append(opening_line)
        conversation_history.append({'speaker': '老师', 'utterance': opening, 'role': 'Teacher'})

        for item in plan:
            if should_stop and should_stop():
                print("--- 模拟已取消 ---")
                cancelled = True
                break

            if item[0] == 'transition':
                _, from_phase, to_phase = item
                stage_idx  = int(to_phase[1]) - 1
                s_name, s_desc = self.stage_defs[stage_idx]
                trans_text = self.teacher_agent.phase_transition(s_name, s_desc)
                t_line     = f"[教师] 老师：{trans_text}"
                print(t_line)
                output_lines.append(t_line)
                conversation_history.append({'speaker': '老师', 'utterance': trans_text, 'role': 'Teacher'})

            else:
                _, phase, batch_turns = item
                stage_code, stage_name, stage_desc = self._get_stage(batch_turns[0][0], dialogue_turns)
                recent       = [t for t in conversation_history if t['role'] != 'Teacher'][-8:]
                batch_dialogue = []

                for global_turn, speaker, role, utterance in self.batch_gen.generate_batch(
                    group_names=group_agent_names,
                    turn_sequence=batch_turns,
                    recent_history=recent,
                    group_facts=canvas_manager.get_canvas_state(),
                    task_name=task_name,
                    stage_name=stage_name,
                    stage_desc=stage_desc,
                ):
                    if should_stop and should_stop():
                        print("--- 模拟已取消 ---")
                        cancelled = True
                        break
                    output_line = f"{global_turn}/{dialogue_turns} {speaker}（{role}）：{utterance}"
                    print(output_line)
                    output_lines.append(output_line)
                    entry = {'speaker': speaker, 'utterance': utterance, 'role': role}
                    conversation_history.append(entry)
                    if role in SUBSTANTIVE_ROLES:
                        batch_dialogue.append(entry)

                if batch_dialogue:
                    canvas_manager.update_canvas(batch_dialogue)
                    canvas_state = canvas_manager.get_canvas_state()
                    last_turn    = batch_turns[-1][0]
                    state_line   = f"[画面状态] {last_turn}/{dialogue_turns}：{json.dumps(canvas_state, ensure_ascii=False)}"
                    print(state_line)
                    output_lines.append(state_line)

        status = "已取消" if cancelled else "完成"
        print(f"\n--- 模拟{status}. 结果已保存至 {output_file_path} ---")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        return conversation_history
