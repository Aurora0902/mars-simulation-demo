
from .controller import MarsController
from .generation_basic import BatchDialogueGenerator


class BasicMarsController(MarsController):
    """
    对照组控制器：与 MarsController 相同的角色采样逻辑，
    但使用"基础提示词"生成器（批量=5，无链式回应指导，无风格优化）。
    MAX_CONSECUTIVE=3 保留连续发言可能性，对照组不做此限制。
    """
    MAX_CONSECUTIVE = 3  # 对照组：允许同一人最多连续3轮发言

    def __init__(
        self,
        matrix_path=None,
        agents_data=None,
        initial_dist_path=None,
        role_bias: dict = None,
        stage_defs: list = None,
        transition_matrix_df=None,
        initial_dist_series=None,
    ):
        super().__init__(matrix_path, agents_data, initial_dist_path,
                         role_bias=role_bias, stage_defs=stage_defs,
                         transition_matrix_df=transition_matrix_df,
                         initial_dist_series=initial_dist_series)
        # 覆盖父类生成器，使用基础版
        self.batch_gen = BatchDialogueGenerator(self.speaker_selector.all_agents_data)
