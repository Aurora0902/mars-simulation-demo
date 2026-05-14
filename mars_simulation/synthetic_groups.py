"""
典型合作学习小组定义（用于替代真实学生数据）

角色偏置与个人特征依据（基于外部权威合作学习文献 + 真实学生数据参考）：

一、个人特征参考说明
  成绩分值（score）使用与真实数据相同的量纲（约200-290，总分制）。
  avg_utterance_length 参考真实学生数据范围（约7-17字），按性别/能力/组类型设定。
  speaking_count 参考真实学生数据（约30-155），影响模拟中的发言比例。
  role_weights（r1-r8）依据真实Logit模型的预测规律估算：
    - 主要驱动因素：性别（女→r3协调组织者↑；男→r2想法提出者↑，r8干扰者↑）
    - 好友同组（hasF）：男生→r8↑↑；女生→r4↑，r3略降
    - 学业成绩：影响有限且角色特异性强（本研究Logit结果）
  具体参考学生（data_with_prob.xlsx）：
    - M,noF,高分:  张卓(281,r2=0.223,r4=0.207,r8=0.099)
    - M,noF,中分:  陆冉熠(265.5,r2=0.224,r4=0.213,r8=0.122)
    - M,hasF,中分: 马琸骁(263.5,r2=0.202,r4=0.208,r8=0.255)
    - M,hasF,低分: 倪铭骏(201.5,r2=0.136,r4=0.197,r8=0.399) ← 低分男生有好友的极端情况
    - F,noF,高分:  王琪文(287,r3=0.314,r4=0.211,r1=0.128)
    - F,noF,中分:  赵锦润(261,r3=0.339,r4=0.222)、唐钟毓(276.5,r3=0.324,r4=0.216)
    - F,hasF,FF:   刘书航(277.5,r3=0.233,r4=0.253,r8=0.060)

二、能力分组文献
- Webb (1982/1989/2002): 高能力生→知识贡献/想法主导；低能力组帮助供给崩溃
- Gillies & Ashman (2000): 低能力同质组主要特征为消极退缩，并非干扰为主
- Saleh et al. (2005): 异质组中等能力生受抑制/边缘化，消极参与增加
- Lou et al. (1996): 中等同质组接近基准分布

三、地位/友伴文献（已移除105/106组，本文件只保留四类典型组）
"""

import pandas as pd
import numpy as np

ROLE_COL_NAMES = [
    '知识贡献者', '想法提出者', '协调组织者', '反思评价者',
    '社交支持者', '消极参与者', '争吵促进者', '干扰者',
]


def _norm(weights):
    arr = np.array(weights, dtype=float)
    return (arr / arr.sum()).tolist()


# ──────────────────────────────────────────────────
# 四类典型组定义
# role_bias: 角色转换矩阵列向量缩放系数（>1=升高该角色出现频率，<1=降低）
# role_weights 顺序: [知识,想法,协调,反思,社交,消极,争吵,干扰] = r1-r8
# ──────────────────────────────────────────────────
GROUP_DEFS = {
    101: {
        'label': '高成就同质组',
        'type': 'high_ability',
        'description': '成员均为高学业成就学生（班级成绩排名前20%）',
        'literature_basis': 'Webb (1982, 1989, 2002); Fuchs et al. (1998); Lou et al. (1996)',
        # role_bias 依据（外部文献）：
        # - Webb (1982/1989): 高能力生主导解释/知识贡献 → 知识贡献者↑、想法提出者↑
        # - Fuchs et al. (1998): 高能力同质组存在真实认知冲突 → 争吵促进者中度偏高
        # - Webb (2002): 若规范不良，高能力生互相竞争 → 协调组织者不过度拔高
        # - 高能力生整体仍有少量消极参与（不代表零脱离）
        'role_bias': {
            '知识贡献者': 1.65, '想法提出者': 1.45, '协调组织者': 1.25,
            '反思评价者': 1.25, '社交支持者': 0.85,
            '消极参与者': 0.45, '争吵促进者': 1.30, '干扰者': 0.55,
        },
        'members': [
            {
                'name': '李明轩', 'gender': 'M', 'score': 283,
                'speaking_count': 70, 'friend_in_group': '张雨涵',
                'avg_utterance_length': 12.0,
                # M, 异性好友, 高分：参考王安越(M,280,hasF) + 跨性别衰减(75%M-noF + 25%M-hasF)
                # 主导：r2想法提出者, r4反思评价者；r8干扰者中等偏高（好友效应部分）
                'role_weights': _norm([0.089, 0.221, 0.157, 0.208, 0.099, 0.037, 0.066, 0.123]),
            },
            {
                'name': '张雨涵', 'gender': 'F', 'score': 281,
                'speaking_count': 62, 'friend_in_group': '李明轩',
                'avg_utterance_length': 14.0,
                # F, 异性好友, 高分：参考王琪文(F,287,noF) + 跨性别好友衰减(75%F-noF + 25%F-F)
                # 主导：r3协调组织者, r4反思评价者
                'role_weights': _norm([0.127, 0.144, 0.305, 0.220, 0.113, 0.010, 0.050, 0.031]),
            },
            {
                'name': '王俊杰', 'gender': 'M', 'score': 280,
                'speaking_count': 65, 'friend_in_group': '',
                'avg_utterance_length': 12.0,
                # M, 无好友, 高分：直接参考张卓(M,281,noF)
                # 主导：r2想法提出者, r4反思评价者；r8干扰者中等
                'role_weights': _norm([0.093, 0.223, 0.170, 0.207, 0.102, 0.036, 0.071, 0.099]),
            },
            {
                'name': '陈思雨', 'gender': 'F', 'score': 285,
                'speaking_count': 58, 'friend_in_group': '',
                'avg_utterance_length': 15.0,
                # F, 无好友, 高分：直接参考王琪文(F,287,noF)
                # 主导：r3协调组织者, r4反思评价者；r8干扰者极低
                'role_weights': _norm([0.128, 0.145, 0.314, 0.211, 0.115, 0.010, 0.055, 0.022]),
            },
        ],
    },

    102: {
        'label': '低成就同质组',
        'type': 'low_ability',
        'description': '成员均为低学业成就学生（班级成绩排名后20%）',
        'literature_basis': 'Gillies & Ashman (2000); Webb (1989); Lou et al. (1996)',
        # role_bias 依据（外部文献）：
        # - Gillies & Ashman (2000): 低能力同质组核心特征是"消极退缩"和"无人回应的求助"
        # - Webb (1989): 低能力组帮助供给崩溃后学生陷入沉默，非转向干扰
        # - 干扰者不应过高；消极参与者才是主要负向行为
        # 注：基础矩阵消极参与者列权重仅2.6%，系数3.0使其稳态约~10%；
        # 干扰者提升至0.70（男生低分好友效应，不说话时偶发干扰）
        'role_bias': {
            '知识贡献者': 0.55, '想法提出者': 0.70, '协调组织者': 0.50,
            '反思评价者': 0.75, '社交支持者': 1.20,
            '消极参与者': 3.00, '争吵促进者': 0.50, '干扰者': 0.70,
        },
        'members': [
            {
                'name': '赵小东', 'gender': 'M', 'score': 215,
                'speaking_count': 38, 'friend_in_group': '周浩',
                'avg_utterance_length': 9.0,
                # M, 同性好友, 低分：参考倪铭骏(M,201.5,hasF,r8=0.399)，分数略高故r8稍低
                # 主导：r8干扰者（男-男好友叠加低分效应）
                'role_weights': _norm([0.060, 0.140, 0.110, 0.198, 0.062, 0.032, 0.013, 0.385]),
            },
            {
                'name': '周浩', 'gender': 'M', 'score': 208,
                'speaking_count': 32, 'friend_in_group': '赵小东',
                'avg_utterance_length': 8.0,
                # M, 同性好友, 低分：更接近倪铭骏(r8=0.399)
                'role_weights': _norm([0.058, 0.136, 0.108, 0.197, 0.058, 0.031, 0.012, 0.400]),
            },
            {
                'name': '吴小云', 'gender': 'F', 'score': 218,
                'speaking_count': 45, 'friend_in_group': '',
                'avg_utterance_length': 10.0,
                # F, 无好友, 低分：成绩效应有限，接近F-noF基准；r6消极略高
                # 主导：r3协调组织者（女生效应）；r6消极参与者略升
                'role_weights': _norm([0.122, 0.134, 0.316, 0.218, 0.107, 0.013, 0.046, 0.044]),
            },
            {
                'name': '孙佳颖', 'gender': 'F', 'score': 213,
                'speaking_count': 40, 'friend_in_group': '',
                'avg_utterance_length': 9.0,
                # F, 无好友, 低分：同吴小云
                'role_weights': _norm([0.120, 0.132, 0.317, 0.217, 0.105, 0.014, 0.045, 0.050]),
            },
        ],
    },

    103: {
        'label': '能力异质组',
        'type': 'mixed_ability',
        'description': '高中低能力混合（1高成就＋2中等＋1低成就），研究推荐最优分组方式',
        'literature_basis': 'Saleh et al. (2005); Webb (1982, 1989); Lou et al. (1996)',
        # role_bias 依据（外部文献）：
        # - Saleh et al. (2005): 高能力生→知识贡献/领导；
        #   中等能力生→被边缘化/抑制，消极参与升高；低能力生→被动接受
        # - Webb (1982/1989): 高能力生主导解释，低/中能力生求助常被忽视
        # - 消极参与者不应设为极低（中低能力生受压制导致退缩）
        'role_bias': {
            '知识贡献者': 1.50, '想法提出者': 1.15, '协调组织者': 1.20,
            '反思评价者': 1.10, '社交支持者': 1.15,
            '消极参与者': 0.90, '争吵促进者': 0.85, '干扰者': 0.60,
        },
        'members': [
            {
                'name': '刘浩然', 'gender': 'M', 'score': 283,
                'speaking_count': 78, 'friend_in_group': '',
                'avg_utterance_length': 15.0,
                # M, 无好友, 高分：直接参考张卓(M,281,noF)
                'role_weights': _norm([0.093, 0.223, 0.170, 0.207, 0.102, 0.036, 0.071, 0.099]),
            },
            {
                'name': '何欣', 'gender': 'F', 'score': 268,
                'speaking_count': 62, 'friend_in_group': '林宇',
                'avg_utterance_length': 12.0,
                # F, 异性好友, 中等成绩：75%F-noF + 25%F-F好友效应
                # 主导：r3协调组织者, r4反思评价者
                'role_weights': _norm([0.126, 0.143, 0.307, 0.225, 0.112, 0.011, 0.044, 0.032]),
            },
            {
                'name': '林宇', 'gender': 'M', 'score': 262,
                'speaking_count': 58, 'friend_in_group': '何欣',
                'avg_utterance_length': 11.0,
                # M, 异性好友, 中等偏低成绩：参考马琸骁(M,263.5,hasF,r8=0.255)
                # 主导：r8干扰者偏高（男性+好友效应）
                'role_weights': _norm([0.071, 0.202, 0.106, 0.208, 0.093, 0.033, 0.032, 0.255]),
            },
            {
                'name': '朱晓燕', 'gender': 'F', 'score': 218,
                'speaking_count': 42, 'friend_in_group': '',
                'avg_utterance_length': 9.0,
                # F, 无好友, 低分：同低成就组女生估计
                'role_weights': _norm([0.122, 0.134, 0.316, 0.218, 0.107, 0.013, 0.046, 0.044]),
            },
        ],
    },

    104: {
        'label': '中等同质组',
        'type': 'medium_ability',
        'description': '成员均为中等学业成就学生（班级成绩排名30%-70%）',
        'literature_basis': 'Lou et al. (1996); Webb (1989)',
        # role_bias 依据（外部文献）：
        # - Lou et al. (1996): 中等能力同质组接近总体基准分布，无极端偏向
        # - Webb (1989): 中等同质组求助行为有一定回应，但知识供给深度不及高能力组
        # - 整体最接近"基准"，任务角色与非任务角色基本均衡
        'role_bias': {
            '知识贡献者': 0.90, '想法提出者': 0.95, '协调组织者': 1.00,
            '反思评价者': 1.00, '社交支持者': 1.10,
            '消极参与者': 1.05, '争吵促进者': 0.90, '干扰者': 1.00,
        },
        'members': [
            {
                'name': '徐子涵', 'gender': 'M', 'score': 265,
                'speaking_count': 68, 'friend_in_group': '',
                'avg_utterance_length': 12.0,
                # M, 无好友, 中等成绩：直接参考陆冉熠(M,265.5,noF)
                'role_weights': _norm([0.086, 0.224, 0.177, 0.213, 0.119, 0.021, 0.038, 0.122]),
            },
            {
                'name': '马晓彤', 'gender': 'F', 'score': 261,
                'speaking_count': 58, 'friend_in_group': '沈梦琪',
                'avg_utterance_length': 11.0,
                # F, 同性好友, 中等成绩：参考刘书航(F,277.5,F-F好友,r3=0.233,r4=0.253)
                'role_weights': _norm([0.125, 0.157, 0.233, 0.253, 0.114, 0.015, 0.043, 0.060]),
            },
            {
                'name': '沈梦琪', 'gender': 'F', 'score': 268,
                'speaking_count': 63, 'friend_in_group': '马晓彤',
                'avg_utterance_length': 12.0,
                # F, 同性好友, 中等成绩：同马晓彤，略高成绩
                'role_weights': _norm([0.126, 0.153, 0.228, 0.257, 0.113, 0.015, 0.043, 0.065]),
            },
            {
                'name': '郑博文', 'gender': 'M', 'score': 260,
                'speaking_count': 60, 'friend_in_group': '',
                'avg_utterance_length': 11.0,
                # M, 无好友, 中等成绩：参考陆冉熠/宓轩磊(M,275.5,noF)
                'role_weights': _norm([0.087, 0.226, 0.175, 0.211, 0.117, 0.023, 0.044, 0.117]),
            },
        ],
    },
}


def build_agents_dataframe() -> pd.DataFrame:
    """
    将 GROUP_DEFS 转换为 SpeakerSelector 可直接使用的 DataFrame。
    列：name(index), group, gender, score, speaking_count, friend_in_group,
        avg_utterance_length, score_percentile, 8个角色概率列(Chinese names)。
    """
    rows = []
    for gid, gdef in GROUP_DEFS.items():
        for m in gdef['members']:
            row = {
                'name':               m['name'],
                'group':              gid,
                'gender':             m['gender'],
                'score':              m['score'],
                'speaking_count':     m['speaking_count'],
                'friend_in_group':    m.get('friend_in_group', ''),
                'avg_utterance_length': m['avg_utterance_length'],
            }
            for col, val in zip(ROLE_COL_NAMES, m['role_weights']):
                row[col] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    # 用全局百分位，确保高成就组成员不会因组内比较而被标记为"成绩中等"
    df['score_percentile'] = df['score'].rank(pct=True) * 100
    df.set_index('name', inplace=True)
    return df
