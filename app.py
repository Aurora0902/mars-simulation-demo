import os
import sys
import json
import queue
import threading
import re
import glob
import traceback
import uuid
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, Response, stream_with_context, jsonify, request

app = Flask(__name__)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MATRIX_PATH    = os.path.join(BASE_DIR, '修正后的角色转换矩阵.xlsx')
INIT_DIST_PATH = os.path.join(BASE_DIR, 'initial_role_distribution.csv')
HISTORY_DIR    = os.path.join(BASE_DIR, 'history')
os.makedirs(HISTORY_DIR, exist_ok=True)

TEST_TURNS = 50  # None → 使用每组 base_turns

# ── 载入合成小组数据（启动时一次性读取，不再每次模拟重复 IO）──────────────
from mars_simulation.synthetic_groups import GROUP_DEFS, build_agents_dataframe

AGENTS_DF = build_agents_dataframe()   # 全局 DataFrame，各控制器共享

# 预读转换矩阵和初始分布，避免每次模拟都 read_excel
TRANSITION_MATRIX_DF = pd.read_excel(MATRIX_PATH, index_col=0, sheet_name=1, engine='openpyxl')
INITIAL_DIST_DF      = pd.read_csv(INIT_DIST_PATH, index_col='role')
INITIAL_DIST_SERIES  = INITIAL_DIST_DF['frequency']


ROLE_COLORS = {
    '知识贡献者': '#4A90D9',
    '想法提出者': '#7B68EE',
    '协调组织者': '#2ECC71',
    '反思评价者': '#F39C12',
    '社交支持者': '#FF69B4',
    '消极参与者': '#95A5A6',
    '争吵促进者': '#E74C3C',
    '干扰者':    '#FF8C00',
}

# ── 辅助 ────────────────────────────────────────────────────────────────────

def _parse_task_config(req) -> dict:
    """从请求 query 参数提取任务配置，未提供时使用默认值。"""
    task_name = req.args.get('task', '设计一个创意生态角').strip() or '设计一个创意生态角'
    stages = []
    defaults = [
        ("主题确定", "讨论并确定创作的主题、风格。"),
        ("内容扩充", "讨论并确定画面中的具体元素、构图、色彩。"),
        ("分工合作", "讨论并确定每个人的绘画任务。"),
        ("检查与润色", "检查画面，进行最后的修改与润色。"),
    ]
    for i, (dn, dd) in enumerate(defaults, 1):
        n = req.args.get(f's{i}name', dn).strip() or dn
        d = req.args.get(f's{i}desc', dd).strip() or dd
        stages.append((n, d))
    return {'task_name': task_name, 'stages': stages}


def _build_group_info() -> dict:
    """从 GROUP_DEFS 构建与前端兼容的 groups dict。"""
    result = {}
    for gid, gdef in GROUP_DEFS.items():
        members_raw = gdef['members']
        base_turns  = sum(m['speaking_count'] for m in members_raw)
        sim_turns   = TEST_TURNS if TEST_TURNS else base_turns

        members = []
        for m in members_raw:
            sc    = m['speaking_count']
            # 用绝对分数阈值（高成就组成员不会因组内比较被标记为"中等"）
            score_label = '成绩较好' if m['score'] >= 275 else ('成绩较差' if m['score'] <= 230 else '成绩中等')
            friend_raw  = m.get('friend_in_group', '')
            members.append({
                'name':         m['name'],
                'display':      m['name'][0] + m['name'][0],
                'gender':       m['gender'],
                'score_label':  score_label,
                'friend_disp':  (friend_raw[0] + friend_raw[0]) if friend_raw else '',
                'speaking_cnt': sc,
                'speaking_pct': round(sc / base_turns * 100) if base_turns else 0,
            })

        result[gid] = {
            'label':       gdef['label'],
            'description': gdef['description'],
            'type':        gdef['type'],
            'members':     members,
            'base_turns':  base_turns,
            'sim_turns':   sim_turns,
        }
    return result


GROUPS = _build_group_info()

_NAME_TO_DISPLAY: dict = {
    m['name']: m['display']
    for g in GROUPS.values() for m in g['members']
}

_NAME_HAS_FRIEND: dict = {
    m['name']: bool(m['friend_disp'])
    for g in GROUPS.values() for m in g['members']
}

_NAME_STARS: dict = {
    m['name']: ('⭐⭐⭐' if m['score_label'] == '成绩较好'
                else '⭐' if m['score_label'] == '成绩较差' else '⭐⭐')
    for g in GROUPS.values() for m in g['members']
}


def display_name(full_name: str) -> str:
    base = _NAME_TO_DISPLAY.get(full_name, full_name[0] + full_name[0] if full_name else full_name)
    heart = ' ❤️' if _NAME_HAS_FRIEND.get(full_name) else ''
    stars = ' ' + _NAME_STARS.get(full_name, '') if full_name in _NAME_STARS else ''
    return f"{base}{heart}{stars}"


# ── SSE helpers ──────────────────────────────────────────────────────────────

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


TURN_RE    = re.compile(r'^(\d+)/(\d+)\s+(.+?)（(.+?)）：(.+)$')
TEACHER_RE = re.compile(r'^\[教师\]\s*老师：(.+)$')
CANVAS_RE  = re.compile(r'^\[画面状态\]\s*(\d+)/(\d+)：(.+)$')
CHECK_RE   = re.compile(r'^\s*\[检查未通过')
HIDDEN_STATUS_RE = re.compile(
    r'^\s*('
    r'发言上限：'
    r'|--- 正在为小组:'
    r'|--- 正在更新(?:备忘录|画面状态)'
    r'|--- 模拟(?:完成|已取消|已保存|已取消)'
    r'|\[retry'       # 重试调试信息
    r'|\[fallback\]'  # fallback 调试信息
    r'|\[stream\]'    # 流中断调试信息
    r'|\[basic'       # basic 组调试信息
    r'|\[画面状态\]'  # 画面状态已由 CANVAS_RE 处理
    r'|调用 DeepSeek'
    r')'
)


# 线程安全的 stdout 捕获机制：每个线程通过 thread-local 路由到自己的队列。
# 旧版本直接替换 sys.stdout，并发请求时会互相破坏 stdout 状态，
# 导致"老师发言后就没下文"的诡异 bug。
_capture_local      = threading.local()
_original_stdout    = sys.stdout


class _StdoutDispatcher:
    """全局 stdout 代理：根据当前线程的 LineCapture 路由写入。"""
    def write(self, text):
        _original_stdout.write(text)
        cap = getattr(_capture_local, 'capture', None)
        if cap is not None:
            cap._handle_write(text)

    def flush(self):
        _original_stdout.flush()


# 安装一次全局 dispatcher
if not isinstance(sys.stdout, _StdoutDispatcher):
    sys.stdout = _StdoutDispatcher()


class LineCapture:
    """把当前线程内的 print 行送入指定队列。线程安全。"""
    def __init__(self, q):
        self.q     = q
        self._buf  = ''
        self._prev = None

    def _handle_write(self, text):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self.q.put(line)

    def __enter__(self):
        self._prev = getattr(_capture_local, 'capture', None)
        _capture_local.capture = self
        return self

    def __exit__(self, *_):
        _capture_local.capture = self._prev
        if self._buf:
            self.q.put(self._buf)


def save_history(group_id: int, turns_used: int, log: list) -> str:
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'{group_id}_{ts}.json'
    info  = GROUPS[group_id]
    data  = {
        'group_id':    group_id,
        'group_label': info['label'],
        'group_type':  info['type'],
        'timestamp':   datetime.now().isoformat(timespec='seconds'),
        'turns_used':  turns_used,
        'members':     info['members'],
        'conversation': log,
    }
    with open(os.path.join(HISTORY_DIR, fname), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return fname


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/groups')
def api_groups():
    return jsonify(GROUPS)


@app.route('/api/recommendations/<int:group_id>')
def api_recommendations(group_id: int):
    """
    基于本次真实模拟数据，调用 LLM 动态生成教学建议。
    必需 query 参数：
      role_counts     = JSON 字符串，{"知识贡献者":8,"干扰者":15,...}
      top_speaker     = 发言最多的学生显示名
      top_speaker_pct = 其发言占比，如 "36%"
      top_speaker_gap = 比第二名多出的轮数
    """
    if group_id not in GROUPS:
        return jsonify({'error': 'unknown group'}), 404

    gtype  = GROUPS[group_id]['type']
    ginfo  = GROUPS[group_id]

    # ── 解析本次模拟统计 ────────────────────────────────────────────
    raw_counts = request.args.get('role_counts', '')
    role_counts = {}
    try:
        role_counts = json.loads(raw_counts) if raw_counts else {}
    except Exception:
        pass

    total = sum(role_counts.values()) or 1
    POS_ROLES = {'知识贡献者', '想法提出者', '协调组织者', '反思评价者'}
    NEG_ROLES = {'消极参与者', '争吵促进者', '干扰者'}
    pos_pct = round(sum(role_counts.get(r, 0) for r in POS_ROLES) / total * 100)
    neg_pct = round(sum(role_counts.get(r, 0) for r in NEG_ROLES) / total * 100)
    soc_pct = round(role_counts.get('社交支持者', 0) / total * 100)
    dominant     = max(role_counts, key=role_counts.get) if role_counts else '—'
    dominant_pct = round(role_counts.get(dominant, 0) / total * 100) if role_counts else 0
    top_speaker     = request.args.get('top_speaker', '')
    top_speaker_pct = request.args.get('top_speaker_pct', '')
    try:
        top_speaker_gap = int(request.args.get('top_speaker_gap', '0') or 0)
    except ValueError:
        top_speaker_gap = 0
    try:
        top_speaker_pct_num = int(str(top_speaker_pct).rstrip('%') or 0)
    except ValueError:
        top_speaker_pct_num = 0

    speaker_focus_is_clear = bool(top_speaker and top_speaker_pct_num >= 35 and top_speaker_gap >= 3)
    if speaker_focus_is_clear:
        speaker_balance_note = (
            f"发言相对集中在{top_speaker}，其发言占{top_speaker_pct}，"
            f"比第二多{top_speaker_gap}轮。"
        )
    elif top_speaker:
        speaker_balance_note = (
            f"{top_speaker}发言最多（{top_speaker_pct}），但领先第二名仅{top_speaker_gap}轮，"
            "只能视为轻微差异，不应解读为主导或压制。"
        )
    else:
        speaker_balance_note = "暂无发言者分布数据。"

    # ── 统计卡片数据（固定结构，快速渲染用）──────────────────────────
    sim_stats = {
        'pos_pct':      pos_pct,
        'neg_pct':      neg_pct,
        'soc_pct':      soc_pct,
        'dominant':     dominant,
        'dominant_pct': dominant_pct,
        'top_speaker':  top_speaker,
        'top_speaker_pct': top_speaker_pct,
        'top_speaker_gap': top_speaker_gap,
        'speaker_focus_is_clear': speaker_focus_is_clear,
        'speaker_balance_note': speaker_balance_note,
        'total':        total,
        'role_counts':  role_counts,
    }

    # ── 调用 LLM 生成个性化建议 ────────────────────────────────────
    role_dist_lines = '\n'.join(
        f"  {role}: {cnt}轮 ({round(cnt/total*100)}%)"
        for role, cnt in sorted(role_counts.items(), key=lambda x: -x[1])
    ) if role_counts else '（无数据）'

    member_lines = '\n'.join(
        f"  {m['display']}：{m['gender']}性，{m['score_label']}"
        + (f"，与{m['friend_disp']}是好友" if m['friend_disp'] else "")
        for m in ginfo['members']
    )

    prompt = f"""你是一位专业的合作学习研究者，正在为一位小学教师提供基于模拟数据的教学建议。

【本次模拟小组】
组型：{ginfo['label']}（{ginfo['description']}）
成员构成：
{member_lines}

【本次模拟角色分布（共{total}轮）】
{role_dist_lines}
积极学习角色合计：{pos_pct}%
消极/冲突角色合计：{neg_pct}%
出现最多的角色：{dominant}（{dominant_pct}%）
发言者分布判断：{speaker_balance_note}

【诊断边界】
- 不要把轻微差异夸大成"占据主导"或"需要干预"。只有当某学生发言占比≥35%，且比第二名至少多3轮时，才可以说发言相对集中。
- 角色占比也是如此：只有某一类消极/冲突角色明显偏高，或积极学习角色明显不足时，才把它作为主要问题。
- 如果本次数据没有明显问题，建议应以"维持优势、轻度支架、继续观察"为主，不要硬找严重问题。

【参考理论框架（请在建议中合理引用）】
- 本研究 Logit 回归结果：好友同组（熟人效应）对任务型角色有全面负效应，干扰者相对概率升高；性别是最强预测因子（男生更多干扰者，女生更多任务型角色）；高成就学生争吵促进者亦显著升高
- Webb, N.M. (1989)：异质组中高成就学生给予解释有助于低成就学生理解
- Johnson & Johnson (1994)：积极互赖与个人责任是合作学习的核心结构条件
- Cohen, E.G. (1994)：地位处理技术可改善地位差异对参与机会的压制
- Lou et al. (1996)：低成就同质组效果最差，异质组学习收益最高
- Gillies (2003)：提供结构化对话支架对低能力学生参与有显著促进作用

请根据以上真实模拟数据，生成一份教学建议报告，包含：

1. **模拟诊断**（2-3句）：结合本次角色分布数据，描述这个小组在本次模拟中呈现出的核心合作特征和值得关注的问题。要具体，说明哪个角色多、哪个少、意味着什么。

2. **干预建议**（恰好3条）：每条建议针对本次模拟暴露出的具体问题，给出可操作的教学策略。每条格式如下：
### 建议标题
建议内容（2-3句，具体可操作）
📚 文献依据：引用名（年份）简短说明

3. **分组反思**（1-2句）：基于本次数据，对这种分组方式提出一个教师可以思考的问题或改进方向。

输出要求：
- 语言面向一线小学教师，避免过度学术化
- 建议要结合本次具体数据（如"本次模拟中干扰者占X%"），不要泛泛而谈
- 干预建议标题只使用三级标题"###"，不要使用"####"
- 直接输出正文，不要加"好的""以下是"等前缀"""

    llm_text = ''
    try:
        from mars_simulation.generation import _get_next_client
        resp = _get_next_client().chat.completions.create(
            model='deepseek-chat',
            messages=[
                {'role': 'system', 'content': '你是一位合作学习领域的专业研究者，为教师提供基于数据的个性化建议。'},
                {'role': 'user',   'content': prompt},
            ],
            temperature=0.6,
            max_tokens=1200,
        )
        llm_text = resp.choices[0].message.content.strip()
    except Exception as e:
        llm_text = f'（建议生成失败：{e}）'

    return jsonify({
        'group_label': ginfo['label'],
        'group_type':  gtype,
        'sim_stats':   sim_stats,
        'llm_advice':  llm_text,
    })


# 每个 (group_id, sim_type) 对应一个活跃的 cancel_event
# 新请求进来时立即 set 旧的，避免"僵尸线程"占用 API 连接
_active_cancels: dict = {}
_active_cancels_lock  = threading.Lock()


def _simulate_response(group_id: int, ctrl_factory, record_history: bool = True,
                       sim_type: str = 'main'):
    if group_id not in GROUPS:
        return Response('unknown group', status=404)

    task_cfg     = _parse_task_config(request)
    ginfo        = GROUPS[group_id]
    actual_turns = TEST_TURNS if TEST_TURNS else ginfo['base_turns']
    member_names = [m['name'] for m in ginfo['members']]

    q: queue.Queue = queue.Queue()
    cancel_event   = threading.Event()

    # 取消同组同类型的旧模拟（如果有）
    sim_key = f"{group_id}_{sim_type}"
    with _active_cancels_lock:
        old = _active_cancels.get(sim_key)
        if old:
            old.set()
        _active_cancels[sim_key] = cancel_event

    run_id = uuid.uuid4().hex[:8]

    def run():
        try:
            ctrl = ctrl_factory()
            with LineCapture(q):
                ctrl.run_simulation(
                    group_agent_names=member_names,
                    dialogue_turns=actual_turns,
                    output_file_path=os.path.join(BASE_DIR, f'sim_{group_id}_{run_id}_tmp.txt'),
                    task_name=task_cfg['task_name'],
                    should_stop=cancel_event.is_set,
                )
        except Exception as exc:
            traceback.print_exc()
            q.put({'type': 'error', 'msg': f'模拟运行失败：{exc}'})
        else:
            q.put({'type': 'done'})

    threading.Thread(target=run, daemon=True).start()

    def generate():
        history_log = []
        turns_used  = 0
        try:
            yield sse({'type': 'init', 'members': ginfo['members'],
                       'turns': actual_turns, 'label': ginfo['label'],
                       'task_name': task_cfg['task_name']})

            while True:
                try:
                    line = q.get(timeout=10)
                except queue.Empty:
                    yield ': ping\n\n'
                    continue

                if isinstance(line, dict):
                    if line.get('type') == 'done':
                        if record_history:
                            fname = save_history(group_id, turns_used, history_log)
                            yield sse({'type': 'done', 'filename': fname})
                        else:
                            yield sse({'type': 'done'})
                        break
                    if line.get('type') == 'error':
                        yield sse(line)
                        break
                    continue

                if line is None:
                    if record_history:
                        fname = save_history(group_id, turns_used, history_log)
                        yield sse({'type': 'done', 'filename': fname})
                    else:
                        yield sse({'type': 'done'})
                    break

                m = TEACHER_RE.match(line)
                if m:
                    evt = {'type': 'teacher', 'text': m.group(1)}
                    history_log.append(evt)
                    yield sse(evt)
                    continue

                m = TURN_RE.match(line)
                if m:
                    n, total, speaker, role, text = m.groups()
                    turns_used = int(n)
                    evt = {
                        'type':    'turn',
                        'turn':    int(n),
                        'total':   int(total),
                        'speaker': speaker,
                        'display': display_name(speaker),
                        'role':    role,
                        'color':   ROLE_COLORS.get(role, '#888'),
                        'text':    text,
                    }
                    history_log.append(evt)
                    yield sse(evt)
                    continue

                m = CANVAS_RE.match(line)
                if m:
                    n, total, raw_items = m.groups()
                    try:
                        items = json.loads(raw_items)
                    except Exception:
                        items = [raw_items]
                    evt = {'type': 'canvas', 'turn': int(n), 'total': int(total), 'items': items}
                    history_log.append(evt)
                    yield sse(evt)
                    continue

                if CHECK_RE.match(line):
                    continue

                if line.strip() and not HIDDEN_STATUS_RE.match(line.strip()):
                    yield sse({'type': 'status', 'msg': line.strip()})
        finally:
            cancel_event.set()
            # 清理活跃记录（只清理自己的，避免覆盖新请求的）
            with _active_cancels_lock:
                if _active_cancels.get(sim_key) is cancel_event:
                    del _active_cancels[sim_key]

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/simulate/<int:group_id>')
def api_simulate(group_id: int):
    task_cfg = _parse_task_config(request)

    def ctrl_factory():
        from mars_simulation.controller import MarsController
        return MarsController(
            matrix_path=None, agents_data=AGENTS_DF, initial_dist_path=None,
            role_bias=GROUP_DEFS[group_id].get('role_bias'),
            stage_defs=task_cfg['stages'],
            transition_matrix_df=TRANSITION_MATRIX_DF,
            initial_dist_series=INITIAL_DIST_SERIES,
        )
    return _simulate_response(group_id, ctrl_factory, record_history=True, sim_type='main')


@app.route('/api/simulate/basic/<int:group_id>')
def api_simulate_basic(group_id: int):
    task_cfg = _parse_task_config(request)

    def ctrl_factory():
        from mars_simulation.controller_basic import BasicMarsController
        return BasicMarsController(
            matrix_path=None, agents_data=AGENTS_DF, initial_dist_path=None,
            role_bias=GROUP_DEFS[group_id].get('role_bias'),
            stage_defs=task_cfg['stages'],
            transition_matrix_df=TRANSITION_MATRIX_DF,
            initial_dist_series=INITIAL_DIST_SERIES,
        )
    return _simulate_response(group_id, ctrl_factory, record_history=False, sim_type='basic')


@app.route('/api/history')
def api_history():
    files  = sorted(glob.glob(os.path.join(HISTORY_DIR, '*.json')), reverse=True)
    result = []
    for f in files[:60]:
        try:
            with open(f, encoding='utf-8') as fp:
                d = json.load(fp)
            result.append({
                'filename':    os.path.basename(f),
                'group_label': d.get('group_label', ''),
                'timestamp':   d.get('timestamp', ''),
                'turns_used':  d.get('turns_used', 0),
                'members':     [m['display'] for m in d.get('members', [])],
            })
        except Exception:
            pass
    return jsonify(result)


@app.route('/api/history/<path:filename>')
def api_history_detail(filename):
    filename = os.path.basename(filename)
    path     = os.path.join(HISTORY_DIR, filename)
    if not os.path.exists(path):
        return Response('not found', status=404)
    with open(path, encoding='utf-8') as f:
        return jsonify(json.load(f))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5020'))
    app.run(host='0.0.0.0', debug=False, port=port, threaded=True, use_reloader=False)
