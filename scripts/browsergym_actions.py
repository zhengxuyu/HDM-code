"""每个 browsergym 动作:不改代码就能当候选项选的,和必须产生内容的。

**问题是「不做任何改造」。** 那把动作分成三类而不是两类,因为「参数取值在概念上
可枚举」和「现有代码不改就能把它枚举出来」不是一回事:

    selectable_now      候选项已经在构造,快思考现在就能选
    selectable_in_principle
                        取值能从现有 observation 字段得到,但没有构造代码。
                        不需要模型产生任何 token,需要的是我们写枚举
    needs_content       至少一个参数的取值无法枚举,只能由模型生成

第二类是这份文件存在的理由。`docs/datasets.md` 1.5 节按「参数能不能枚举」分了两类,
读起来像是 `select_option` 和 `press` 现在就能用 —— 不能,它们的候选项没人构造。
把那一步说清,是因为「给快思考加动作」这个决定要知道代价是「写枚举」还是「换模式」。

每条记录都带 `candidates_from`:取值从哪个 observation 字段来。没有来源的就是
`needs_content`,这是这份划分唯一的判据。

    python scripts/browsergym_actions.py -o src/slime_htm/envs/browsergym-actions.json
"""

import argparse
import json
from pathlib import Path

# browsergym 0.14.3 的 observation 字段 **【实测,core/env.py:670-688】**,
# 枚举候选项只能从这些里取。
OBSERVATION_FIELDS = (
    "chat_messages", "goal", "goal_object", "open_pages_urls", "open_pages_titles",
    "active_page_index", "url", "screenshot", "dom_object", "axtree_object",
    "extra_element_properties", "focused_element_bid", "last_action",
    "last_action_error", "elapsed_time",
)

# 候选槽上限:快思考从单 token 码里读答案,这个 tokenizer 给 52 个。
# **任何公式算出来超过 52 的,接进去就要截断,而截断就是丢候选项。**
SLOTS = 52

# 实测的页面规模,代入公式用 **【28853 的 corpus,317 步,MiniWoB click 组】**:
# 候选数(含 scroll)中位 3、均值 4.8、最大 15,顶到 52 的步数 **0**。
# 减去 scroll 那一项,可点击元素数 E 中位 2、最大 14。
MEASURED = {
    "venue": "miniwob click",
    "steps": 317,
    "candidates_median": 3,
    "candidates_max": 15,
    "candidates_at_limit": 0,
    "elements_median": 2,
    "elements_max": 14,
    # WebArena 是另一个量级,数字来自 candidates() 自己的 docstring:一篇英文
    # Wikipedia 文章 3476 个 clickable,232 个渲染出来,视口内 10 个正文链接。
    "webarena_note": "一篇 Wikipedia 文章 3476 clickable / 232 rendered / 视口内 10",
}

# 候选项个数的公式。E = 视口内可点击元素数,K = 键名集合大小,
# S = <select> 个数,O = 每个的选项数,T = 开着的标签数。
FORMULA = {
    "click": "E", "dblclick": "E", "hover": "E", "focus": "E", "clear": "E",
    "check": "E", "uncheck": "E",
    "go_back": "1", "go_forward": "1", "new_tab": "1", "tab_close": "1",
    "noop": "1(固定一个等待时长)",
    "tab_focus": "T",
    "scroll": "2(两个方向)",
    "press": "E x K",
    "keyboard_press": "K", "keyboard_up": "K", "keyboard_down": "K",
    "select_option": "S x O",
    "drag_and_drop": "E^2",
    "mouse_click": "E", "mouse_dblclick": "E", "mouse_down": "E",
    "mouse_up": "E", "mouse_move": "E",
    "mouse_drag_and_drop": "E^2",
    "scroll_at": "E x 2",
}

# 构造的代价,这是「能不能构造出来」的真答案:三种,不是一种。
BUILD_COST = {
    "same_loop": "和 click 同一个循环,遍历 extra_element_properties 即可。几行",
    "constant": "常量,不看页面",
    "needs_a_subset": "**能构造,但必须先裁取值集合。** known_key 接受 43 个具名键 "
                      "加任意单字符(~138 个基础键),再乘修饰键组合约 828 种 —— "
                      "完整集合一个元素就塞满 52 槽。要先定一个小键表(Enter/Tab/"
                      "方向键之类),而定它就是一个需要预注册的选择",
    "needs_hierarchy": "**要先改读取方式。** 取值在 axtree_object 里,但把 option "
                       "配到它的 <select> 需要父子关系,现在 element_names() 读的是"
                       "拍平后的字符串",
    "quadratic": "**能构造但是平方级。** 52 槽只够 7 个元素(7x6=42),第 8 个就溢出",
}

COST_OF = {
    **{a: "same_loop" for a in ("click", "dblclick", "hover", "focus", "clear",
                                "check", "uncheck", "mouse_click",
                                "mouse_dblclick", "mouse_down", "mouse_up",
                                "mouse_move", "scroll_at")},
    **{a: "constant" for a in ("go_back", "go_forward", "new_tab", "tab_close",
                               "noop", "scroll", "tab_focus")},
    **{a: "needs_a_subset" for a in ("press", "keyboard_press", "keyboard_up",
                                     "keyboard_down")},
    "select_option": "needs_hierarchy",
    "drag_and_drop": "quadratic",
    "mouse_drag_and_drop": "quadratic",
}


# 每个动作:(元数, 分类, 取值来源, 说明)。
#
# 分类判据只有一条:每个参数的取值,能不能从 OBSERVATION_FIELDS 里列举出来。
# 「现在能不能」另记在 WIRED_UP 里,两件事分开,因为混在一起正是 1.5 节读起来
# 像「select_option 现在就能用」的原因。
ACTIONS = {
    # --- 只要一个 bid,或者完全不要参数 ---
    "click": (1, "selectable", "extra_element_properties 的键",
              "每个元素一个候选项。快思考现在唯一在用的那个"),
    "dblclick": (1, "selectable", "extra_element_properties 的键", "同 click"),
    "hover": (1, "selectable", "extra_element_properties 的键", "同 click"),
    "focus": (1, "selectable", "extra_element_properties 的键", "同 click"),
    "clear": (1, "selectable", "extra_element_properties 的键",
              "清空一个输入框。**不需要内容** —— 它只删不写"),
    "check": (1, "selectable", "extra_element_properties 的键",
              "任何 venue 预设都给不到:不在任何子集里(见 datasets.md 1.5)"),
    "uncheck": (1, "selectable", "extra_element_properties 的键", "同 check,拿不到"),
    "go_back": (0, "selectable", "无参", "一个候选项"),
    "go_forward": (0, "selectable", "无参", "一个候选项"),
    "new_tab": (0, "selectable", "无参", "一个候选项"),
    "tab_close": (0, "selectable", "无参", "一个候选项"),
    "noop": (1, "selectable", "少数常量",
             "wait_ms 有默认值,固定取几个即可。也不在任何子集里"),

    # --- 取值是有限集合或页面自带的列表 ---
    "tab_focus": (1, "selectable", "open_pages_urls 的长度",
                  "index 的上界就是开着的标签数,observation 直接给"),
    "press": (2, "selectable", "extra_element_properties 的键 × 有限键名集合",
              "键名是有限的,而且这个集合我们自己就维护着:browser.py 的 _KEY_NAMES"),
    "scroll": (2, "selectable", "少数常量",
               "现在就是快思考的另一半:CONTENT_FREE 两个方向"),
    "select_option": (2, "selectable", "axtree_object 里 role=option 的节点",
                      "**有一个坑。** <option> 标签会被标 bid(standard_html 标 125 "
                      "个标准标签,option 在内),所以选项在无障碍树里有自己的节点和文本。"
                      "但要把它和它所属的 <select> 配对需要树的父子关系,而现在的 "
                      "element_names() 读的是拍平后的字符串,没有层级。所以取值在 "
                      "observation 里,构造代码不在"),
    "drag_and_drop": (2, "selectable", "extra_element_properties 的键 × 自身",
                      "两个 bid 的叉乘。可枚举但是平方级,52 个候选槽放不下"),

    # --- 坐标式:取值能从 bbox 推出来 ---
    "mouse_click": (3, "selectable", "extra_element_properties 的 bbox",
                    "坐标可以取元素中心。browsergym 自己的 coord 子集就是这么用的"),
    "mouse_dblclick": (3, "selectable", "extra_element_properties 的 bbox", "同上"),
    "mouse_down": (3, "selectable", "extra_element_properties 的 bbox", "同上"),
    "mouse_up": (3, "selectable", "extra_element_properties 的 bbox", "同上"),
    "mouse_move": (2, "selectable", "extra_element_properties 的 bbox", "同上"),
    "mouse_drag_and_drop": (4, "selectable", "bbox × bbox", "同 drag_and_drop,平方级"),
    "scroll_at": (4, "selectable", "少数常量 × bbox", "位置加方向,都可枚举"),
    "keyboard_press": (1, "selectable", "有限键名集合", "不带 bid 的 press"),
    "keyboard_up": (1, "selectable", "有限键名集合", "同上"),
    "keyboard_down": (1, "selectable", "有限键名集合", "同上"),

    # --- 取值无法枚举:必须由模型产生 ---
    "fill": (2, "needs_content", None,
             "**这是那条线本身。** value 是自由文本,页面上没有一个字段列出「应该填"
             "什么」。option (b) 曾提议用 goal 的子串当候选,那是一个近似而不是枚举:"
             "答案得从页面读出来的任务照样到不了"),
    "send_msg_to_user": (1, "needs_content", None,
                         "答案本身。WebArena 的 string_match 任务全靠它,而答案要从"
                         "页面读,不在任何可枚举集合里"),
    "keyboard_type": (1, "needs_content", None, "不带 bid 的 fill,同样是自由文本"),
    "keyboard_insert_text": (1, "needs_content", None, "同上"),
    "report_infeasible": (1, "needs_content", None,
                          "reason 是自由文本。**但它有一个可枚举的退化版**:固定一句"
                          "理由就能当一个候选项,因为评分看的是「有没有报不可行」而"
                          "不是理由写了什么。现在没有接"),
    "goto": (1, "needs_content", "dom_object 里的 href(部分)",
             "**部分可枚举。** 页面上的链接能从 dom_object 取,那是一个有限集合;"
             "但任意 URL 不行。按「不改造」算它需要内容:现在没有任何代码从 "
             "dom_object 取 href"),
    "upload_file": (2, "needs_content", None,
                    "file 是本地路径。严格说不是模型「产生内容」,但也枚举不出来 —— "
                    "要环境预先放文件并告知路径,现在没有这个机制"),
    "mouse_upload_file": (3, "needs_content", None, "同 upload_file"),
}

# 现在真的在构造候选项的 —— 不是「原则上能」,是代码里有。
# **【实测,src/slime_htm/envs/browser.py】** candidates() 遍历
# extra_element_properties 产出 click,content_free_actions() 产出两个 scroll。
WIRED_UP = ("click", "scroll")

# 被可见性过滤挡掉的:有 bid、在无障碍树里,但 candidates() 的
# 「rendered + 在视口内」会滤掉它们,所以即使枚举了也选不到。
FILTERED_OUT = {
    "option": "折叠的 <select> 里的 <option> 有 bid,但 bbox 为空或不在视口,"
              "所以 candidates() 的可见性过滤会去掉它们。这就是为什么「直接 click "
              "一个 option」替代不了 select_option",
}


def _at_scale(name):
    """公式代入实测规模后的候选项个数,和它放不放得下 52 槽。"""
    E, K, T = MEASURED["elements_median"], 138, 2
    biggest = MEASURED["elements_max"]
    counts = {
        "E": (E, biggest),
        "E x K": (E * K, biggest * K),
        "E^2": (E * E, biggest * biggest),
        "E x 2": (E * 2, biggest * 2),
        "K": (K, K),
        "T": (T, T),
    }
    formula = FORMULA.get(name, "")
    if formula.startswith("1") or formula.startswith("2("):
        return {"median": 1 if formula.startswith("1") else 2,
                "worst": 1 if formula.startswith("1") else 2, "fits": True}
    if formula == "S x O":
        return {"median": None, "worst": None, "fits": None,
                "why": "S 和 O 要从页面数,现在读不到层级,所以没有实测值"}
    median, worst = counts.get(formula, (None, None))
    if median is None:
        return {"median": None, "worst": None, "fits": None}
    return {"median": median, "worst": worst, "fits": worst <= SLOTS,
            "slots": SLOTS}


def shared_budget(rows):
    """**52 个槽是所有动作共享的**,所以单看一个动作放不放得下没有意义。

    这是「会不会有超大可选集」的真答案:会,而且不用等到 `press`。把所有和 click
    同循环的动作加进去,在实测的最坏页面上就已经超了四倍 —— 而那是 MiniWoB,
    已知最小的 venue。
    """
    groups = [
        ({"constant"}, "只加常量动作"),
        ({"constant", "same_loop"}, "常量 + 和 click 同循环的"),
        ({"constant", "same_loop", "quadratic"}, "再加平方级的"),
        ({"constant", "same_loop", "quadratic", "needs_a_subset"},
         "全加(含完整键集合)"),
    ]
    found = []
    for kinds, label in groups:
        median = worst = 0
        picked = []
        for row in rows:
            if row.get("build_cost") not in kinds:
                continue
            scale = row["at_measured_scale"]
            if scale["median"] is None:
                continue
            median += scale["median"]
            worst += scale["worst"]
            picked.append(row["action"])
        found.append({"label": label, "actions": sorted(picked),
                      "median_total": median, "worst_total": worst,
                      "fits": worst <= SLOTS, "over_by": max(0, worst - SLOTS)})
    return found


def build(signatures=None):
    """三类清单。`signatures` 是上游的真实元数,用来交叉验证这份表。"""
    rows = {"selectable_now": [], "selectable_in_principle": [], "needs_content": []}
    mismatched = []
    for name, (arity, kind, source, note) in sorted(ACTIONS.items()):
        if signatures is not None and name in signatures:
            # 上游签名含默认值参数,所以比较的是「必填参数个数 <= 我们记的元数」。
            required = signatures[name]["required"]
            if required > arity:
                mismatched.append(
                    {"action": name, "ours": arity, "upstream_required": required})
        row = {"action": name, "arity": arity, "candidates_from": source,
               "note": note}
        if kind != "needs_content":
            row["candidates_formula"] = FORMULA.get(name)
            row["build_cost"] = COST_OF.get(name)
            row["at_measured_scale"] = _at_scale(name)
        if kind == "needs_content":
            rows["needs_content"].append(row)
        elif name in WIRED_UP:
            rows["selectable_now"].append(row)
        else:
            rows["selectable_in_principle"].append(row)
    return rows, mismatched


def upstream_signatures():
    """每个动作的参数个数,从装着的 browsergym 读。缺包时返回 None。"""
    try:
        import ast
        import inspect

        from browsergym.core.action import functions
    except ImportError:
        return None
    found = {}
    for node in ast.parse(inspect.getsource(functions)).body:
        import ast as _ast

        if isinstance(node, _ast.FunctionDef) and not node.name.startswith("_"):
            required = len(node.args.args) - len(node.args.defaults)
            found[node.name] = {"parameters": len(node.args.args),
                                "required": required}
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    signatures = upstream_signatures()
    rows, mismatched = build(signatures)

    found = {
        "browsergym": "0.14.3",
        "definition": {
            "selectable_now": "候选项已经在构造,快思考现在就能选,不产生任何 token",
            "selectable_in_principle": "取值能从现有 observation 字段得到,但没有"
                                       "构造代码。缺的是我们写枚举,不是模型产生内容",
            "needs_content": "至少一个参数的取值无法从 observation 枚举,只能由模型生成",
        },
        "observation_fields": list(OBSERVATION_FIELDS),
        "slots": SLOTS,
        "measured_scale": MEASURED,
        "build_cost_kinds": BUILD_COST,
        "wired_up": list(WIRED_UP),
        "filtered_out": FILTERED_OUT,
        "counts": {name: len(value) for name, value in rows.items()},
        "shared_budget": shared_budget(
            rows["selectable_now"] + rows["selectable_in_principle"]),
        **rows,
        "upstream_signatures": signatures,
        "arity_mismatches": mismatched,
    }
    if signatures is None:
        found["upstream_note"] = (
            "browsergym 没装,所以元数没有和上游交叉验证。装了之后重跑这个脚本,"
            "arity_mismatches 会列出对不上的动作")
    Path(args.output).write_text(json.dumps(found, indent=1, ensure_ascii=False,
                                            sort_keys=True) + "\n")

    for name, value in rows.items():
        print(f"{name:26} {len(value):>2}  "
              f"{' '.join(r['action'] for r in value)}")
    print()
    for row in found["shared_budget"]:
        verdict = ("放得下" if row["fits"]
                   else f"**超 {row['over_by']} 个**")
        print(f"{row['label']:34} {len(row['actions']):>2} 个   "
              f"中位 {row['median_total']:>4}   最坏 {row['worst_total']:>5}   "
              f"{verdict}")
    if mismatched:
        print(f"\n**元数和上游对不上**: {mismatched}")
    elif signatures:
        print(f"\n元数与上游 {len(signatures)} 个动作签名一致")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
