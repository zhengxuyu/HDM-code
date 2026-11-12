# 数据集与任务空间分析

**写作日期** 2026-10-07。**写作者** exps(提交实验、监控、回报的那一侧)。

## 怎么读这份文件

每个数字都标了来源,因为这个项目里被转述的数字出过错,而且两个方向都出过。三类标记:

- **【实测】** 我自己在包、任务文件或已入库的结果文件上数出来的,命令可重跑。
- **【入库】** 来自 `experiments/*/results/*.json`,可从文件重算。
- **【转述】** 同行(paper-writer / code-reviewer)给的,我**没有**独立验证。决策若取决于它,先验证再用。

一条贯穿全文的区分:**"benchmark 有多少任务"和"我们的动作空间能碰到多少任务"是两件事。** 后者才决定一个 gate 能不能产出判定,而这份文件的主要内容就是把后者算出来。

---

## 1. browsergym:包、venue、动作子集

### 1.1 装了什么

```
site-packages/browsergym/:   core   miniwob   utils
browsergym-core      0.14.3
browsergym-miniwob   0.14.3
```
**【实测】** 上游还分发 `browsergym-webarena`、`-visualwebarena`、`-workarena`、`-assistantbench`、`-weblinx`,**本环境一个都没装**;`libwebarena` 也没装。

gym registry 里 **125 个** `browsergym/miniwob.*` 任务,外加 1 个 `browsergym/openended`。**【实测】**

我们自己的 `src/slime_htm/envs/gate.py:VENUES` 声明三个 venue:`miniwob` / `webarena` / `workarena`。其中只有 `miniwob` 在本环境可跑;另两个的 `packages` 字段列的依赖缺失,`--check-env` 会在排队之前拒绝。

### 1.2 先说清 `bid`,后面每个动作都带着它

**`bid` = browsergym id**,是 browsergym 自己往每个 DOM 元素上写的一个 HTML 属性,
属性名就叫 `bid` **【实测,`browsergym/core/constants.py:1`】**:

```python
BROWSERGYM_ID_ATTRIBUTE = "bid"  # Playwright's default is "data-testid"
```

那句注释点出了用途:它充当 Playwright 的 test-id 定位符。所以 `click('a48')` 的意思
是「点 `bid="a48"` 的那个元素」,browsergym 按这个属性去定位它。

**怎么生成的** **【实测,`core/javascript/frame_mark_elements.js:122-138`,每次构造
observation 时跑一遍】**:

- 普通元素拿**数字**,来自挂在 `window` 上的递增计数器:`0` `1` `2` …
- `iframe` / `frame` 拿**字母**:`a` `b` … `z` `aA` `aB` …
- 嵌套时拼接成 `${父bid}${本地序号}`

所以 `a48` 读作「**a 号 iframe 里的第 48 个元素**」。MiniWoB 的任务页面装在 iframe
里,这就是我们看到的 bid 普遍带字母前缀、而不是纯数字的原因。

**它怎么进到模型的输入里**:同一段 JS 把 `browsergym_id_<bid>` 推进元素的
`aria-roledescription` 和 `aria-description`,于是 bid 出现在无障碍树里,
`flatten_axtree_to_str`(我们的 `state_text` 调它)一拍平,模型读到的页面文本就带着
它。没有这一步,模型写不出一个合法的 `click('a48')`。

**一个有后果的性质:bid 不是稳定标识符。** 计数器挂在 `window` 上,所以同一个页面文档
内分配后写进 DOM 并保持,但**导航到新页面就是新 window、计数器归零** —— 同一个元素会
拿到不同编号,两个不同页面上的 `bid="5"` 是毫无关系的两个元素。这正是
`parse_action(allowed_bids=...)` 只用**当前这一步**页面给出的集合去校验的原因:跨步骤
复用一个记下来的编号不安全,而慢思考是自由书写的,完全可能写出上一步看到的那个。

> **命名撞车,先说破。** 下一节表格里的 `bid` 是**子集名**(按元素 id 操作的那 11 个
> 动作),和这里的**属性名**同名而不是一回事。子集之所以叫这个名字,是因为它里面每个动
> 作的第一个参数都是一个 bid。

---

### 1.3 动作空间不是"每 venue 一个",而是 16 个可组合子集

`browsergym.core.action.highlevel.ACTION_SUBSETS` 全部内容 **【实测】**:

| 子集 | 个数 | 动作 |
|---|---|---|
| `bid` | 11 | `clear click dblclick drag_and_drop fill focus hover press scroll select_option upload_file` |
| `coord` | 13 | `keyboard_down keyboard_insert_text keyboard_press keyboard_type keyboard_up mouse_click mouse_dblclick mouse_down mouse_drag_and_drop mouse_move mouse_up mouse_upload_file scroll_at` |
| `chat` | 1 | `send_msg_to_user` |
| `nav` | 3 | `go_back go_forward goto` |
| `tab` | 3 | `new_tab tab_close tab_focus` |
| `infeas` | 1 | `report_infeasible` |
| `webarena` | 14 | `click fill hover scroll select_option send_msg_to_user keyboard_press go_back go_forward goto new_tab tab_close tab_focus report_infeasible` |
| `visualwebarena` | 15 | = `webarena` + `upload_file` |
| `workarena` | 11 | `clear click dblclick drag_and_drop fill focus hover press scroll select_option send_msg_to_user` |
| `workarena++` | 18 | = `workarena` + `nav` + `tab` + `report_infeasible` |
| `assistantbench` | 8 | `click fill go_back goto press scroll select_option send_msg_to_user` |
| `weblinx` | 9 | `click fill goto hover new_tab scroll send_msg_to_user tab_close tab_focus` |
| `miniwob_all` | 10 | `click fill keyboard_press keyboard_type mouse_click mouse_dblclick mouse_down mouse_move mouse_up scroll` |
| `miniwob_liu18` | 2 | `click fill` |
| `miniwob_shi17` | 6 | `keyboard_press mouse_click mouse_dblclick mouse_down mouse_up scroll` |
| `miniwob_humphreys22` | 8 | `keyboard_press keyboard_type mouse_click mouse_dblclick mouse_down mouse_move mouse_up scroll` |

真正的分野是 **`bid` 对 `coord`**:按元素 id 操作,还是按屏幕坐标操作。MiniWoB 原始论文里 shi17 与 humphreys22 是坐标式,liu18 是 id 式但只有两个动作。

**除 `miniwob_shi17` 和 `miniwob_humphreys22` 外,每个 venue 预设都含 `fill`。**
那两个是坐标式的,整套没有打字动作(`keyboard_type` 在 `miniwob_humphreys22` 里,
`miniwob_shi17` 连它都没有)。原文写的是「每一个预设都含 `fill`」,对着上面那张表就能
看出不成立 —— 这一条是后面 WebArena 结论的根,所以它错不起。

### 1.4 我们实际用的动作空间

**三层,数字不一样,之前这一节只写了其中一层而且写错了。**

这一节原来写「`bid`(11) + `chat`(1) + `noop`(1) = 13」。那是**推的,不是量的**:
`gym.make` 不传 `action_mapping`,所以环境构造的是 `HighLevelActionSet` 的默认值。
对 `browsergym/miniwob.click-test` 实读 `env.unwrapped.action_mapping.__self__.action_set`:

```
环境给的       20 个
  chat    1   send_msg_to_user
  infeas  1   report_infeasible
  bid    11   clear click dblclick drag_and_drop fill focus hover press scroll
              select_option upload_file
  nav     3   go_back go_forward goto
  tab     3   new_tab tab_close tab_focus
  (总是允许) noop
```

我们并不用这 20 个。慢思考受 `ACTION_ARITY` 白名单限制,快思考受候选集构造限制:

| 层 | 个数 | 哪些 |
|---|---:|---|
| 环境给的 | **20** | 上表 |
| 慢思考能写(`ACTION_ARITY`) | **9** | `click fill select_option send_msg_to_user scroll goto go_back hover press` |
| 快思考能选 | **2** | `click` `scroll` |

**快思考够不到 18 个,但那 18 个分成性质完全不同的两堆**(按 1.6 那份量过的分类):

```
selectable_in_principle  13   clear dblclick drag_and_drop focus go_back go_forward
                              hover new_tab noop press select_option tab_close tab_focus
needs_content             5   fill goto report_infeasible send_msg_to_user upload_file
```

中间那 13 个**不需要改变快思考是什么** —— 仍然零生成、仍然是一个索引,缺的只是构造代码。
后面 5 个必须生成内容,接上去快思考就不是快思考了。任何"扩大快思考"的提案,第一个要回答
的就是它动的是哪一堆。

白名单(9 个)是故意的:慢思考是个写代码样文本的语言模型,一个我们没打算支持的动作是
一个事后无法解释的步骤。

**但快思考实到的不是"无内容的 8 个"。** `content_free_actions()` 只返回滚动:

```python
CONTENT_FREE = (f"scroll(0, {SCROLL_STEP})", f"scroll(0, -{SCROLL_STEP})")
```

所以**快思考的候选集 = 每个在视口内的页面元素一个 `click('<bid>')`(上限 52 个槽)+ 向下滚动,滚过之后再加向上滚动**。就是 **click + scroll**。

`drag_and_drop` 被明确排除:它要两个元素 id,是叉乘而不是常量。`text_actions()` 实现在 `browser.py:351`,**没有接上任何运行路径**,并有测试断言 gate 不调用它。

### 1.5 哪些动作需要"产生内容":按参数来源,不按参数个数

**【实测,`browsergym-core==0.14.3` 的 `core/action/functions.py` 与 `highlevel.py`】**
全量 **35** 个动作函数,其中 **32** 个能通过某个子集拿到。`check`、`uncheck`、
`noop` **不在任何 `ACTION_SUBSETS` 子集里,但永远可用**:`highlevel.py:310` 是
`allowed_actions = [noop]  # the noop action is always allowed`,构造函数从这个列表起步
再加子集。原文写成「任何 venue 预设都给不到」,把"不在子集里"当成了"拿不到",而这两件
事在这个构造函数里恰好相反。`check` / `uncheck` 才是真的任何预设都拿不到。
所以勾选框任务在任何 venue 预设下都只能用
`click` 去点,没有语义化的勾选动作可用。

1.4 节那张"需要内容 5 / 无内容 8"的表,分的是**参数是不是只有 bid**。那和"这一步
需不需要模型产生内容"**不是同一条线**,而后者才是两段模式的分界。按参数的来源重分:

| 参数来源 | 枚举得出来吗 | 动作 |
|---|---|---|
| 无参,或只有 bid | **能**,页面给出全部 bid | `click` `dblclick` `hover` `focus` `clear` `check` `uncheck` `go_back` `go_forward` `new_tab` `tab_close` |
| bid × bid | 能,但是叉乘 | `drag_and_drop` |
| 屏幕坐标 | 能,从元素 bbox 推 | `mouse_click` `mouse_dblclick` `mouse_down` `mouse_up` `mouse_move` `mouse_drag_and_drop` `mouse_upload_file` `scroll_at` |
| 少数数值常量 | 能 | `scroll(dx, dy)` `tab_focus(index)` `noop(wait_ms)` |
| 页面上的 `<option>` 值/标签 | **能**,`<select>` 自带选项表 | `select_option` |
| Playwright 的键名,有限集合 | **能**,我们代码里就有 `_KEY_NAMES` | `press` `keyboard_press` `keyboard_up` `keyboard_down` |
| URL | 页面链接能,任意 URL 不能 | `goto` |
| 本地文件路径 | 不能,但那是环境提供的,不是模型写的内容 | `upload_file` `mouse_upload_file` |
| **自由文本** | **不能** | **`fill`** `send_msg_to_user` `keyboard_type` `keyboard_insert_text` `report_infeasible` |

**所以真正"必须由模型产生内容"的只有 5 个,而且不是 1.4 节列的那 5 个:**

```
1.4 节(按参数个数)   fill  select_option  send_msg_to_user  press  upload_file
按可枚举性           fill  send_msg_to_user  keyboard_type  keyboard_insert_text  report_infeasible
```

两个划分都含 `fill` 和 `send_msg_to_user`;分歧在 `select_option`、`press`、
`upload_file`。

**这条分歧有后果,不只是叫法。** `select_option` 的取值是页面 `<select>` 自己列出
的选项,`press` 的取值是 Playwright 有限的键名 —— 两者都能像 `click('<bid>')` 一样
**逐个枚举成候选项**,快思考读 logit 就能选,不需要产生任何文本。也就是说这两个动作
可以进快思考的候选集而**不越过**"这一步要不要产生内容"那条线。目前它们没有进:
`content_free_actions()` 只返回滚动(见 1.4)。

**我没有重算加上它们之后 WebArena 的可达任务数**,所以不声称 3.3/3.4 节那些数字会变。
那些任务卡住的地方主要是 `fill`(搜索框、表单、发帖正文),加 `select_option` 和
`press` 未必能动多少 —— 但"未必"不是一个计数,要就得重算
(`scripts/webarena_span_coverage.py` 是重算它的脚本)。

> **这一节推翻了我自己的话。** 我在对话里按 1.4 节那张表回答过"哪些动作需要 fill
> 内容",把 `select_option` 和 `press` 算成了需要内容的。按签名看它们不是:
> `select_option(bid, options)` 的 options 来自页面,`press(bid, key_comb)` 的
> key 来自一个有限集合 —— 而那个集合我们自己在 `browser.py` 里就维护着
> (`_KEY_NAMES`,用来拦 `press('42', 'NotAKey')`)。按参数个数分类,和按"参数能不能
> 枚举"分类,在这三个动作上给出不同答案,而只有后者对应两段模式的分界。

---

### 1.6 两份清单:不改代码就能选的,和必须产生内容的

枚举在 **`src/slime_htm/envs/browsergym-actions.json`**,由
`scripts/browsergym_actions.py` 生成。判据只有一条:**每个参数的取值,能不能从
observation 的字段里列出来**(字段全名单在那个文件的 `observation_fields` 里,
**【实测,`core/env.py:670-688`】**)。

「不做任何改造」把动作分成**三类**而不是两类,因为「概念上可枚举」和「现有代码不改
就能枚举」不是一回事:

| 类 | 个数 | 动作 |
|---|---|---|
| **`selectable_now`** 候选项已经在构造,快思考现在就能选 | **2** | `click` `scroll` |
| **`selectable_in_principle`** 取值在 observation 里,但没有构造代码 | **25** | `check` `clear` `dblclick` `drag_and_drop` `focus` `go_back` `go_forward` `hover` `keyboard_down` `keyboard_press` `keyboard_up` `mouse_click` `mouse_dblclick` `mouse_down` `mouse_drag_and_drop` `mouse_move` `mouse_up` `new_tab` `noop` `press` `scroll_at` `select_option` `tab_close` `tab_focus` `uncheck` |
| **`needs_content`** 取值无法枚举,只能由模型生成 | **8** | `fill` `goto` `keyboard_insert_text` `keyboard_type` `mouse_upload_file` `report_infeasible` `send_msg_to_user` `upload_file` |

2 + 25 + 8 = 35,正好是全部动作;脚本把元数和上游 35 个签名交叉验证,**对得上**。

**中间那一类是这份清单存在的理由。** 1.5 节按「参数能不能枚举」分了两类,读起来像是
`select_option` 和 `press` 现在就能用 —— 不能,它们的候选项没人构造。缺的是**我们写
枚举代码**,不是换模式。那 25 个里任何一个要接进快思考,代价是写构造函数,不是放弃
「不产生内容」这条线。

几条值得单独记的:

- **`select_option` 有个坑。** `<option>` 标签**会**被标 bid(默认 `standard_html`
  标 125 个标准标签,option 在内),所以选项在无障碍树里有自己的节点和文本。但把它和
  所属的 `<select>` 配对需要树的父子关系,而 `element_names()` 读的是**拍平后的字符
  串**,没有层级。取值在 observation 里,构造代码不在。
- **「直接 click 一个 option」替代不了它。** 折叠的 `<select>` 里的 option 有 bid,
  但 bbox 为空或不在视口,`candidates()` 的可见性过滤会去掉它们。记在那个文件的
  `filtered_out` 里。
- **`clear` 不需要内容** —— 它只删不写,所以它在第二类而不是第三类。
- **`check` / `uncheck` 任何 venue 预设都给不到**(见 1.5;`noop` 不同,它永远可用),所以即使写了
  枚举也用不上,除非自己拼 `HighLevelActionSet`。
- **`goto` 是部分可枚举的**:页面上的链接能从 `dom_object` 取,任意 URL 不行。按
  「不改造」算它在第三类,因为现在没有任何代码从 `dom_object` 取 href。
- **`report_infeasible` 有一个可枚举的退化版**:固定一句理由就是一个候选项,因为评分
  看的是有没有报不可行、而不是理由写了什么。现在没接。
- **`drag_and_drop` 和 `mouse_drag_and_drop` 是平方级**,可枚举但 52 个候选槽放不下。

重跑:

```bash
python scripts/browsergym_actions.py -o src/slime_htm/envs/browsergym-actions.json
```

---

### 1.7 那 25 个构造得出来吗?会不会候选集爆掉?

两个问题分开答,因为答案不一样:**绝大多数构造得出来而且很便宜,但候选集会爆,而且不用
等到最贵的那几个。** 数字都在 `browsergym-actions.json` 的 `at_measured_scale` 和
`shared_budget` 里,由同一个脚本算。

#### 构造的代价有四种,不是一种

| 代价 | 个数 | 动作 | 要写什么 |
|---|---|---|---|
| `same_loop` | 13 | `click` `dblclick` `hover` `focus` `clear` `check` `uncheck` `mouse_click` `mouse_dblclick` `mouse_down` `mouse_up` `mouse_move` `scroll_at` | 和 `click` 同一个循环,遍历 `extra_element_properties`。几行 |
| `constant` | 7 | `go_back` `go_forward` `new_tab` `tab_close` `noop` `scroll` `tab_focus` | 常量,不看页面 |
| `needs_a_subset` | 4 | `press` `keyboard_press` `keyboard_up` `keyboard_down` | **能构造,但得先裁键集合**(见下) |
| `needs_hierarchy` | 1 | `select_option` | **要先改读取方式**:取值在 `axtree_object` 里,但配对要父子关系,现在读的是拍平字符串 |
| `quadratic` | 2 | `drag_and_drop` `mouse_drag_and_drop` | 能构造但是 `E²` |

所以"构造得出来吗"的答案是:**20 个几乎零成本**(13 个复用现成循环 + 7 个常量),
4 个要先做一个需要预注册的选择,1 个要改读取层,2 个数量上不现实。

#### 会爆,而且在最小的 venue 上就会

候选槽上限 **52**(快思考从单 token 码里读答案,这个 tokenizer 给 52 个),而且
**所有动作共享这 52 个**。代入实测规模 **【28853 的 corpus,317 步,MiniWoB click
组:候选数中位 3、最大 15、顶到 52 的步数 0;扣掉 scroll,可点击元素数 E 中位 2、
最大 14】**:

| 加什么 | 动作数 | 中位 | 最坏 | 放得下? |
|---|---|---|---|---|
| 现在实际在用(`click` + `scroll`) | 2 | 4 | 16 | 放得下 |
| 只加常量动作 | 7 | 9 | 9 | 放得下 |
| 常量 + 和 `click` 同循环的 | 20 | 37 | **205** | **超 153 个** |
| 再加平方级的 | 22 | 45 | **597** | 超 545 个 |
| 全加(含完整键集合) | 26 | 735 | **2943** | 超 2891 个 |

**最要紧的一行是第三行。** 把所有"几乎零成本"的动作加进去,中位页面用掉 37/52 已经
很紧,最坏页面直接 **205,是 52 槽的 3.9 倍** —— 而这是 **MiniWoB**,已知最小的 venue。(我第一版这里写的是「超四倍」,没算就写的;205/52 = 3.94,测试把它拦下来了。)WebArena
的 `E` 大一个量级(一篇 Wikipedia 文章 3476 个 clickable、232 个渲染、视口内 10 个
正文链接),那里连 `click` 自己都会顶上限,`candidates()` 的 `dropped` 计数就是为了
记这个。

#### 三种爆法,形状不同

- **`press` 是叉乘**:`E × K`。`known_key` 接受 43 个具名键加任意单字符
  (约 138 个基础键),再乘修饰键组合约 828 种。中位页面 **276**,最坏 **1932**。
  要用它必须先定一个小键表(`Enter`/`Tab`/方向键之类),而**定哪几个键是一个需要
  预注册的选择** —— 它决定哪些任务可达,事后改就是调参。
- **`keyboard_press` / `up` / `down` 最反直觉**:它们只有一个参数、看起来最简单,
  但候选集是 `K`,**和页面无关的常量 138**。页面越小它占的比例越离谱,而 MiniWoB
  的页面正是最小的。
- **`drag_and_drop` 是平方级**:`E²`,52 槽只够 **7** 个元素(7×6=42),第 8 个溢出。
  `candidates()` 的 docstring 早就说过它"是叉乘而不是常量",这里是那句话的数字版。

#### 对决定的意义

"给快思考加动作"不是一个开关,是三个独立决定:

1. **加哪些**(13 个 `same_loop` 里挑,还是全要) —— 便宜,但共享预算会被吃掉
2. **怎么分配 52 槽**(现在是 `click` 吃满、`scroll` 占 1~2) —— 加动作就得定优先级,
   而 `dropped` 已经在记被截断的数量
3. **键表取哪几个**(若要 `press`) —— 这一条改变可达任务集,必须事后不可动

`select_option` 单独一类:它是这 25 个里**唯一**取值在 observation 里却读不出来的,
而它也是 WebArena 表单任务的关键动作。要它就得让 `elements()` 读
`axtree_object` 而不是拍平后的字符串。

---

### 1.8 慢思考的动作空间不是快思考的超集

慢思考**自由书写**动作文本,由 `browser.parse_action` 解析。白名单是 `ACTION_ARITY`,**9 个** **【实测】**:

```
click(1)  fill(2)  select_option(2)  send_msg_to_user(1)  scroll(2)
goto(1)   go_back(0)  hover(1)  press(2)
```

两件要紧的事:

- **它含 `goto` 和 `go_back`**,两个都来自 `nav` 子集 —— 环境确实给了(见 1.4 的 20 个),
  所以慢思考写它们是合法的。**但「快思考在任何配置下都不能做导航」是错的**:`go_back()`
  的元数是 **0**,它和 `scroll` 一样是常量动作,加一个候选槽就能枚举。真正枚举不了的是
  `goto(url)` —— url 要生成。这里原来把两个合在一起说,于是把一个槽位的欠缺说成了结构性
  的不可能。
- **它不含** `clear`、`dblclick`、`drag_and_drop`、`focus`、`upload_file`、`noop`。所以慢思考**不是**快思考的超集,两者是交叉的:快思考有 `dblclick`/`focus`/`clear` 的枚举能力(虽然目前没接上),慢思考有 `goto`。

解析规则:检查元数、bid 必须在页面上、一行里最后一个调用胜出(`Final: click('42')` 和 `click('42')  # done` 都接受)、参数经 `repr` 渲染再用 `ast` 验证成"一个调用 + 全是字面常量"、解析不出来的动作被记录并结束该 episode(**从不**替换成一次点击)。

> `browser.py` 顶部那段注释写的是"BrowserGym 的 `chat`+`bid` 动作集提供十三个",
> 而实测环境给的是 **20** 个(1.4)。那个 13 从来没被量过,它是把 `bid`(11)+`chat`(1)
> +`noop`(1) 加出来的,而构造函数还加了 `infeas`、`nav`、`tab`。注释里接着的
> "需要内容 5 + 无内容 7" 只点了 12 个,无内容那边漏了 `clear`。**两处都是推算当成实测
> 的产物**,改注释不影响代码行为,但这一节引用的数必须是 20。

> **我要更正自己**:我在对话里多次把快思考说成"拿到 13 个动作里无内容的那一半"。错的。是 click + scroll 两类。代码没错,我的概括错了。

---

## 2. MiniWoB

### 2.1 任务划分

125 个任务被切成三组,三组**不池化**(预登记) **【实测,`--list` 对启动树】**:

| 组 | 任务数 | 含义 |
|---|---|---|
| `click` | 54 | 主组,跑两个种子 |
| `pointer` | 18 | 对照:应只被 `always` 赢下,且只靠"够得着" |
| `beyond-click` | 53 | 对照:若 `never` 在这里成功,说明划分错了 |
| 合计 | **125** | 正好是 registry 全量 |

### 2.2 click 组结果(唯一有判定的组)

**【入库】** `experiments/miniwob/results/`,由 gate 自己的 `--combine` 产出,不是我做的平均:

| | never | always | oracle | both_right | helps | hurts | reach | reasoning | bad |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 (28831) | 0.222 | 0.481 | 0.481 | 0.222 | 0.259 | **0.000** | 0.259 | **0.000** | 1 |
| seed 1 (28832/28844) | 0.370 | 0.556 | 0.630 | 0.296 | 0.259 | 0.074 | ~~0.222~~ **0.259** | ~~0.037~~ **0.000** | 4 |
| **合并 108 对** | **0.296** | **0.519** | **0.556** | **0.259** | **0.259** | 0.037 | ~~0.241~~ **0.259** | ~~0.019~~ **0.000** | 5 |

合并判定 **PASS**,精确符号检验 **p = 9.65e-06**,helps 28 个任务 / hurts 4 个,`incomplete_pairs` 0。

### 2.3 三件从这张表里读出来、但单看一个种子会读错的事

1. ~~**`helps_by_reasoning` 不是 0。**~~ **撤回 2026-10-06:那个 0.037 是我们自己的
   bug,`helps_by_reasoning` 在两个种子上都是 0。** `episode_record` 的键列表漏掉了
   `used_outside_action`,而 `rebuild_pairs` 在**续跑**时从 episode 行拼 pairs ——
   于是 `.get()` 返回 `None`,`not None` 为真,每一对读回来的 helps 都被计成
   reasoning。28832 挂过(`hang-stack-28832.txt`)、被续跑,它的 2 个就是这么来的。

   从 step row 的 `outside_fast_set` 重算(这个字段每次门控运行都写了),
   code-reviewer 独立复现一致:

   ```
   seed 0    14 helps   14 reach   0 reasoning    （和它的 JSON 一致）
   seed 1    14 helps   14 reach   0 reasoning    （它的 JSON 说 12 / 2）
   合并      28 helps   28 reach   0 reasoning    （两个 JSON 说 26 / 2）
   ```

   **所以"同动作空间内的推理一个都没赢"是对的,而我为了避免它而改成的那个比例是错的。**
   修在 `5ca7ed3`:写入端补齐、`four_way` 改成下标访问让旧文件报错而不是给错答案、
   再加一个喂修复前行的测试。
2. **`oracle == always` 只在 seed 0 成立。** seed 1 的 `hurts` 0.074 让 oracle 高出 always **+0.074**,合并 +0.037。**我曾据此论证"always 之上没有正确率空间、平局是最好结果",那个论证基于一个种子,已撤回。**
3. **种子间差异很大。** 同样 54 个任务,`never` 在两个种子间差 **15 个百分点**(0.222 / 0.370);`hurts` 0 对 4;`bad` 1 对 4。**任何单种子 MiniWoB 数字要么用合并值,要么标明种子。**

### 2.4 作废的那一次

**28816 没有判定,也不会有。** 它只跑了 144 个 episode 里的 28 个,而且两个设计缺陷同时在:快思考只有 13 个动作里的 1 个,慢思考又从同一个纯点击候选表里读。它那个 `always − never = +0.000` 被这两个缺陷决定。记录为**被设计缺陷取代,而非被它的数字取代**。它唯一保留的发现:四个任务在 1–2 步内不思考完成,所以快思考确实能完成多步浏览器 episode。

---

## 3. WebArena

### 3.1 任务文件

`test.raw.json`,**812 个任务**,sha256 `7b50386fd69163dbc05d615d834df4c6ed2c35596e97a1b10d17451c02537652` —— 与 `src/slime_htm/envs/webarena-plan.json` 预登记的摘要逐位一致,所以下面数的是同一份文件。**【实测】**

每个任务带 `sites`、`intent`、`intent_template_id`、`eval`。评测器四个通道:

| 通道 | 判据 | 快思考(click+scroll)能否达到 |
|---|---|---|
| `answer` | `reference_answers`,读**发送出去的答案** | 不能,任何价格都不能(要 `send_msg_to_user` 带文本) |
| `page` | `program_html.required_contents`,文本必须**出现在页面上** | 不能。在论坛或 issue tracker 上这意味着有人打出来的正文 |
| `url_search` | `reference_url` 带**自由文本检索**参数 | 不能,要往输入框里打字 |
| 无 | 以上皆无 | **可能**,这是点击可达性的**上界** —— 这样的任务仍可能需要一条纯点击走不通的路径 |

### 3.2 我第一版规则是错的,这里记下来

我最初把 **每一个** query 参数都当成"要打字",得出 shopping 22 / gitlab 2。先把任务文件里出现过的全部 query key 列出来才看清 **【实测】**:

```
label_name[] 10   product_list_order 8   price 5   product_list_dir 3
sort 2   state 2   assignee_username 1   reviewer_username 1     ← 筛选/排序,点出来的
q 10     query 5   search 1                                      ← 自由文本,打出来的
```

`?label_name[]=help%20wanted` 正是**在 GitLab 界面上点一下标签筛选**产生的 URL。把"要打字"收窄到 `{q, query, search, keywords}` 之后,结果与转述的计数**逐站点、逐 intent 精确吻合**。我那一版会让结论更强(更少可达),但理由是错的。

### 3.3 逐站点计数

单站点任务(一个容器就是一个站点,多站点任务排除),**【实测】**:

| 站点 | 可用任务 | 无需打字 | 去重后 distinct intent | 镜像体积(压缩) |
|---|---|---|---|---|
| shopping | 187 | **32** | **8** | 5.0 GB |
| gitlab | 180 | 15 | 7 | 20.5 GB |
| shopping_admin | 182 | 3 | 2 | 1.2 GB |
| map | 109 | **0** | **0** | 1.1 GB |
| reddit | 106 | **0** | **0** | 4.3 GB |

去重按 `intent_template_id`:shopping 的 158–162 是同一个模板的五次填充,而符号检验是**按任务**配对的,所以五个填充不是五个独立单位。

### 3.4 判定

**`MIN_PAIRS = 20`** —— 低于它 `verdict()` 返回 `pass = None`,因为符号检验和 margin 都说不了任何话。

- **最好的站点只有 8 个 distinct intent,需要 20。**
- **把所有站点合并**(预登记明令禁止,因为三组不池化)也只有 **50 个任务 / 17 个 distinct intent** **【实测】** —— **连破规矩都够不着底线。**
- `reddit` 与 `map` 上 `never` **恰好是 0**,于是 `both_right = 0`,门槛 `both_right >= 0.20` **按构造必然失败**。

**所以这不是选站点的问题,是动作空间撞上场地。** 对照 1.2 节:`webarena` 预设自己的 14 个动作里 `fill`、`select_option`、`send_msg_to_user` 都在 —— **这个 benchmark 的原生动作空间本来就假定会打字**,而我们砍掉的正是它评测器依赖的那部分。

### 3.5 还没有答案的两个前置问题

无论选哪种快思考,这两个都必须先答(预检尚未获批,未提交):

1. **`POST /init` 到底会不会恢复状态?** 每个 `am1n3e/webarena-verified-*` 镜像在 8877 上有控制面 `/status /is_ready /init /start /stop /restart`,**没有 `/reset`**;`/init` 在 10 秒内没返回。测试必须是破坏性的:取指纹 → 改动 → 确认指纹变了 → 调 `/init` → 看是否变回。**中间那步改动不做,这个测试什么都证明不了** —— 会恢复的 `/init` 和什么都不做的 `/init` 从外面看一样。`init_seconds` 决定每题重置在一次运行里放不放得下。 **【转述】**
2. **原始 task id 在这些镜像上还存不存在?** webarena-verified **重写了任务定义和评测器**,而 `gate.py` 读 libwebarena 的原始 `test.raw.json`。若 40 个预登记 id 有缺失,**两者不是同一个 benchmark**,计划要重建而不是绕过。 **【转述】**

### 3.6 镜像来源

archive.org 的 WebArena 条目已下架(`is_dark: true`,每个 tar 403),CMU 镜像站不可用。可用的是 Docker Hub 的 `am1n3e/webarena-verified-*`,六个站点约 **32 GB 压缩**,对原始的 200 GB+。 **【转述】**

端口必须取自 `scripts/webarena_env.py:docker_run(site)` 而不是写进启动器:它们是 WebArena 自己的端口,`url_match` 评测器硬编码了它们,**改端口会让任务以"像是 agent 出错"的方式失败**。

---

## 4. 其他数据集的任务情况

| 数据集 | 题量 | 形态 | 关键结果 |
|---|---|---|---|
| MMLU-Pro | 测试集按 stride 8 取 **1500** 题,选项 4–10 个 | 多选,`load_mmlu_pro` **不打乱**选项 | **【入库】** 27B:coded 0.6107 / sum **0.646** / mean 0.638 / pmi 0.5127 |
| AGIEval | **335** 题(available 2340,stride 7) | 多选 | **【入库】** fast 0.7164,slow 0.8776;λ=0.1 升级 0.504 拿到思考收益的 99.3% |
| GPQA-Diamond | **198** 题(全量) | 四选 | **【入库】** fast 0.5404,slow 0.7424;**五个 λ 全部等于 slow**,λ=0.1 升级率 0.82 —— 路由退化成 always-think |
| JevBench | 公开 **231** 条 | 139 `choice` / 74 `noul` / 18 `score` | **【转述】** λ=0 给 0.922,对 fast 0.861、slow 0.957;λ=0.1 以 3.5 点换 P95 快 5.8× |
| MATH-500 | 500 题 | 自由作答 | 27866 跑到 365/500 被我误取消;365 行从日志抢救,`slow_text` 永久丢失 |

一个跨数据集的量 **【入库】** `gold-length-profile-three-benchmarks.json`:MMLU-Pro 上正确选项的长度百分位是 **0.5539**(字符)/ 0.5461(词)—— 正确答案普遍**偏长**,但各子类差异大(biology 0.724,law 0.552),且有子类反向。这是 `sum` 优于 `mean` 的迁移论证所依赖的事实。

---

## 5. 跨数据集的结论

1. **决定一个 gate 能否产出判定的,不是 benchmark 多大,而是"我们的动作空间 ∩ benchmark 评测维度"有多大。** MiniWoB 的交集够大(54 个 click 任务,12 个 `both_right`,合并 p = 9.65e-06);WebArena 在评测维度上的交集是**空的**(最好站点 8 个 intent / 需要 20)。

2. **两次"路由退化"是镜像的。** GPQA:几乎每题都需要思考,fast 0.540 对 slow 0.742,于是五个 λ 全部等于 always-think。WebArena:几乎每题都需要打字,于是 `never` 归零。**两者都不是 bug,都是关于场地的事实。**

3. **同一个缺陷在三个尺度上出现过,形状相同。** 28816:快思考只有 1/13 个动作 → `always − never = +0.000`。28845:快思考没有 scroll 且被过滤到首屏 → 13.8% episode 以 `no_steps` 丢弃,活下来的每个都只在一屏里选。WebArena:快思考不会打字 → 0 个任务可达。**每次都是"动作空间被削,结果读起来像方法的结论"。**

4. **单种子数字不是性质,是测量** —— 但这一条我举的两个例子里有一个是假的,而**那个
   假例子恰好支持了我从它得出的教训**,所以连教训一起改。

   `oracle == always` 确实只在 seed 0 成立,那个例子站得住。
   `helps_by_reasoning` **两个种子都是 0**:seed 1 的 0.037 是上面第 1 条那个 bug。

   原来写的是"宁可给比例或合并值,也不要给一个零 —— 一个零会邀请一个一般陈述"。
   **实际发生的是反过来的:零是测量,比例是 bug,而我因为怕那个零太强而换上了比例。**
   所以正确的教训不是"避开零",是:**一个让你意外的数字,先去复现它,再去为它改句子。**
   seed 1 的 0.037 从行里复现不出来 —— exps 在 2026-10-07 就说过"这个拆分从已入库数据
   里不可验证,两个可得来源给 28/0 和 26/2,都不该当成已确立" —— 而我当时已经把比例
   写进了文件。

---

## 附:重跑这些计数的命令

```bash
# browsergym 动作子集
python -c "from browsergym.core.action.highlevel import ACTION_SUBSETS
for n,f in sorted(ACTION_SUBSETS.items()): print(n, len(f), sorted(x.__name__ for x in f))"

# 每个动作的真实签名,也就是 1.4 那张表的依据
python -c "
import ast, inspect
from browsergym.core.action import functions
tree = ast.parse(inspect.getsource(functions))
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
        print(n.name, '|', ', '.join(
            f'{a.arg}: {ast.unparse(a.annotation) if a.annotation else \"?\"}'
            for a in n.args.args) or '(no args)')"

# 定义了但任何子集都拿不到的动作(实测是 check / uncheck;noop 不在子集里但永远可用)
python -c "
import ast, inspect
from browsergym.core.action import functions
from browsergym.core.action.highlevel import ACTION_SUBSETS
defined = {n.name for n in ast.parse(inspect.getsource(functions)).body
           if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')}
offered = {f.__name__ for fs in ACTION_SUBSETS.values() for f in fs}
print('defined', len(defined), 'offered', len(offered),
      'unreachable', sorted(defined - offered))"

# 两份动作清单(1.6),连同和上游签名的交叉验证
python scripts/browsergym_actions.py -o src/slime_htm/envs/browsergym-actions.json

# MiniWoB 分组
python -m slime_htm.envs.gate --venue miniwob --groups click --list | head -1

# WebArena 逐站点(规则见 3.1、3.2;TYPED_KEYS = {q, query, search, keywords})
python scripts/webarena_plan.py        # 产出 src/slime_htm/envs/webarena-plan.json

# MiniWoB 合并判定
python -m slime_htm.envs.gate --venue miniwob \
  --combine experiments/miniwob/results/miniwob-click-seed0-28831.json,\
experiments/miniwob/results/miniwob-click-seed1-28832.json --output /tmp/combined.json
```
