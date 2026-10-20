# 攀岩视频对比 — 本次方案总结(2026-10-13)

## 方案全貌

```
视频 A, B
   │
   ├─► Step 1: 紫色岩点识别(每视频独立)
   │   ✅ 做对了:HSV + 多重过滤 + 可选 Jev 判定
   │   A: 38 strong + 21 weak;B: 40 strong + 38 weak
   │
   ├─► Step 2: 覆盖帧区间 + 攀爬 step 事件(每视频独立)
   │   ✅ 做对了:每帧 limb→hold assignment + 事件流
   │   A: 67 事件(16 acquire / 14 release / 37 swap)
   │   B: 72 事件
   │
   ├─► Step 3: 跨视频 hold 匹配(建立共享词典)
   │   ⚠️  部分成功:DTW on y_frac + hue 得到 19 对 "both" 匹配
   │   加 15 A-only + 13 B-only,共 47 canonical holds
   │
   └─► Step 4: step 配对(跨视频同一 beta)
       ❌ 失败:无论用什么规则,配对都"对不上"
```

## 哪里对了,哪里错了

### ✅ 做得好的部分

**岩点识别(Step 1)**
- 从 7 个 → 10 → 27(单视频精度 85%+)
- 关键参数:`HUE_MIN=112` 排除蓝墙是转折点
- 加入 Jev 作为 fall-back 验证,token 成本可控(~15k 一个视频)
- `src/climbing_compare/purple_holds.py` 统一两种方法(strict / jev)

**覆盖帧区间(Step 2)**
- 从稀疏触碰 → 连续区间(gap_tol + min_run)
- Timeline chart 清晰显示攀爬进程
- 发现并剔除下降段(topout 之后)
- `src/climbing_compare/coverage.py` 可复用

**Step 事件流(Step 2.5)**
- 肢体-hold assignment + 事件生成
- 文件级 step snapshot(67 张图)可用来人工复现攀登动作
- `src/climbing_compare/steps.py` 可复用

**Canonical hold 词典(Step 3a)**
- A/B hold DTW 配对(19 对)
- 保留 a_only/b_only(28 个独有)
- 用同一 C01-C47 ID 让两边 steps 直接可比
- 这个数据结构**正确**,下游任何对比都应基于它

### ❌ 失败的部分 — Step 配对

**尝试的 4 种配对规则**(全部不理想):

1. **(limb, hold) first-touch 相同**(`pair_steps_ab.py`)
   - 17 对配对,但大量 A_only/B_only(28+34)
   - 两个 climber 很少用同一肢体登同一 hold

2. **接触集完全相同**(`pair_states_ab.py` v1)
   - 0 对配对
   - 4 肢体位置组合在两视频中几乎永不重合

3. **累计访问集相同**(v2)
   - 0 对配对
   - 两人访问 hold 顺序差异大(A 从中部起手,B 从底部按序)

4. **"最高进度 cid" 对齐**(v3,shared-only)
   - 8 对配对,但图像看起来仍不像"同一瞬间"
   - 因为两人 limb-hold 组合在同一高度上完全不同

5. **按同一 canonical hold 首次登上**(`pair_by_hold.py`)
   - 15 对,但 **11/15 对两人用的肢体不同**
   - 用户反馈"完全对不上"

## 根本原因:**两段攀爬 beta 差异过大**

```
视频 A climber 行为:
  - 开始帧:手已经扣在中部 hold(C09)上
  - 攀爬节奏:快,先手后脚
  - 多次 matching(两手同一 hold)
  - 对墙熟悉

视频 B climber 行为:  
  - 开始帧:在地面 → 走到墙前 → 调整起手
  - 攀爬节奏:慢,脚先踩稳再手
  - 按部就班从 C01 逐步往上
  - 对墙陌生
```

**不是算法问题,是数据问题**:两次攀爬**本质上就是不同的 beta**。要让 step-level 对齐,前提是两人**用同一套动作**攀同一条路线,或至少动作序列高度相似。

现实中观察到的差异:
- 起手位置不同
- 手脚使用策略不同(A 爱用脚踩,B 爱手抓)
- 中间 matching 次数不同
- 节奏快慢差 2 倍

## 可以继承的资产

**代码**(`src/climbing_compare/`):
- `purple_holds.py` — 岩点检测(two methods, unified schema)
- `coverage.py` — 覆盖区间
- `steps.py` — step 事件流
- `align.py` — DTW hold 配对
- `canonical.py` — 跨视频 canonical hold 词典

**数据产物**(`hold_detect_diag/`):
- `A/holds_strict.json`, `A/coverage_canonical.json`, `A/steps_canonical.json`
- 同上 for B
- `canonical_holds.json` — 47 个 canonical hold + A/B 映射
- `canonical_overlay.png` — A/B 并排 canonical ID 标注

**可视化**:
- `A/steps/*.png` — 67 张单步 snapshot
- `B/steps/*.png` — 72 张
- `route_map.png` — 攀爬路径箭头图

## 建议新方案的方向

**问题的本质**:step 粒度对两次 beta 不同的攀爬做对齐,是**无解的**。因为 "step" 定义依赖 limb 配置,而两人根本不用相同配置。

### 建议 1:**只做 hold 级对比,不做 step 级**(放弃 step 对齐)
对每个共享 canonical hold,对比两人:
- 停留时长(touch intervals 总长)
- 使用了哪些肢体(集合,不管顺序)
- 肢体接触姿势的关节角度(climber 身体平均几何)
- 不比"谁先谁后"、"谁快谁慢"

这不需要时间对齐,也不需要 step 对应。**适合你的核心需求"看 beta 差异"**。

### 建议 2:**端到端训练一个动作识别模型**
- 用 I3D / SlowFast / TimeSformer 跑两个视频,得到动作段落
- 用 contrastive learning 把"动作内容"向量化
- 两边视频向量相似度 → 对齐
- 要训练数据,成本高

### 建议 3:**纯姿态 DTW,放弃岩点**
回到最早的 pose-based 方案,但做得更好:
- 两边 pose 的 **归一化**(以墙为参考系,去除相机差异)
- DTW on 关节角度向量
- 关键帧 = DTW cost 曲线的 local min(两人在那一刻姿态最像)
- 一开始失败是因为没做机位配准 — 现在你有干净帧和共享 hold,可以用它们**估相机单应性**,再做 pose 归一化

### 建议 4:**接受"不能完美对齐",改做分段统计分析**
- 两段视频各自分成 N 段(如按高度 y_frac 分 10 段)
- 每段统计:耗时、使用的 hold、动作次数、停留长度
- 直接出一张"分段 beta 对比表",不画时间对齐图
- 简单但对用户很有信息量

---

**哪个方向展开,我都可以帮你起始。目前的代码已经把 hold-level 分析做得很干净,建议 1(hold 级 beta 对比)是最快能产出有用结果的方向。**
