# 图像反馈液滴生成：多方向文献、开源资源与可落地创新

检索与核查日期：2026-09-16。研究主线为“图像识别状态 → 控制或改良液滴生成”，不限定电、磁、声或某一种仪器。

## 一、范围与证据口径

本轮通过公开网络检索出版社、PubMed/PMC、作者机构全文和 GitHub，覆盖视觉反馈、贝叶斯优化、强化学习、几何调节、流体缓冲、压力控制、按需包封、二次分滴、质量检测与开源硬件。不是全网穷尽检索，也不是正式新颖性查全。搜索引擎的抓取日期不当作论文发表日期。

原有 [23 篇辅助执行器文献](active_droplet_generation_innovation_20260915_zh.md) 保留作为物理机制库；本文补充软件、流体结构、观察方式和应用目标。每个方法后直接列论文题名、刊物、年份与链接。论文事实和本项目建议分开陈述，不把建议的组合算作论文已验证结果。

当前条件：两路可控供液、工业相机、十字交汇生成芯片，用户已有 DEP 分选经验。现有信号发生器按人工设置处理，但这不限制研究其他可通过串口、微控制器或电机驱动器调节的附件。新增附件是否可控须逐个核实接口，不能预设已经接入。

项目现状：已有尺寸反馈、独立频率监测、BO 寻优、动力学标定和前馈框架；已有功能不是新增创新。`docs/control_optimization_roadmap.md` 记录的当前批次控制仍受泵通信与响应限制，不应把约 100 FPS 采集能力等同于约 100 Hz 闭环能力。

实施等级：**A：现有装置/离线数据可先验证；B：加装附件或局部芯片改版；C：显著改造或另建平台。** 等级表示项目适配判断，不是采购报价或效果保证。

## 二、18 条具体方法

### 1. 自动寻找“稳定且合格”的生成区域【A】

**文献已做：**视觉测量与 BO 配合选择实验参数；较新工作还将形态分类和数据质量筛选纳入实验流程。

**本项目可做：**从“找到一个目标尺寸点”扩展为“找到对小扰动不敏感的一片合格区域”。在已有 BO 上增加稳定性地图，记录目标误差、尺寸 CV、异常比例、试剂消耗和调节时间；工作点优先选在区域内部。

**最小实验：**固定芯片/配方，对已有流量空间做有限扫描，按完整实验批次标注稳定区，再比较现有 BO 与加入质量约束后的选择。以达标所需实验次数、有效滴比例和复现性评价，不能只比较优化损失。

**边界：**BO+视觉早已有先例；创新应落在可靠性约束、稀缺数据或跨装置验证。Cho 工作中的频率由流量与体积关系计算，不能视为独立逐滴计数证据。

**文献：**Siemenn et al., *A Machine Learning and Computer Vision Approach to Rapidly Optimize Multiscale Droplet Generation*, ACS Applied Materials & Interfaces, 2022, 14:4668–4679，[DOI](https://doi.org/10.1021/acsami.1c19276)；Cho et al., *Autonomous Bayesian Optimization-Based Control System for Droplet Generation*, Small Methods, 2025，[DOI](https://doi.org/10.1002/smtd.202500984)。[开源实现](https://github.com/PV-Lab/ML-Multiscale-Droplets)。

### 2. 换芯片、换配方后的少样本重新标定【A】

**文献已做：**DAFD 将几何、流量与生成性能关联，研究设计反求与迁移学习。

**本项目可做：**保存历次芯片/配方模型，新工况只做少量局部试验，校正增益、延迟和可达范围。模型不确定性高时退回重新标定，而不是直接沿用旧参数。

**最小实验：**按整块芯片或整次配方分训练/测试组；比较完全重标定、直接搬旧模型和少样本更新的误差及实验次数。

**边界：**在同一录像随机分帧不能证明跨芯片泛化；DAFD 预测稳态性能不等于已解决当前泵的动态滞后。

**文献：**Lashkaripour et al., *Machine learning enables design automation of microfluidic flow-focusing droplet generation*, Nature Communications, 2021, 12:25，[DOI](https://doi.org/10.1038/s41467-020-20284-z)。[GitHub：DAFD](https://github.com/CIDARLAB/DAFD)。此论文已在旧文献库中，本轮补充开源入口及新的项目验证目标。

### 3. 图像状态驱动的策略控制，与 PI 做公平比较【A，后期】

**文献已做：**通过显微图像和强化学习调节泵流量，控制油包水滴尺寸。

**本项目可做：**把尺寸、变化趋势、生成模式、历史命令和测量有效性作为状态。先离线回放/仿真，再在已验证范围内试验；研究是否减少大幅目标变化时的过渡废滴。

**最小实验：**固定同一执行器边界和观察信息，比较 PI、局部预测控制和学习策略。记录训练消耗与失败试验，不能只报告训练完成后的最佳效果。

**边界：**换成强化学习本身不是充分创新；没有可信模拟环境和回退策略时，不宜先让策略在真实装置上探索。

**文献：**Dressler, Howes, Choo & deMello, *Reinforcement Learning for Dynamic Microfluidic Control*, ACS Omega, 2018, 3:10084–10091，[DOI](https://doi.org/10.1021/acsomega.8b01485)；[开放正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6644574/)。

### 4. 生成口与下游两个观察区，提前发现失稳【A/B】

**文献已做：**实时相位成像用于研究颈缩并结合压力脉冲反馈；视频分析可提取尺寸、速度及形变。

**本项目可做：**生成口观察液舌长度、断裂位置漂移、可分辨的颈部状态；下游检查最终体积和间距。提前信号只在领先时间大于执行延迟时用于前馈，否则仅用于预警。

**最小实验：**用既有视频标注异常开始时间，比较“仅下游尺寸”与“增加生成口状态”的预警提前量和误报率。不要先宣称能逐滴预测毫秒级断裂。

**边界：**普通明场图像不具备定量相位成像的三维测量能力；两个区域不在同一视野内时需分阶段实验或第二观察通道。

**文献：***Model-based feedback control for on-demand droplet dispensing system with precise real-time phase imaging*, Sensors and Actuators B: Chemical, 2022, 365:131936，[DOI](https://doi.org/10.1016/j.snb.2022.131936)；Basu, *Droplet morphometry and velocimetry (DMV): a video processing software for time-resolved, label-free tracking of droplet parameters*, Lab on a Chip, 2013, 13:1892–1901，[DOI](https://doi.org/10.1039/C3LC50074H)。两观察区的因果预警设计是本项目建议。

### 5. 管路阻尼与图像闭环共同设计【A/B，实用优先】

**文献已做：**利用弹性外接管路的柔顺性衰减流源波动，已用于注射泵驱动的液滴生成。

**本项目可做：**用几种可更换管路/缓冲配置，对比液滴波动和泵阶跃响应，寻找“稳态均匀度—调节速度”的折中。初版手动换模块即可验证；可切换旁路需要额外阀门与标定，不能把被动管路称为新增自动执行器。

**最小实验：**相同芯片、配方和供液基准，记录逐滴尺寸时间序列与小阶跃响应。评价波动幅度、周期成分、延迟、稳定时间，而非只看一张尺寸直方图。

**边界：**阻尼可能增加记忆和滞后；论文报告的 0.39% 为面积 CV，不可直接写成直径或体积 CV，也不能作为本项目保证。

**文献：***Damping hydrodynamic fluctuations in microfluidic systems*, Chemical Engineering Science, 2018, 178:238–247，[DOI](https://doi.org/10.1016/j.ces.2017.12.045)；*Standing Air Bubble-Based Micro-Hydraulic Capacitors for Flow Stabilization in Syringe Pump-Driven Systems*, Micromachines, 2020, 11:396，[DOI](https://doi.org/10.3390/mi11040396)。后一种需专门捕获气腔，不能往主供液管随意引入气泡。

### 6. 加入可编程压力调节，做压力内环与视觉外环【B】

**文献已做：**公开压力泵方案使用压力传感器、调节器、微控制器和电脑接口，并验证微流控出滴。

**本项目可做：**慢速视觉外环给目标，局部压力内环稳定执行。可以作为替代供液或独立控制腔驱动；具体液路拓扑确认前不把压力源与注射泵串联叠加。

**最小实验：**先测无芯片负载及代表性流阻下的压力响应，再接芯片，与原注射泵比较总响应及漂移。压力响应快不等于液滴响应一定快。

**文献：**Gao et al., *µPump: An open-source pressure pump for precision fluid handling in microfluidics*, HardwareX, 2020, 7:e00096，[DOI](https://doi.org/10.1016/j.ohx.2020.e00096)；*Open-source pneumatic pressure pump for drop-based microfluidic flow controls*, Engineering Research Express, 2023, 5:035014，[DOI](https://doi.org/10.1088/2631-8695/ace299)。补充开源硬件：[Rio controller](https://github.com/wenzel-lab/rio-controller)。

### 7. 电机调柔性流道或喷口间隙【B】

**文献已做：**柔性壁同轴发生器可通过改变连续相通道宽度调滴；聚焦口膜阀可改变局部几何。

**本项目可做：**先微分头手动调节，再用带限位的电机/位移台；图像观察实际间隙与最终滴径，避免把电机步数直接当作真实间隙。这个新增控制量不依赖现有信号发生器。

**最小实验：**固定供液，做位移上行/下行循环，量化调节范围、迟滞、偏心、泄漏及重复性，再决定是否自动化。

**边界：**电机化及视觉补偿是这里提出的改进；原文并非已经证明此组合。刚性玻璃芯片不能直接夹压，需要柔性生成段或改版。

**文献：**Yazdanparast, Rezai & Amirfazli, *Microfluidic Droplet-Generation Device with Flexible Walls*, Micromachines, 2023, 14:1770，[DOI](https://doi.org/10.3390/mi14091770)；Abate et al., *Valve-based flow focusing for drop formation*, Applied Physics Letters, 2009, 94:023503，[DOI](https://doi.org/10.1063/1.3067862)。

### 8. 可调阶梯乳化：通过几何扩大尺寸范围【C】

**文献已做：**Tuna-step 用受压薄膜改变阶梯喷口高度，并实现多喷口并联。

**本项目可做：**图像反馈调膜片压力，将泵主要用于产量调节；研究换配方后是否能通过少量标定恢复性能。多喷口阶段逐口统计，不能只测混合输出平均值。

**最小实验：**单喷口完成压力—体积—频率关系后再并联；优先证明扩大可调范围且维持均匀度。

**边界：**需要更换生成结构；固定分散相流量时体积与频率仍满足守恒关系，不能任意独立设定。

**文献：**Nalin et al., *Tuna-step: tunable parallelized step emulsification for the generation of droplets with dynamic volume control to 3D print functionally graded porous materials*, Lab on a Chip, 2024, 24:113–126，[DOI](https://doi.org/10.1039/D3LC00658A)。

### 9. 看见颗粒再生成：按需单颗粒包封【B/C】

**文献已做：**实时图像识别配合微阀按需生成，验证无标记单细胞/微球包封。

**本项目可做：**先用可见微球建立检测—到达时间预测—阀触发—包封结果核对，再扩展磁珠。评价单珠率、空滴率、多珠率和有效产量，而不是只调滴径。

**最小实验：**低速验证单颗粒闭环及延迟分布，再逐步提速。已有慢速泵只维持供液，逐事件触发需要近芯片快速阀和合适驱动。

**边界：**论文摘要的 150 Hz 是理论对应吞吐，不能等同于本装置实测；原文包封效率也不能直接搬用。需核实当前相机能否解析颗粒与到达时间。

**文献：**Wang et al., *Label-free active single-cell encapsulation enabled by microvalve-based on-demand droplet generation and real-time image processing*, Talanta, 2024, 276:126299，[DOI](https://doi.org/10.1016/j.talanta.2024.126299)。

### 10. 根据空滴/单珠/多珠比例，慢速优化包封工况【A/B】

**文献已做：**YOLO 可先定位液滴，再识别内部细胞，统计包封分布；有论文配套代码。

**本项目可做：**复用磁珠统计模块，按窗口统计结果，以目标包封比例为目标调供液；浓度先人工分组。若想在体积固定时独立调浓度，需要额外稀释/混合支路。

**最小实验：**人工标注空滴、单珠、多珠和无法判断样本；比较不同工况下的统计准确性与合格滴产量。用扩张观察段提高内容物可见性时，要检查是否引入合并。

**边界：**仅改变平均浓度不能承诺突破随机泊松包封限制；本方法与第 9 条逐颗粒触发不同。

**文献：**Gardner et al., *Deep learning detector for high precision monitoring of cell encapsulation statistics in microfluidic droplets*, Lab on a Chip, 2022，[DOI](https://doi.org/10.1039/D2LC00462C)。[GitHub](https://github.com/karl-gardner/droplet_detection)。

### 11. 先生成母滴，再视觉反馈分出目标体积【B/C】

**文献已做：**图像反馈和压力网络可捕获、分裂、合并液滴；膜位移陷阱平台可执行可编程操作。

**本项目可做：**上游只需稳定地产生较大母滴，下游根据视觉量取目标子滴，研究扩大体积覆盖范围、按需剂量序列。明确这是生成后的二次加工。

**最小实验：**低频捕获一个母滴，量取分裂前后体积并核对守恒；与直接生成目标滴比较误差、速度及回收率。

**边界：**增加液路与控制复杂度，吞吐可能下降；有细胞/颗粒时分裂可能改变内容物分配。

**文献：**Wong & Ren, *Microfluidic droplet trapping, splitting and merging with feedback controls and state space modelling*, Lab on a Chip, 2016, 16:3317–3329，[DOI](https://doi.org/10.1039/C6LC00626D)；Harriot et al., *Programmable Control of Nanoliter Droplet Arrays Using Membrane Displacement Traps*, Advanced Materials Technologies，2023 年首次上线，[DOI](https://doi.org/10.1002/admt.202300963)。

### 12. 生成—间距—分选协同优化【A/B】

**文献已做：**2026 年工作将生成、油相间距调节、图像检测和 DEP 分选整合在同一系统。

**本项目可做：**利用已有分选经验，以“单位时间收集的目标合格滴数/消耗试剂”为目标，调生成工况或额外间隔油流量；重点减少滴间干扰。

**最小实验：**保持分选判据，比较不同生成工况下的漏分、误分、合并和收集产量。生成总数、剔除数和收集数分别记录。

**边界：**下游分选成功不等于主动单细胞包封；新增间隔油通常需要独立支路/供液能力。本文建议的全链路优化不是该论文自动等价实现。

**文献：***Intelligent label-free droplet microfluidic sorting system for single-cell encapsulation and morphology-guided screening*, Microsystems & Nanoengineering, 2026，[原文](https://www.nature.com/articles/s41378-026-01349-3)。

### 13. 图像检测卫星滴，优化生成并做结构去除【A/B/C】

**文献已做：**阶梯乳化与确定性侧向位移（DLD）结构结合，分离主滴和卫星滴。

**本项目可做：**先把卫星滴数和体积分数加入图像评价，优化泵工况；若仍不合格，再考虑下游去除结构。生成前后分开观察，区分抑制产生与产生后去除。

**最小实验：**高分辨离线标注卫星滴，确认最小可检尺寸；报告漏检下限、主滴回收率和卫星污染率。

**边界：**DLD 通常需要芯片改版且会增加压降；小卫星滴若低于光学分辨能力不能当作不存在。

**文献：***Microfluidic Coupling of Step Emulsification and Deterministic Lateral Displacement for Producing Satellite-Free Droplets and Particles*, Micromachines, 2023, 14:622，[DOI](https://doi.org/10.3390/mi14030622)。

### 14. 合成图像＋少量真实标注，提高跨工况测量可靠性【A】

**文献已做：**BYG-Drop 将合成图像、图像风格转换和目标检测结合，有权重及处理脚本。

**本项目可做：**构造光晕、亮度漂移、拖影和背景纹理变化，训练/验证检测的拒识能力。首要目标是避免错误反馈，不是只提高平均检测率。

**最小实验：**同一真实独立测试集比较现有检测器、合成增强和真实增强；统计尺寸偏差、漏检、假阳性及由误检诱发的错误控制次数。

**边界：**BYG-Drop 面向球形滴；项目塞状滴不能直接以检测框宽度当等效球径。相邻帧不能跨训练/测试泄漏；合成图像不能替代物理标定。

**文献：***BYG-Drop, a tool for enhanced droplet detection in liquid-liquid systems through machine learning and synthetic imaging*, Frontiers in Chemical Engineering, 2024, 6:1415453，[DOI](https://doi.org/10.3389/fceng.2024.1415453)。[GitHub](https://github.com/banag0/BYGDrop)。

### 15. 短曝光/频闪辅助视觉，减少运动模糊【B，基础设施】

**已公开资源：**开源频闪显微平台提供 LED 与相机同步的搭建方案。

**本项目可做：**保留相机和显微光路，先改善照明和曝光；需要时加入同步 LED，比较同一液滴速度下的轮廓偏差和有效测量率。

**最小实验：**连续照明与短脉冲照明对照，人工复核边缘位置、尺寸及计数，不只展示图像更清晰。

**边界：**本项是测量改良，不是独立生成机制；频闪不会自动增加相机的非重复事件采样率。只有硬件触发兼容时才能同步，不假定当前相机已接线。

**出处：**[Strobe-enhanced microscopy stage：作者项目说明](https://wenzel-lab.github.io/strobe-enhanced-microscopy-stage/)；[GitHub](https://github.com/wenzel-lab/strobe-enhanced-microscopy-stage)。关联硬件应用论文：*Plasmid Stability Analysis with Open-Source Droplet Microfluidics*, JoVE, 2024，[DOI](https://doi.org/10.3791/67659)，关联由 [Rio 仓库](https://github.com/wenzel-lab/rio-controller) 明确给出；不将此应用论文冒充频闪测量性能专论。

### 16. 可替换振动毛细管模块，扩展到无芯片生成【C】

**文献已做：**振动尖端毛细管可生成用于数字 PCR 的液滴；另有 3D 打印压电按需发生器及开源搭建资源。

**本项目可做：**把视觉质量评价平台用于不同发生器，对比固定振动参数下的尺寸分布与可重复性；可编程版本需新增有明确接口的驱动器。

**最小实验：**先在独立模块验证可见液滴，标定体积与生成频率，再讨论与十字芯片共用控制器。

**边界：**自由喷射/落滴、油中振动毛细管和封闭微流道是不同流动体系；不能直接复用尺寸模型或宣称 DropGen 已适配油水芯片。

**文献：**He et al., *A portable droplet generation system for ultra-wide dynamic range digital PCR based on a vibrating sharp-tip capillary*, Biosensors and Bioelectronics, 2021, 191:113458，[DOI](https://doi.org/10.1016/j.bios.2021.113458)；Ionkin & Harris, *Note: A versatile 3D-printed droplet-on-demand generator*, Review of Scientific Instruments, 2018, 89:116103，[DOI](https://doi.org/10.1063/1.5054400)。[DropGen 图纸与代码](https://github.com/harrislab-brown/DropGen)。

### 17. 大喷口＋气泡触发，研究高黏样品和抗堵塞【C】

**文献已做：**气泡触发可加速某些生物/聚合物流体的乳化，并在较大流道中产生较小液滴。

**本项目可做：**面向高黏或颗粒样品建立新模块，用图像评价气泡触发、滴径、卫星滴和连续运行时间；比较大通道是否减少堵塞停机。

**最小实验：**先在非生物模型液中验证，再考察气泡分离与后续分选兼容性。

**边界：**额外气相是原理的一部分，需要独立流路；不能把现有系统里的偶发气泡视作可控触发器。此路线成本与工艺改变较大。

**文献：**Yan et al., *Rapid Encapsulation of Cell and Polymer Solutions with Bubble-Triggered Droplet Generation*, Macromolecular Chemistry and Physics, 2017（2016 上线），[DOI](https://doi.org/10.1002/macp.201600297)。

### 18. 异常识别＋受约束恢复，建立稳定运行能力【A/B】

**文献已做：**自主生成系统利用图像质量/形态判断剔除失败数据并重新实验；图像反馈系统已能闭环调尺寸。

**本项目可做：**区分“照明/失焦导致不可测”“真实生成状态改变”“疑似液路异常”，分别采取重采集、返回已知稳定工况或停机。以异常段标注、误恢复率、恢复时间和废滴数评价。

**最小实验：**先用已有录像和离线光照变化验证分类，再做受控小流量变化。压力传感器可帮助区分液压与视觉问题；仅凭图像不能可靠认定堵塞或润湿失效的根因。

**边界：**这里的多类诊断与恢复策略是拟议扩展，引用文献不证明完整“自愈系统”已经实现；不将故障样本简单删除后只展示成功率。

**文献：**Cho et al., *Autonomous Bayesian Optimization-Based Control System for Droplet Generation*, Small Methods, 2025，[DOI](https://doi.org/10.1002/smtd.202500984)；Crawford et al., *Image-based closed-loop feedback for highly mono-dispersed microdroplet production*, Scientific Reports, 2017，[DOI](https://doi.org/10.1038/s41598-017-11254-5)。

## 三、GitHub：实际看到了什么，能借用到哪一步

本轮核查仓库主页、README、目录或论文关联，未安装、运行、下载大数据集或验证设备兼容性。以下均为复用候选，不是开箱即用承诺。许可证按页面明确可见内容记录，未确认的不得推断可任意复制。

| 项目 | 查见的资源 | 对 MCS_QT 的用途与边界 | 许可/核查备注 |
|---|---|---|---|
| [PV-Lab/ML-Multiscale-Droplets](https://github.com/PV-Lab/ML-Multiscale-Droplets) | `bo.py`、`segmentation.py`、`loss.py`、示例 notebook；有论文 DOI | 参考图像质量损失和实验寻优；项目已有 BO，避免重复移植 | 有 LICENSE，条款尚未逐项核查 |
| [CIDARLAB/DAFD](https://github.com/CIDARLAB/DAFD) | 流动聚焦设计/性能预测项目 | 稳态模型与初始工况推荐；不代替动态辨识 | 本轮未确认许可证条款 |
| [karl-gardner/droplet_detection](https://github.com/karl-gardner/droplet_detection) | YOLOv3/v5 notebook、数据链接、论文关联 | 内容物统计与包封评价 | 外部数据链接未下载；许可未确认 |
| [banag0/BYGDrop](https://github.com/banag0/BYGDrop) | 权重、图像处理脚本、论文引用 | 合成增强与离线鲁棒性基准 | 页面显示 GPL-3.0；原模型面向球形滴 |
| [wenzel-lab/rio-controller](https://github.com/wenzel-lab/rio-controller) | 压力、温度、成像等模块及软件，关联 JoVE 论文 | 新增独立可编程附件的工程参考 | 硬件/相关目录 CERN-OHL-W-2.0，软件 GPL-3.0；wish list 不计为已实现 |
| [wenzel-lab/strobe-enhanced-microscopy-stage](https://github.com/wenzel-lab/strobe-enhanced-microscopy-stage) | 搭建说明与频闪成像硬件资源 | 改善运动模糊，保留本项目相机需另适配 | 未运行；具体许可复用前核查 |
| [harrislab-brown/DropGen](https://github.com/harrislab-brown/DropGen) | STEP/STL、PCB、ESP32-S3 控制及 Wiki；关联 2015/2018 论文 | 独立压电发生器原型和模块化设计参考 | 页面显示 CC-BY-SA-4.0；不等于现有微流道附件 |
| [vsariola/acoustofluidic-control](https://github.com/vsariola/acoustofluidic-control) | MATLAB 控制、实验脚本、测试；多篇声学操纵论文 | 在线辨识/自适应策略参考 | MIT；主要为操纵/分选，不当作油水生成闭环 |
| [wenzel-lab/droplet_AInalysis](https://github.com/wenzel-lab/droplet_AInalysis) | YOLOv8 静态、视频、摄像头分析与权重 | 作为离线检测对照 | 本轮未确认配套论文及许可，不赋予论文级效果证据 |
| [GaudiLabs/OpenDrop](https://github.com/GaudiLabs/OpenDrop) | 电润湿数字微流控设计与软件 | 电极阵列/模块化控制启发 | GPL-3.0；开放表面数字微流控，不是连续十字发生器 |

额外线索：2023 开源气压泵论文链接到 [thechanglab](https://github.com/thechanglab) 组织页，本轮没有定位到确切配套仓库，因此不列为已经找到可运行源码。µPump 的公开材料主要在论文补充附件，不编造 GitHub 地址。

## 四、给本项目的选择建议

### 近期：可以利用现有条件先做出证据

1. **方法 5：管路/缓冲设计与视觉控制共同优化。** 有可换结构，有明确物理机制，主要比较均匀度与响应的折中；与单纯换算法区分明显。
2. **方法 1+2：质量约束下的跨芯片自动标定。** 复用现有 BO、数据管理和图像模块，研究更换芯片后减少调参和失败试验。
3. **方法 4+18：生成口预警与异常恢复。** 先离线验证是否有足够可观测信息和提前量，再决定是否接入控制。
4. **方法 10+12：从尺寸合格转向包封与分选产量合格。** 与已有磁珠识别、DEP 分选经验联系紧密，但检测准确性必须先验证。

### 中期：真正增加可控制的硬件变量

- **方法 7：电机位移控制柔性生成段。** 增加位移自由度，图像测结构和液滴；首轮可先手调证明因果关系。
- **方法 6+9：压力/微阀辅助按需生成。** 适合逐事件触发，但需要独立硬件与时间同步，不能用慢速泵冒充快速阀。
- **方法 11：视觉计量的二次分滴。** 适合按需体积和操作灵活性，需接受较大结构改造。

### 长期与另建平台

可调阶梯、振动毛细管、大通道气泡触发，以及旧文档里的液态金属电磁辅助腔，均保留为储备。它们提供不同适用范围，不承诺都可直接加装到现有芯片。

## 五、所有路线共用的验收要求

- 标清“图像监测/人工调节”“图像闭环调泵”“图像闭环调新增执行器”；三者不能混称。
- 至少对照现有方案、相同结构的固定设置、带反馈设置，区分结构收益与算法收益。
- 同时报告尺寸/体积误差、CV、频率与计数有效性、过渡废滴、持续运行和实验次数。面积、直径、体积的 CV 不混用。
- 以不同日期/芯片/批次作独立重复；测量误差要小于声称的改进。未解析的卫星滴或颗粒标为不可判定。
- 保留守恒条件：稳态无旁路、无合并/分裂且全部分散相进入统计液滴时，`Qd = f × 平均滴体积`。
- 图像频率、决策频率、执行器带宽与液路总延迟分别测量；软件换线程不能消除物理延迟。
- 可复用代码先离线验证和审查许可；未运行的仓库只算资源线索。真实设备控制经编排层统一管理。

本轮只新增和更新调研文档，未改控制代码、未连接设备。后续创新主张仍需对具体设计继续检索；“多加一个模块”“使用 AI”“使用 PID/MPC”均不自动构成新颖性。
