import { Stack, H2, Text } from 'qoder/canvas';

export default function XRRFlowchart() {
  const W = 1280;
  const H = 990;

  // Colors
  const c1bg = '#E8F5E9'; // stage1 green bg
  const c1bdr = '#66BB6A';
  const c1accent = '#2E7D32';
  const c2bg = '#E3F2FD'; // stage2 blue bg
  const c2bdr = '#42A5F5';
  const c2accent = '#1565C0';
  const c3bg = '#FFF3E0'; // stage3 orange bg
  const c3bdr = '#FFA726';
  const c3accent = '#E65100';
  const nodeFill = '#FFFFFF';
  const nodeStroke = '#455A64';
  const arrowColor = '#546E7A';
  const txtColor = '#212121';
  const subColor = '#616161';
  const loopBg = '#FFFDE7';
  const loopBorder = '#FDD835';

  const boxW = 260;
  const boxH = 44;
  const rx = 8;

  // helper: rounded rect node
  const Box = ({ x, y, w, h, fill, stroke, text, sub, bold, fontSize }: any) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={rx} fill={fill || nodeFill} stroke={stroke || nodeStroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - (sub ? 5 : 0)} textAnchor="middle" dominantBaseline="middle"
        fontSize={fontSize || 13} fontWeight={bold ? '700' : '500'} fill={txtColor} fontFamily="system-ui,sans-serif">
        {text}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" dominantBaseline="middle"
          fontSize={11} fill={subColor} fontFamily="system-ui,sans-serif">
          {sub}
        </text>
      )}
    </g>
  );

  // helper: diamond
  const Diamond = ({ cx, cy, w, h, text, fill, stroke }: any) => (
    <g>
      <polygon
        points={`${cx},${cy - h / 2} ${cx + w / 2},${cy} ${cx},${cy + h / 2} ${cx - w / 2},${cy}`}
        fill={fill || nodeFill} stroke={stroke || nodeStroke} strokeWidth={1.5}
      />
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle"
        fontSize={11} fontWeight="500" fill={txtColor} fontFamily="system-ui,sans-serif">
        {text}
      </text>
    </g>
  );

  // helper: arrow line
  const Arrow = ({ x1, y1, x2, y2, label, labelOff }: any) => {
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    return (
      <g>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={arrowColor} strokeWidth={1.8}
          markerEnd="url(#arrowhead)" />
        {label && (
          <text x={mx + (labelOff?.x || 0)} y={my + (labelOff?.y || -8)}
            textAnchor="middle" fontSize={10} fill={c1accent} fontWeight="600" fontFamily="system-ui,sans-serif">
            {label}
          </text>
        )}
      </g>
    );
  };

  // helper: curved arrow
  const CurveArrow = ({ path, label }: any) => (
    <g>
      <path d={path} fill="none" stroke={arrowColor} strokeWidth={1.8} markerEnd="url(#arrowhead)" />
      {label && (
        <text x={0} y={0} fontSize={10} fill={c1accent} fontWeight="600" fontFamily="system-ui,sans-serif">
          {label}
        </text>
      )}
    </g>
  );

  // helper: stage label
  const StageLabel = ({ x, y, h, text, color }: any) => (
    <g>
      <rect x={x} y={y} width={26} height={h} rx={4} fill={color} />
      <text x={x + 13} y={y + h / 2} textAnchor="middle" dominantBaseline="middle"
        transform={`rotate(-90,${x + 13},${y + h / 2})`}
        fontSize={11} fontWeight="700" fill="#FFFFFF" fontFamily="system-ui,sans-serif">
        {text}
      </text>
    </g>
  );

  const CX = W / 2 + 20; // main column center x

  return (
    <Stack gap={8}>
      <H2>XRR 多解反演算法流程图</H2>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: '#FAFAFA', borderRadius: 8, fontFamily: 'system-ui,sans-serif' }}>
        <defs>
          <marker id="arrowhead" markerWidth="5" markerHeight="4" refX="4.5" refY="2" orient="auto">
            <polygon points="0 0, 5 2, 0 4" fill={arrowColor} />
          </marker>
        </defs>

        {/* ===== STAGE 1 ===== */}
        <rect x={40} y={12} width={W - 80} height={372} rx={10} fill={c1bg} opacity={0.6} />
        <StageLabel x={52} y={16} h={364} text="第一阶段" color={c1accent} />

        {/* Title */}
        <text x={CX} y={36} textAnchor="middle" fontSize={14} fontWeight="700" fill={c1accent}>
          第一阶段：Ring Topology PSO 自适应候选区域搜索
        </text>

        {/* Start */}
        <Box x={CX - boxW / 2} y={48} w={boxW} h={38} fill="#C8E6C9" stroke={c1accent} text="给定 XRR 实验曲线 R_obs(q)" bold />
        <Arrow x1={CX} y1={86} x2={CX} y2={96} />

        {/* B_global */}
        <Box x={CX - boxW / 2} y={96} w={boxW} h={38} text="设定全局参数范围 B_global" />
        <Arrow x1={CX} y1={134} x2={CX} y2={144} />

        {/* Init particles */}
        <Box x={CX - boxW / 2} y={144} w={boxW} h={38} text="初始化 N_p 个粒子（归一化空间）" />
        <Arrow x1={CX} y1={182} x2={CX} y2={196} />

        {/* Loop box */}
        <rect x={CX - 175} y={196} width={350} height={130} rx={10} fill={loopBg} stroke={loopBorder} strokeWidth={2} />
        <text x={CX - 170} y={212} fontSize={11} fontWeight="700" fill="#F9A825">PSO 迭代循环</text>

        {/* Loop steps */}
        {[
          '计算适应度：f(θ) = MSE(log R_sim, log R_obs)',
          'Ring 邻域更新速度（lbest 拓扑）',
          '位置更新 + 越界反射',
          '惯性权重线性衰减 w: 0.9 → 0.4',
        ].map((t, i) => (
          <g key={i}>
            <text x={CX - 155} y={230 + i * 25} fontSize={11} fill={txtColor}>{t}</text>
            {i < 3 && <line x1={CX - 160} y1={238 + i * 25} x2={CX + 165} y2={238 + i * 25} stroke="#E0E0E0" strokeWidth={0.5} />}
          </g>
        ))}

        {/* Loop feedback arrow */}
        <path d={`M${CX + 175} 260 L${CX + 215} 260 L${CX + 215} 200 L${CX + 180} 200`}
          fill="none" stroke={arrowColor} strokeWidth={1.5} markerEnd="url(#arrowhead)" />
        <text x={CX + 220} y={232} fontSize={10} fontWeight="600" fill="#F57F17">t &lt; T ?</text>

        <Arrow x1={CX} y1={326} x2={CX} y2={340} />

        {/* Extract candidates */}
        <Box x={CX - boxW / 2} y={340} w={boxW} h={38} text="提取低误差候选粒子 (f < ε_pso)" />
        <Arrow x1={CX} y1={378} x2={CX} y2={388} />

        {/* ===== CLUSTERING ROW (between stages) ===== */}
        <rect x={CX - boxW / 2} y={388} width={boxW} height={38} rx={rx} fill="#FFF9C4" stroke="#FDD835" strokeWidth={1.5} />
        <text x={CX} y={407} textAnchor="middle" dominantBaseline="middle" fontSize={12} fontWeight="600" fill={txtColor}>
          ① DBSCAN 聚类（粗粒度）
        </text>
        <Arrow x1={CX} y1={426} x2={CX} y2={436} />

        {/* ===== STAGE 2 ===== */}
        <rect x={40} y={436} width={W - 80} height={200} rx={10} fill={c2bg} opacity={0.6} />
        <StageLabel x={52} y={440} h={192} text="第二阶段" color={c2accent} />

        <text x={CX} y={458} textAnchor="middle" fontSize={14} fontWeight="700" fill={c2accent}>
          第二阶段：条件神经网络后验采样
        </text>

        {/* Parallel B_m boxes */}
        {['B₁', 'B₂', '…', 'B_M'].map((b, i) => {
          const bx = CX - 200 + i * 110;
          return (
            <g key={i}>
              <rect x={bx - 36} y={468} width={72} height={30} rx={6} fill="#BBDEFB" stroke={c2bdr} strokeWidth={1.5} />
              <text x={bx} y={483} textAnchor="middle" dominantBaseline="middle" fontSize={11} fontWeight="600" fill={c2accent}>{b}</text>
            </g>
          );
        })}
        <text x={CX + 180} y={484} fontSize={10} fontWeight="700" fill={c2accent}>← 可并行</text>

        <Arrow x1={CX} y1={498} x2={CX} y2={508} />

        {/* NN input */}
        <Box x={CX - 160} y={508} w={320} h={38} text="输入 (R_obs, B_m) → 条件神经网络" />
        <Arrow x1={CX} y1={546} x2={CX} y2={558} />

        {/* Posterior sampling */}
        <Box x={CX - 160} y={558} w={320} h={38} text="网络后验采样：p(θ | R_obs, B_m)" />
        <Arrow x1={CX} y1={596} x2={CX} y2={610} />

        {/* Candidates */}
        <rect x={CX - 160} y={610} width={320} height={38} rx={rx} fill="#BBDEFB" stroke={c2accent} strokeWidth={1.5} />
        <text x={CX} y={629} textAnchor="middle" dominantBaseline="middle" fontSize={12} fontWeight="600" fill={txtColor}>
          大量候选解 θ₁, θ₂, …, θ_K
        </text>
        <Arrow x1={CX} y1={648} x2={CX} y2={660} />

        {/* ===== STAGE 3 ===== */}
        <rect x={40} y={660} width={W - 80} height={280} rx={10} fill={c3bg} opacity={0.6} />
        <StageLabel x={52} y={664} h={272} text="第三阶段" color={c3accent} />

        <text x={CX} y={682} textAnchor="middle" fontSize={14} fontWeight="700" fill={c3accent}>
          第三阶段：物理正向模型验证与局部优化微调
        </text>

        {/* Forward model */}
        <Box x={CX - 160} y={694} w={320} h={38} text="XRR 传递矩阵法正演验证" />
        <Arrow x1={CX} y1={732} x2={CX} y2={744} />

        {/* Diamond 1: R-factor < eps_pass */}
        <Diamond cx={CX} cy={762} w={200} h={36} text="R-factor < ε_pass ?" />

        {/* Yes → keep */}
        <Arrow x1={CX - 100} y1={762} x2={CX - 160} y2={762} label="是" labelOff={{ x: -14, y: -8 }} />
        <Box x={CX - 260} y={744} w={80} h={36} fill="#C8E6C9" stroke={c1accent} text="保留" bold fontSize={11} />

        {/* No → second diamond */}
        <Arrow x1={CX + 100} y1={762} x2={CX + 160} y2={762} label="否" labelOff={{ x: 14, y: -8 }} />
        <Diamond cx={CX + 220} cy={762} w={180} h={36} text="R-factor < ε_reject ?" />

        {/* Yes from diamond 2 → fine-tune */}
        <path d={`M${CX + 220} ${762 + 18} L${CX + 220} ${812}`}
          fill="none" stroke={arrowColor} strokeWidth={1.8} markerEnd="url(#arrowhead)" />
        <text x={CX + 240} y={788} fontSize={10} fontWeight="600" fill={c3accent}>是</text>

        <Box x={CX + 140} y={812} w={160} h={38} text="局部优化微调" sub="LM / Nelder-Mead" />

        {/* Feedback from fine-tune back to forward model */}
        <path d={`M${CX + 300} ${831} L${CX + 340} ${831} L${CX + 340} ${713} L${CX + 165} ${713}`}
          fill="none" stroke={arrowColor} strokeWidth={1.5} strokeDasharray="5,3" markerEnd="url(#arrowhead)" />
        <text x={CX + 348} y={772} fontSize={9} fill={c3accent} fontWeight="600">微调后重新正演</text>

        {/* No from diamond 2 → discard */}
        <path d={`M${CX + 220 + 90} ${762} L${CX + 340} ${762}`}
          fill="none" stroke={arrowColor} strokeWidth={1.8} markerEnd="url(#arrowhead)" />
        <text x={CX + 268} y={748} fontSize={10} fontWeight="600" fill={c3accent}>否</text>
        <Box x={CX + 345} y={744} w={60} h={36} fill="#ECEFF1" stroke="#78909C" text="丢弃" bold fontSize={11} />

        {/* Merge arrows to clustering */}
        <path d={`M${CX - 220} ${780} L${CX - 220} ${885} L${CX - 165} ${885}`}
          fill="none" stroke={arrowColor} strokeWidth={1.8} markerEnd="url(#arrowhead)" />
        <path d={`M${CX + 220} ${850} L${CX + 220} ${866} L${CX + 160} ${866}`}
          fill="none" stroke={arrowColor} strokeWidth={1.8} markerEnd="url(#arrowhead)" />

        {/* Dedup clustering */}
        <rect x={CX - 160} y={866} width={320} height={38} rx={rx} fill="#FFF9C4" stroke="#FDD835" strokeWidth={1.5} />
        <text x={CX} y={885} textAnchor="middle" dominantBaseline="middle" fontSize={12} fontWeight="600" fill={txtColor}>
          ② DBSCAN 聚类去重（细粒度）→ 排序 + 置信度
        </text>
        <Arrow x1={CX} y1={904} x2={CX} y2={916} />

        {/* Final output */}
        <rect x={CX - 200} y={916} width={400} height={52} rx={10} fill="#C8E6C9" stroke={c1accent} strokeWidth={2.5} />
        <text x={CX} y={936} textAnchor="middle" dominantBaseline="middle" fontSize={13} fontWeight="700" fill={txtColor}>
          输出 N 个多解反演结果
        </text>
        <text x={CX} y={956} textAnchor="middle" dominantBaseline="middle" fontSize={10} fill={subColor}>
          参数 θ + 拟合曲线 R_sim + R-factor + 置信度 C_i
        </text>

        {/* Bottom annotation */}
        <text x={CX} y={H - 8} textAnchor="middle" fontSize={9} fill="#9E9E9E" fontStyle="italic">
          全局优化提供多峰候选区域 → 条件神经网络局部快速采样 → 物理正向模型验证与微调
        </text>
      </svg>
    </Stack>
  );
}
