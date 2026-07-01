import { Stack, H2, Text } from 'qoder/canvas';

export default function InnovationTable() {
  const rows = [
    {
      id: '1',
      title: 'Ring Topology PSO\n多峰候选区域自适应搜索',
      diff: '传统优化只找全局最优；Ring PSO 自然分离多个局部极小区域（Li 2010, HRTPSO 2020）',
    },
    {
      id: '2',
      title: '条件神经网络\n以局部空间为输入',
      diff: '不是全局反演，而是在小生境约束下快速采样，降低搜索难度',
    },
    {
      id: '3',
      title: '先验泄漏防护\n训练策略',
      diff: '先采样 box 再采样参数，避免网络学到"先验暗示答案"的捷径',
    },
    {
      id: '4',
      title: '物理正向模型\n做验证兜底',
      diff: '神经网络提供候选解，传递矩阵法做物理一致性验证，不依赖纯黑箱',
    },
    {
      id: '5',
      title: '以制备为导向的\n多解输出',
      diff: '不是给出不确定性分布，而是给出离散的、可操作的、对应具体材料组合的方案集',
    },
  ];

  const C = {
    headerBg: '#311B92',       // deep blue-purple
    headerText: '#FFFFFF',
    rowAlt: '#EDE7F6',         // light purple
    rowWhite: '#FFFFFF',
    borderColor: '#D1C4E9',    // soft purple border
    idBg: '#B39DDB',           // medium purple circle
    idText: '#311B92',
    titleColor: '#1A237E',     // dark blue
    diffColor: '#37474F',
  };

  const SVG_W = 900;
  const COL1_W = 220;   // id + title column
  const COL2_X = COL1_W;
  const COL2_W = SVG_W - COL2_X;
  const ROW_H = 62;
  const HEADER_H = 48;
  const SVG_H = HEADER_H + rows.length * ROW_H + 4;

  return (
    <Stack gap={12} style={{ padding: 16 }}>
      <H2>创新点总结</H2>
      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', background: '#FFFFFF', borderRadius: 10, border: `2px solid ${C.headerBg}` }}>
        <defs>
          <style>{`
            .th { font-weight: 700; font-size: 14px; fill: ${C.headerText}; }
            .td-id { font-weight: 800; font-size: 15px; fill: ${C.idText}; }
            .td-title { font-weight: 700; font-size: 12px; fill: ${C.titleColor}; }
            .td-diff { font-weight: 400; font-size: 12px; fill: ${C.diffColor}; }
          `}</style>
        </defs>

        {/* ===== HEADER ===== */}
        <rect x={0} y={0} width={SVG_W} height={HEADER_H} fill={C.headerBg} rx={8} ry={8} />
        <rect x={0} y={20} width={SVG_W} height={HEADER_H - 20} fill={C.headerBg} />
        <line x1={COL1_W} y1={6} x2={COL1_W} y2={HEADER_H - 4} stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} />
        <text x={COL1_W / 2} y={HEADER_H / 2} textAnchor="middle" dominantBaseline="middle" className="th">创新点</text>
        <text x={COL2_X + COL2_W / 2} y={HEADER_H / 2} textAnchor="middle" dominantBaseline="middle" className="th">与现有工作的差异化</text>

        {/* ===== DATA ROWS ===== */}
        {rows.map((row, i) => {
          const rowY = HEADER_H + i * ROW_H;
          const bg = i % 2 === 0 ? C.rowWhite : C.rowAlt;
          const midY = rowY + ROW_H / 2;
          const titleLines = row.title.split('\n');

          // Wrap diff text into two lines if long
          const diffText = row.diff;
          const mid = Math.ceil(diffText.length / 2);
          let diffLines = [diffText];
          if (diffText.length > 38) {
            // Find a good break point near the middle
            const breakAt = diffText.lastIndexOf('，', mid) > mid - 10
              ? diffText.lastIndexOf('，', mid) + 1
              : diffText.lastIndexOf(' ', mid) > mid - 10
                ? diffText.lastIndexOf(' ', mid) + 1
                : mid;
            diffLines = [diffText.slice(0, breakAt).trim(), diffText.slice(breakAt).trim()];
          }

          return (
            <g key={row.id}>
              {/* Row bg */}
              <rect x={0} y={rowY} width={SVG_W} height={ROW_H} fill={bg} />

              {/* Bottom border */}
              <line x1={0} y1={rowY + ROW_H} x2={SVG_W} y2={rowY + ROW_H} stroke={C.borderColor} strokeWidth={1} />

              {/* Column divider */}
              <line x1={COL1_W} y1={rowY} x2={COL1_W} y2={rowY + ROW_H} stroke={C.borderColor} strokeWidth={1} />

              {/* ID circle */}
              <circle cx={30} cy={midY} r={14} fill={C.idBg} />
              <text x={30} y={midY} textAnchor="middle" dominantBaseline="middle" className="td-id">{row.id}</text>

              {/* Title (multi-line) */}
              {titleLines.map((line, li) => (
                <text
                  key={li}
                  x={54}
                  y={midY - ((titleLines.length - 1) * 8) + li * 16}
                  dominantBaseline="middle"
                  className="td-title"
                >
                  {line}
                </text>
              ))}

              {/* Diff text (wrapped) */}
              {diffLines.length === 1 ? (
                <text x={COL2_X + 16} y={midY} dominantBaseline="middle" className="td-diff">{diffLines[0]}</text>
              ) : (
                diffLines.map((line, li) => (
                  <text
                    key={li}
                    x={COL2_X + 16}
                    y={midY - ((diffLines.length - 1) * 8) + li * 16}
                    dominantBaseline="middle"
                    className="td-diff"
                  >
                    {line}
                  </text>
                ))
              )}
            </g>
          );
        })}
      </svg>
    </Stack>
  );
}
