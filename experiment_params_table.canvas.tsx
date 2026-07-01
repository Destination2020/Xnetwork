import { Stack, H2 } from 'qoder/canvas';

export default function ExperimentParamsTable() {
  const C = {
    headerBg: '#00695C',
    subHeaderBg: '#00897B',
    rowAlt1: '#F5F5F5',
    rowAlt2: '#EDE7F6',
    borderColor: '#B2DFDB',
    catBg: '#E0F2F1',
    txtColor: '#212121',
    valColor: '#37474F',
    headerText: '#FFFFFF',
  };

  const COL_W = [200, 220, 140];
  const SVG_W = COL_W.reduce((a, b) => a + b, 0);
  const HEADER_H = 40;
  const ROW_H = 44;

  const rows = [
    {
      cat: '常数背景噪声',
      items: [
        { param: '高角度尾部背景强度', desc: '背景强度水平', val: '7.0 × 10⁻⁷' },
        { param: '仪器分辨率展宽', desc: 'HWHM 展宽参数', val: '0.03°' },
        { param: '角度零点偏移', desc: '实验角度轴偏移', val: '0.000°' },
        { param: '强度缩放因子', desc: '实验强度归一化误差', val: '1.00' },
        { param: '随机噪声标准差', desc: '背景随机扰动幅度', val: '1.0 × 10⁻⁷' },
      ],
    },
  ];

  const totalRows = rows.reduce((s, g) => s + g.items.length, 0);
  const SVG_H = HEADER_H + totalRows * ROW_H + 4;

  return (
    <Stack gap={12} style={{ padding: 16 }}>
      <H2>实验模拟参数表</H2>
      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', maxWidth: 620, background: '#FFFFFF', borderRadius: 10, border: `2px solid ${C.headerBg}` }}>
        <defs>
          <style>{`
            .th { font-weight: 700; font-size: 13px; fill: ${C.headerText}; }
            .cat { font-weight: 700; font-size: 12px; fill: ${C.txtColor}; }
            .param { font-weight: 600; font-size: 12px; fill: ${C.txtColor}; }
            .desc { font-weight: 400; font-size: 11px; fill: #757575; }
            .val { font-weight: 600; font-size: 13px; fill: ${C.valColor}; font-family: 'Consolas','Courier New',monospace; }
          `}</style>
        </defs>

        {/* Header */}
        <rect x={0} y={0} width={SVG_W} height={HEADER_H} fill={C.headerBg} rx={8} ry={8} />
        <rect x={0} y={16} width={SVG_W} height={HEADER_H - 16} fill={C.headerBg} />
        {['参数类别', '参数名称', '数值'].map((h, i) => {
          const x = COL_W.slice(0, i).reduce((a, b) => a + b, 0) + COL_W[i] / 2;
          return (
            <g key={i}>
              <text x={x} y={HEADER_H / 2} textAnchor="middle" dominantBaseline="middle" className="th">{h}</text>
              {i > 0 && (
                <line
                  x1={COL_W.slice(0, i).reduce((a, b) => a + b, 0)}
                  y1={6}
                  x2={COL_W.slice(0, i).reduce((a, b) => a + b, 0)}
                  y2={HEADER_H - 6}
                  stroke="rgba(255,255,255,0.25)" strokeWidth={1}
                />
              )}
            </g>
          );
        })}

        {/* Data */}
        {(() => {
          let yOff = HEADER_H;
          const elements: any[] = [];

          rows.forEach((group, gi) => {
            const groupH = group.items.length * ROW_H;

            group.items.forEach((item, ii) => {
              const rowY = yOff + ii * ROW_H;
              const bg = ii % 2 === 0 ? C.rowAlt1 : C.rowAlt2;
              const midY = rowY + ROW_H / 2;

              elements.push(
                <g key={`${gi}-${ii}`}>
                  <rect x={0} y={rowY} width={SVG_W} height={ROW_H} fill={bg} />
                  <line x1={0} y1={rowY + ROW_H} x2={SVG_W} y2={rowY + ROW_H} stroke={C.borderColor} strokeWidth={1} />
                  {COL_W.slice(0, -1).map((_, ci) => {
                    const cx = COL_W.slice(0, ci + 1).reduce((a, b) => a + b, 0);
                    return <line key={ci} x1={cx} y1={rowY} x2={cx} y2={rowY + ROW_H} stroke={C.borderColor} strokeWidth={0.5} />;
                  })}
                  <text x={COL_W[0] + 14} y={midY - 7} dominantBaseline="middle" className="param">{item.param}</text>
                  <text x={COL_W[0] + 14} y={midY + 9} dominantBaseline="middle" className="desc">{item.desc}</text>
                  <text x={COL_W[0] + COL_W[1] + COL_W[2] / 2} y={midY} textAnchor="middle" dominantBaseline="middle" className="val">{item.val}</text>
                </g>
              );
            });

            // Category label spanning all rows of this group
            elements.push(
              <g key={`cat-${gi}`}>
                <rect x={1} y={yOff + 1} width={COL_W[0] - 2} height={groupH - 2} fill={C.catBg} rx={4} />
                <line x1={COL_W[0]} y1={yOff} x2={COL_W[0]} y2={yOff + groupH} stroke={C.borderColor} strokeWidth={1} />
                <text
                  x={COL_W[0] / 2}
                  y={yOff + groupH / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="cat"
                >
                  {group.cat}
                </text>
              </g>
            );

            yOff += groupH;
          });

          return elements;
        })()}
      </svg>
    </Stack>
  );
}
