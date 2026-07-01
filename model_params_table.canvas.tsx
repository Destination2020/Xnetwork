import { Stack, H2 } from 'qoder/canvas';

export default function ModelTable() {
  const headers = ['Layer', 'Thickness (nm)', 'Roughness (nm)', 'Re(SLD) (Å⁻²)', 'Im(SLD) (Å⁻²)'];
  
  const rows = [
    { name: 'Air',               values: ['/',      '0.65',  '0',       '0'       ], bold: false },
    { name: 'TiO₂',              values: ['9.7',    '2.739', '4.1×10⁻³','1.73×10⁻⁵'], bold: true  },
    { name: 'Sr₀.₇₅Ba₀.₂₅Nb₂O₆', values: ['8.976',  '0.1',   '4.1×10⁻³','1.54×10⁻⁵'], bold: true  },
    { name: 'Si (Substrate)',    values: ['/',      '0.0',   '2.01×10⁻³','4.57×10⁻⁵'], bold: false },
  ];

  const C = {
    headerBg: '#00695C',       // teal
    headerText: '#FFFFFF',
    subHeaderBg: '#00897B',
    rowAlt1: '#F5F5F5',        // light gray
    rowAlt2: '#EDE7F6',        // light purple
    borderColor: '#B2DFDB',
    nameColor: '#212121',
    valColor: '#37474F',
    sampleColor: '#004D40',
  };

  const SVG_W = 820;
  const COL_W = [180, 130, 130, 140, 140];
  const HEADER_H = 40;
  const SUB_H = 38;
  const ROW_H = 48;
  const SVG_H = HEADER_H + SUB_H + rows.length * ROW_H + 8;

  return (
    <Stack gap={12} style={{ padding: 16 }}>
      <H2>薄膜结构参数表</H2>
      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', background: '#FFFFFF', borderRadius: 10, border: `2px solid ${C.headerBg}` }}>
        <defs>
          <style>{`
            .sample { font-weight: 700; font-size: 15px; fill: ${C.headerText}; }
            .th { font-weight: 700; font-size: 12px; fill: ${C.headerText}; }
            .td-name { font-weight: 400; font-size: 12px; fill: ${C.nameColor}; }
            .td-name-bold { font-weight: 700; font-size: 12px; fill: ${C.nameColor}; }
            .td-val { font-weight: 400; font-size: 12px; fill: ${C.valColor}; font-family: 'Consolas', 'Courier New', monospace; }
          `}</style>
        </defs>

        {/* ===== MAIN HEADER: Sample Name ===== */}
        <rect x={0} y={0} width={SVG_W} height={HEADER_H} fill={C.headerBg} rx={8} ry={8} />
        <rect x={0} y={16} width={SVG_W} height={HEADER_H - 16} fill={C.headerBg} />
        <text x={SVG_W / 2} y={HEADER_H / 2} textAnchor="middle" dominantBaseline="middle" className="sample">
          Air / TiO₂ / Sr₀.₇₅Ba₀.₂₅Nb₂O₆ / Si
        </text>

        {/* ===== SUB HEADER: Column Names ===== */}
        <rect x={0} y={HEADER_H} width={SVG_W} height={SUB_H} fill={C.subHeaderBg} />
        {headers.map((h, i) => {
          const x = COL_W.slice(0, i).reduce((a, b) => a + b, 0) + COL_W[i] / 2;
          return (
            <g key={i}>
              <text x={x} y={HEADER_H + SUB_H / 2} textAnchor="middle" dominantBaseline="middle" className="th">{h}</text>
              {i > 0 && (
                <line
                  x1={COL_W.slice(0, i).reduce((a, b) => a + b, 0)}
                  y1={HEADER_H + 4}
                  x2={COL_W.slice(0, i).reduce((a, b) => a + b, 0)}
                  y2={HEADER_H + SUB_H - 4}
                  stroke="rgba(255,255,255,0.25)" strokeWidth={1}
                />
              )}
            </g>
          );
        })}

        {/* ===== DATA ROWS ===== */}
        {rows.map((row, i) => {
          const rowY = HEADER_H + SUB_H + i * ROW_H;
          const bg = i % 2 === 0 ? C.rowAlt1 : C.rowAlt2;
          const midY = rowY + ROW_H / 2;

          return (
            <g key={i}>
              {/* Row background */}
              <rect x={0} y={rowY} width={SVG_W} height={ROW_H} fill={bg} />

              {/* Bottom border */}
              <line x1={0} y1={rowY + ROW_H} x2={SVG_W} y2={rowY + ROW_H} stroke={C.borderColor} strokeWidth={1} />

              {/* Column dividers */}
              {COL_W.slice(0, -1).map((_, ci) => {
                const x = COL_W.slice(0, ci + 1).reduce((a, b) => a + b, 0);
                return <line key={ci} x1={x} y1={rowY} x2={x} y2={rowY + ROW_H} stroke={C.borderColor} strokeWidth={0.5} />;
              })}

              {/* Layer name */}
              <text
                x={20}
                y={midY}
                dominantBaseline="middle"
                className={row.bold ? 'td-name-bold' : 'td-name'}
              >
                {row.name}
              </text>

              {/* Values */}
              {row.values.map((val, vi) => {
                const x = COL_W.slice(0, vi + 1).reduce((a, b) => a + b, 0) + COL_W[vi + 1] / 2;
                return (
                  <text key={vi} x={x} y={midY} textAnchor="middle" dominantBaseline="middle" className="td-val">
                    {val}
                  </text>
                );
              })}
            </g>
          );
        })}
      </svg>
    </Stack>
  );
}
