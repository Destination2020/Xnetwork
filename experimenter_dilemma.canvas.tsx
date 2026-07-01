import { Stack, H2 } from 'qoder/canvas';

export default function ExperimenterDilemma() {
  const W = 860;
  const H = 520;

  const C = {
    purple: '#4527A0',
    purpleLight: '#EDE7F6',
    blue: '#1565C0',
    blueLight: '#E3F2FD',
    green: '#2E7D32',
    greenLight: '#E8F5E9',
    orange: '#E65100',
    orangeLight: '#FFF3E0',
    red: '#C62828',
    redLight: '#FFEBEE',
    gray: '#616161',
    dark: '#212121',
    arrow: '#546E7A',
    bg: '#FAFAFA',
  };

  return (
    <Stack gap={10} style={{ padding: 16 }}>
      <H2>实验人员真正面临的困境</H2>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: C.bg, borderRadius: 10, border: `2px solid ${C.purple}` }}>
        <defs>
          <marker id="arrow" markerWidth="6" markerHeight="5" refX="5.5" refY="2.5" orient="auto">
            <polygon points="0 0, 6 2.5, 0 5" fill={C.arrow} />
          </marker>
          <marker id="arrowPurple" markerWidth="6" markerHeight="5" refX="5.5" refY="2.5" orient="auto">
            <polygon points="0 0, 6 2.5, 0 5" fill={C.purple} />
          </marker>
        </defs>

        {/* ===== TOP: Target XRR Curve ===== */}
        <rect x={W / 2 - 180} y={24} width={360} height={56} rx={10} fill={C.purpleLight} stroke={C.purple} strokeWidth={2.5} />
        <text x={W / 2} y={45} textAnchor="middle" fontSize={14} fontWeight="700" fill={C.purple}>
          实验目标
        </text>
        <text x={W / 2} y={65} textAnchor="middle" fontSize={13} fontWeight="600" fill={C.dark}>
          制备具有特定 XRR 响应的薄膜器件
        </text>

        {/* Arrow down */}
        <line x1={W / 2} y1={80} x2={W / 2} y2={108} stroke={C.arrow} strokeWidth={2} markerEnd="url(#arrow)" />

        {/* ===== XRR Curve Visualization (mini) ===== */}
        <rect x={W / 2 - 160} y={112} width={320} height={80} rx={8} fill="#FFFFFF" stroke={C.gray} strokeWidth={1.5} />
        <text x={W / 2 - 148} y={130} fontSize={10} fontWeight="600" fill={C.gray}>目标 XRR 曲线</text>
        {/* Simulated XRR curve */}
        <polyline
          points={`
            ${W/2-140},140 ${W/2-125},140 ${W/2-110},141 ${W/2-95},143
            ${W/2-80},148 ${W/2-65},156 ${W/2-50},167 ${W/2-35},175
            ${W/2-20},178 ${W/2-5},174 ${W/2+10},168 ${W/2+25},170
            ${W/2+40},176 ${W/2+55},179 ${W/2+70},175 ${W/2+85},172
            ${W/2+100},176 ${W/2+115},180 ${W/2+130},183 ${W/2+140},185
          `}
          fill="none" stroke={C.purple} strokeWidth={2.5}
        />
        <text x={W / 2 + 148} y={185} textAnchor="end" fontSize={9} fill={C.gray}>θ (°)</text>
        <text x={W / 2 - 148} y={185} fontSize={9} fill={C.gray}>R</text>

        {/* Arrow down */}
        <line x1={W / 2} y1={196} x2={W / 2} y2={226} stroke={C.arrow} strokeWidth={2} markerEnd="url(#arrow)" />

        {/* ===== Core Question ===== */}
        <rect x={W / 2 - 200} y={230} width={400} height={48} rx={24} fill={C.purple} />
        <text x={W / 2} y={254} textAnchor="middle" dominantBaseline="middle" fontSize={15} fontWeight="700" fill="#FFFFFF">
          "哪些材料组合能实现它？"
        </text>

        {/* Three arrows branching out */}
        {/* Left branch */}
        <path d={`M${W/2-80} 278 L${W/2-80} 300 L${130} 300 L${130} 328`}
          fill="none" stroke={C.arrow} strokeWidth={2} markerEnd="url(#arrow)" />
        {/* Center branch */}
        <path d={`M${W/2} 278 L${W/2} 328`}
          fill="none" stroke={C.arrow} strokeWidth={2} markerEnd="url(#arrow)" />
        {/* Right branch */}
        <path d={`M${W/2+80} 278 L${W/2+80} 300 L${W-130} 300 L${W-130} 328`}
          fill="none" stroke={C.arrow} strokeWidth={2} markerEnd="url(#arrow)" />

        {/* ===== Solution A (Green) ===== */}
        <rect x={30} y={332} width={200} height={160} rx={10} fill={C.greenLight} stroke={C.green} strokeWidth={2} />
        <rect x={30} y={332} width={200} height={36} rx={10} fill={C.green} />
        <rect x={30} y={350} width={200} height={18} fill={C.green} />
        <text x={130} y={354} textAnchor="middle" dominantBaseline="middle" fontSize={13} fontWeight="700" fill="#FFFFFF">方案 A</text>
        <text x={130} y={390} textAnchor="middle" fontSize={12} fontWeight="700" fill={C.dark}>Au / Ti / Si</text>
        <line x1={50} y1={404} x2={210} y2={404} stroke={C.green} strokeWidth={0.8} />
        <text x={52} y={424} fontSize={11} fill={C.dark}>✓ 拟合精度高</text>
        <text x={52} y={444} fontSize={11} fill={C.dark}>✓ 工艺成熟</text>
        <text x={52} y={464} fontSize={11} fill={C.red}>✗ Au 价格昂贵</text>
        <text x={52} y={484} fontSize={11} fill={C.red}>✗ 大面积制备困难</text>

        {/* ===== Solution B (Blue) ===== */}
        <rect x={W/2-100} y={332} width={200} height={160} rx={10} fill={C.blueLight} stroke={C.blue} strokeWidth={2} />
        <rect x={W/2-100} y={332} width={200} height={36} rx={10} fill={C.blue} />
        <rect x={W/2-100} y={350} width={200} height={18} fill={C.blue} />
        <text x={W/2} y={354} textAnchor="middle" dominantBaseline="middle" fontSize={13} fontWeight="700" fill="#FFFFFF">方案 B</text>
        <text x={W/2} y={390} textAnchor="middle" fontSize={12} fontWeight="700" fill={C.dark}>Pt / Cr / Si</text>
        <line x1={W/2-80} y1={404} x2={W/2+80} y2={404} stroke={C.blue} strokeWidth={0.8} />
        <text x={W/2-78} y={424} fontSize={11} fill={C.dark}>✓ 成本可控</text>
        <text x={W/2-78} y={444} fontSize={11} fill={C.dark}>✓ 材料易获取</text>
        <text x={W/2-78} y={464} fontSize={11} fill={C.red}>✗ Pt 沉积温度高</text>
        <text x={W/2-78} y={484} fontSize={11} fill={C.red}>✗ 需额外退火步骤</text>

        {/* ===== Solution C (Orange) ===== */}
        <rect x={W-230} y={332} width={200} height={160} rx={10} fill={C.orangeLight} stroke={C.orange} strokeWidth={2} />
        <rect x={W-230} y={332} width={200} height={36} rx={10} fill={C.orange} />
        <rect x={W-230} y={350} width={200} height={18} fill={C.orange} />
        <text x={W-130} y={354} textAnchor="middle" dominantBaseline="middle" fontSize={13} fontWeight="700" fill="#FFFFFF">方案 C</text>
        <text x={W-130} y={390} textAnchor="middle" fontSize={12} fontWeight="700" fill={C.dark}>Mo / SiO₂ / Si</text>
        <line x1={W-210} y1={404} x2={W-50} y2={404} stroke={C.orange} strokeWidth={0.8} />
        <text x={W-208} y={424} fontSize={11} fill={C.dark}>✓ 半导体工艺兼容</text>
        <text x={W-208} y={444} fontSize={11} fill={C.dark}>✓ 成本最低</text>
        <text x={W-208} y={464} fontSize={11} fill={C.red}>✗ 需增加缓冲层</text>
        <text x={W-208} y={484} fontSize={11} fill={C.red}>✗ 层数更多</text>

      </svg>
    </Stack>
  );
}
