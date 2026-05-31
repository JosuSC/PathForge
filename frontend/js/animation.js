/**
 * animation.js
 * ------------
 * Módulo Preview: canvas 2D de vista previa del grafo en Setup.
 * Renderiza nodos como estrellas con aristas curvas.
 *
 * Exporta globalmente: Preview
 */

const Preview = (() => {
    let cv, cx, W, H, hov = null;
    const pos = {};

    // Color palette (sync con graph3d.js)
    const PAL = [
        0x00c8ff, 0xb44dff, 0xffc843, 0xff6b35, 0x00ff8c, 0xff3355,
        0x4fc3f7, 0xce93d8, 0xffcc02, 0x69f0ae, 0xf48fb1, 0x80deea,
        0xff9800, 0x00bcd4, 0x9c27b0, 0x4caf50,
    ];
    const pc = i => '#' + PAL[i % PAL.length].toString(16).padStart(6, '0');

    // ── Init ──────────────────────────────────────────────────
    function init() {
        cv = document.getElementById('preview-canvas');
        cx = cv.getContext('2d');
        resize();
        cv.addEventListener('mousemove', _onHover);
        window.addEventListener('resize', () => { resize(); draw(); });
    }

    function resize() {
        const wrap = cv.parentElement;
        W = cv.width = wrap.clientWidth || 600;
        H = cv.height = wrap.clientHeight || 400;
    }

    // ── Layout: circular with Fibonacci jitter ────────────────
    function _layout() {
        const n = AppState.nodes.length;
        if (!n) return;
        const cx2 = W / 2, cy2 = H / 2;
        const r = Math.min(W, H) * 0.34;
        AppState.nodes.forEach((node, i) => {
            const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
            const jitter = Math.sin(i * 7.3 + 2) * 18;
            pos[node.id] = {
                x: cx2 + Math.cos(angle) * (r + jitter),
                y: cy2 + Math.sin(angle) * (r + jitter * 0.5),
                col: pc(i),
            };
        });
    }

    // ── Draw ──────────────────────────────────────────────────
    function draw() {
        if (!cx) return;
        cx.clearRect(0, 0, W, H);

        if (!AppState.nodes.length) {
            document.getElementById('preview-msg').style.display = 'flex';
            return;
        }
        document.getElementById('preview-msg').style.display = 'none';
        _layout();
        _drawEdges();
        _drawNodes();
    }

    function _drawEdges() {
        AppState.edges.forEach(e => {
            const a = pos[e.from_node], b = pos[e.to_node];
            if (!a || !b) return;

            // Quadratic bezier curve
            const mx = (a.x + b.x) / 2 + (b.y - a.y) * 0.15;
            const my = (a.y + b.y) / 2 - (b.x - a.x) * 0.15;

            cx.save();
            cx.globalAlpha = 0.35;
            cx.strokeStyle = 'rgba(0,200,255,0.55)';
            cx.lineWidth = 1.2;
            cx.setLineDash([4, 6]);
            cx.beginPath();
            cx.moveTo(a.x, a.y);
            cx.quadraticCurveTo(mx, my, b.x, b.y);
            cx.stroke();

            // Arrowhead
            const dx = b.x - mx, dy = b.y - my;
            const ang = Math.atan2(dy, dx);
            cx.globalAlpha = 0.5;
            cx.setLineDash([]);
            cx.fillStyle = 'rgba(0,200,255,0.6)';
            cx.beginPath();
            cx.moveTo(b.x, b.y);
            cx.lineTo(b.x - 8 * Math.cos(ang - 0.4), b.y - 8 * Math.sin(ang - 0.4));
            cx.lineTo(b.x - 8 * Math.cos(ang + 0.4), b.y - 8 * Math.sin(ang + 0.4));
            cx.closePath();
            cx.fill();
            cx.restore();
        });
    }

    function _drawNodes() {
        AppState.nodes.forEach(node => {
            const p = pos[node.id];
            if (!p) return;
            const isH = hov === node.id;
            const isS = document.getElementById('src-sel').value === node.id;
            const r = isH ? 13 : isS ? 12 : 9;

            // Glow
            cx.save();
            cx.globalAlpha = isH ? 0.45 : 0.2;
            cx.fillStyle = p.col;
            cx.beginPath();
            cx.arc(p.x, p.y, r * 2.5, 0, Math.PI * 2);
            cx.fill();

            // Body
            cx.globalAlpha = 1;
            cx.fillStyle = p.col;
            cx.shadowColor = p.col;
            cx.shadowBlur = isH ? 22 : 10;
            cx.beginPath();
            cx.arc(p.x, p.y, r, 0, Math.PI * 2);
            cx.fill();

            // Source marker ring
            if (isS) {
                cx.strokeStyle = '#ffffff';
                cx.lineWidth = 2;
                cx.shadowBlur = 0;
                cx.beginPath();
                cx.arc(p.x, p.y, r + 4, 0, Math.PI * 2);
                cx.stroke();
            }
            cx.restore();

            // Label
            cx.save();
            cx.font = `${isH ? '500 ' : ''}12px 'Rajdhani', sans-serif`;
            cx.fillStyle = isH ? '#fff' : 'rgba(221,238,255,.7)';
            cx.textAlign = 'center';
            cx.fillText(node.label, p.x, p.y + r + 13);
            cx.restore();
        });
    }

    function _onHover(e) {
        const rc = cv.getBoundingClientRect();
        const mx = e.clientX - rc.left;
        const my = e.clientY - rc.top;
        hov = null;
        for (const node of AppState.nodes) {
            const p = pos[node.id];
            if (p && Math.hypot(mx - p.x, my - p.y) < 15) { hov = node.id; break; }
        }
        cv.style.cursor = hov ? 'pointer' : 'default';
        draw();
    }

    return { init, draw };
})();
