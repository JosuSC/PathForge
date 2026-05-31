/**
 * universe.js
 * -----------
 * Módulo SpaceBG: fondo espacial dinámico 2D (canvas).
 * Renderiza: nebulosas, estrellas parpadeantes, constelaciones,
 * planetas con anillos, estrellas fugaces.
 *
 * Exporta globalmente: SpaceBG
 */

const SpaceBG = (() => {
    let cv, cx, W, H;
    let stars = [], nebulae = [], planets = [], shoots = [], consts = [];
    let t = 0;

    // ── Init ─────────────────────────────────────────────────
    function init() {
        cv = document.getElementById('space-canvas');
        cx = cv.getContext('2d');
        resize();
        window.addEventListener('resize', () => { resize(); build(); });
        build();
        render();
    }

    function resize() {
        W = cv.width = window.innerWidth;
        H = cv.height = window.innerHeight;
    }

    // ── Scene builder ─────────────────────────────────────────
    function build() {
        _buildStars();
        _buildNebulae();
        _buildConstellations();
        _buildPlanets();
        shoots = [];
    }

    function _buildStars() {
        stars = [];
        const COLORS = ['#a0cfff', '#ffe8b0', '#c0a0ff', '#ffffff', '#80ffcc', '#ffb0b0'];
        for (let i = 0; i < 900; i++) {
            stars.push({
                x: Math.random() * W,
                y: Math.random() * H,
                r: 0.2 + Math.random() * 1.5,
                a: 0.3 + Math.random() * 0.7,
                da: (Math.random() - 0.5) * 0.007,
                tw: Math.random() * Math.PI * 2,
                col: COLORS[Math.floor(Math.random() * COLORS.length)],
            });
        }
    }

    function _buildNebulae() {
        nebulae = [];
        const PALETTES = [
            ['#0a1a3a', '#1a0a2a'], ['#0a2a1a', '#1a2a0a'],
            ['#2a0a1a', '#1a1a2a'], ['#0a1a2a', '#2a0a2a'],
            ['#1a0a3a', '#0a1a1a'],
        ];
        for (let i = 0; i < 7; i++) {
            const pal = PALETTES[i % PALETTES.length];
            nebulae.push({
                x: Math.random() * W,
                y: Math.random() * H,
                rx: 100 + Math.random() * 220,
                ry: 70 + Math.random() * 160,
                rot: Math.random() * Math.PI,
                c1: pal[0],
                c2: pal[1],
                op: 0.1 + Math.random() * 0.14,
            });
        }
    }

    function _buildConstellations() {
        // Fractional coordinates → pixel positions
        const PATTERNS = [
            [[.08, .18], [.14, .13], [.21, .17], [.27, .12], [.34, .19]],
            [[.58, .08], [.64, .14], [.71, .09], [.69, .19], [.64, .24]],
            [[.84, .38], [.90, .33], [.92, .41], [.88, .47], [.82, .44]],
            [[.38, .68], [.44, .63], [.50, .71], [.48, .79], [.41, .77]],
            [[.18, .83], [.24, .78], [.31, .81], [.29, .89]],
            [[.55, .35], [.60, .28], [.66, .33], [.64, .42], [.58, .45]],
        ];
        consts = PATTERNS.map(pts => ({
            pts: pts.map(p => ({ x: p[0] * W, y: p[1] * H })),
        }));
    }

    function _buildPlanets() {
        planets = [
            { x: W * .76, y: H * .19, r: 17, h: 220, ring: true, sp: .0003 },
            { x: W * .14, y: H * .72, r: 10, h: 40, ring: false, sp: .0005 },
            { x: W * .89, y: H * .76, r: 7, h: 160, ring: false, sp: .0008 },
            { x: W * .44, y: H * .87, r: 13, h: 300, ring: true, sp: .0002 },
        ];
    }

    // ── Render loop ───────────────────────────────────────────
    function render() {
        requestAnimationFrame(render);
        t += 0.007;
        _drawBackground();
        _drawNebulae();
        _drawConstellations();
        _drawStars();
        _drawPlanets();
        _drawShootingStars();
    }

    function _drawBackground() {
        const g = cx.createRadialGradient(W * .5, H * .5, 0, W * .5, H * .5, Math.max(W, H) * .72);
        g.addColorStop(0, '#030b1a');
        g.addColorStop(0.5, '#02060f');
        g.addColorStop(1, '#00000a');
        cx.fillStyle = g;
        cx.fillRect(0, 0, W, H);
    }

    function _drawNebulae() {
        nebulae.forEach(n => {
            cx.save();
            cx.translate(n.x, n.y);
            cx.rotate(n.rot + t * .008);
            const ng = cx.createRadialGradient(0, 0, 0, 0, 0, n.rx);
            ng.addColorStop(0, n.c1);
            ng.addColorStop(1, 'transparent');
            cx.globalAlpha = n.op * (0.8 + 0.2 * Math.sin(t * .3));
            cx.scale(1, n.ry / n.rx);
            cx.beginPath();
            cx.arc(0, 0, n.rx, 0, Math.PI * 2);
            cx.fillStyle = ng;
            cx.fill();
            cx.restore();
        });
    }

    function _drawConstellations() {
        cx.save();
        // Lines
        cx.strokeStyle = 'rgba(140,170,255,0.14)';
        cx.lineWidth = 0.7;
        cx.setLineDash([3, 6]);
        consts.forEach(c => {
            cx.beginPath();
            c.pts.forEach((p, i) => i === 0 ? cx.moveTo(p.x, p.y) : cx.lineTo(p.x, p.y));
            cx.stroke();
        });
        // Dots
        cx.setLineDash([]);
        cx.fillStyle = 'rgba(180,210,255,0.6)';
        consts.forEach(c => c.pts.forEach(p => {
            cx.beginPath();
            cx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
            cx.fill();
        }));
        cx.restore();
    }

    function _drawStars() {
        stars.forEach(s => {
            s.tw += s.da;
            const a = s.a * (0.5 + 0.5 * Math.sin(s.tw));
            cx.save();
            cx.globalAlpha = a;
            cx.fillStyle = s.col;
            cx.beginPath();
            cx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            cx.fill();
            // Cross flare for bright stars
            if (s.r > 1.1) {
                cx.globalAlpha = a * 0.35;
                cx.strokeStyle = s.col;
                cx.lineWidth = 0.5;
                const fl = s.r * 3;
                cx.beginPath();
                cx.moveTo(s.x - fl, s.y); cx.lineTo(s.x + fl, s.y);
                cx.moveTo(s.x, s.y - fl); cx.lineTo(s.x, s.y + fl);
                cx.stroke();
            }
            cx.restore();
        });
    }

    function _drawPlanets() {
        planets.forEach(p => {
            const px = p.x + Math.sin(t * p.sp * 100) * 12;
            const py = p.y + Math.cos(t * p.sp * 80) * 7;
            cx.save();

            // Glow halo
            const gl = cx.createRadialGradient(px, py, 0, px, py, p.r * 3);
            gl.addColorStop(0, `hsla(${p.h},80%,60%,.18)`);
            gl.addColorStop(1, 'transparent');
            cx.fillStyle = gl;
            cx.beginPath();
            cx.arc(px, py, p.r * 3, 0, Math.PI * 2);
            cx.fill();

            // Planet body
            const bo = cx.createRadialGradient(px - p.r * .3, py - p.r * .3, p.r * .1, px, py, p.r);
            bo.addColorStop(0, `hsl(${p.h + 30},70%,72%)`);
            bo.addColorStop(0.5, `hsl(${p.h},60%,42%)`);
            bo.addColorStop(1, `hsl(${p.h - 20},70%,18%)`);
            cx.fillStyle = bo;
            cx.beginPath();
            cx.arc(px, py, p.r, 0, Math.PI * 2);
            cx.fill();

            // Atmosphere rim
            cx.globalAlpha = 0.28;
            cx.strokeStyle = `hsl(${p.h + 40},80%,70%)`;
            cx.lineWidth = 1.5;
            cx.beginPath();
            cx.arc(px, py, p.r + 2, 0, Math.PI * 2);
            cx.stroke();

            // Ring
            if (p.ring) {
                cx.globalAlpha = 0.33;
                cx.strokeStyle = `hsl(${p.h},60%,65%)`;
                cx.lineWidth = 2;
                cx.save();
                cx.translate(px, py);
                cx.scale(1, 0.28);
                cx.beginPath();
                cx.arc(0, 0, p.r * 2.2, 0, Math.PI * 2);
                cx.stroke();
                cx.restore();
            }
            cx.restore();
        });
    }

    function _drawShootingStars() {
        // Spawn
        if (shoots.length < 3 && Math.random() < 0.004) {
            shoots.push({
                x: Math.random() * W,
                y: Math.random() * (H * 0.4),
                vx: 4 + Math.random() * 6,
                vy: 2 + Math.random() * 3,
                len: 65 + Math.random() * 80,
                life: 1,
            });
        }
        shoots = shoots.filter(ss => {
            ss.x += ss.vx; ss.y += ss.vy; ss.life -= 0.024;
            if (ss.life <= 0) return false;
            const mag = Math.hypot(ss.vx, ss.vy);
            cx.save();
            cx.globalAlpha = ss.life * 0.85;
            const sg = cx.createLinearGradient(ss.x, ss.y, ss.x - ss.vx * ss.len / mag, ss.y - ss.vy * ss.len / mag);
            sg.addColorStop(0, '#ffffff');
            sg.addColorStop(1, 'transparent');
            cx.strokeStyle = sg;
            cx.lineWidth = 1.5;
            cx.beginPath();
            cx.moveTo(ss.x, ss.y);
            cx.lineTo(ss.x - ss.vx * ss.len / mag, ss.y - ss.vy * ss.len / mag);
            cx.stroke();
            cx.restore();
            return true;
        });
    }

    return { init };
})();
