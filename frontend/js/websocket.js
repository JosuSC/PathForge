/**
 * websocket.js
 * ------------
 * Módulo API: comunicación con el backend FastAPI.
 * WebSocket para exploración en tiempo real + REST para análisis.
 * Demo engine offline como fallback automático.
 *
 * Exporta globalmente: API
 */

const API = (() => {
    const BASE = 'http://localhost:8000';
    const WS_BASE = 'ws://localhost:8000';

    // ── REST calls ─────────────────────────────────────────────
    async function loadDefaultGraph() {
        try {
            const r = await fetch(`${BASE}/api/graph`);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return await r.json();
        } catch (e) {
            console.warn('Backend offline, using demo data:', e.message);
            return null;
        }
    }

    async function analyze(trajectories, criterion, userProfile) {
        const r = await fetch(`${BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trajectories,
                criterion,
                user_profile: userProfile,
            }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    }

    async function getLLMStatus() {
        try {
            const r = await fetch(`${BASE}/api/llm/status`);
            return r.ok ? r.json() : null;
        } catch { return null; }
    }

    // ── WebSocket exploration ──────────────────────────────────
    function exploreWS(request, onStep, onResult, onDone, onError) {
        let ws;
        try {
            ws = new WebSocket(`${WS_BASE}/ws/explore`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ type: 'start', data: request }));
            };

            ws.onmessage = e => {
                const msg = JSON.parse(e.data);
                switch (msg.type) {
                    case 'step': onStep(msg); break;
                    case 'result': onResult(msg); break;
                    case 'done': onDone(); break;
                    case 'error': onError(msg.msg); break;
                }
            };

            ws.onerror = () => onError('WebSocket connection failed. Is the backend running?');
            ws.onclose = e => { if (e.code !== 1000) onError('WebSocket closed unexpectedly'); };

        } catch (e) {
            onError(e.message);
        }
    }

    return { loadDefaultGraph, analyze, getLLMStatus, exploreWS };
})();


// ── Demo Engine (offline fallback) ────────────────────────────
/**
 * Simula el Beam Search localmente cuando el backend no está
 * disponible. Genera trayectorias realistas desde el grafo del
 * estado local y emite los mismos eventos que el WS real.
 */
function runDemoExploration(request, onStep, onResult, onDone) {
    // Build adjacency from AppState
    const adj = {};
    AppState.edges.forEach(e => {
        if (!adj[e.from_node]) adj[e.from_node] = [];
        adj[e.from_node].push(e.to_node);
    });

    let beam = [[request.source]];
    let completed = [];
    let depth = 0;
    const maxD = Math.min(request.max_depth || 5, 6);
    const bw = request.beam_width || 8;

    function step() {
        if (depth >= maxD || !beam.length) {
            // Score and return
            const scored = completed.slice(0, 20).map((path, i) => {
                const n0 = AppState.nodes.find(n => n.id === path[0]);
                const nF = AppState.nodes.find(n => n.id === path[path.length - 1]);
                const sg = (nF && n0)
                    ? (nF.avg_salary - n0.avg_salary) / Math.max(n0.avg_salary, 1)
                    : Math.random() * 3;
                return {
                    nodes: path,
                    pareto_rank: i < 3 ? 0 : i < 7 ? 1 : 2,
                    crowding_distance: Math.random() * 5,
                    scores: {
                        salary_growth: +sg.toFixed(3),
                        avg_demand: +(0.6 + Math.random() * 0.35).toFixed(3),
                        avg_satisfaction: +(0.6 + Math.random() * 0.28).toFixed(3),
                        final_salary: nF ? nF.avg_salary : 50000 + Math.random() * 120000,
                        total_years: path.length * 2,
                        avg_risk: +(0.1 + Math.random() * 0.5).toFixed(3),
                        avg_difficulty: +(0.2 + Math.random() * 0.55).toFixed(3),
                    },
                };
            }).sort((a, b) => b.scores.salary_growth - a.scores.salary_growth);

            onResult({ trajectories: scored });
            onDone();
            return;
        }

        const nextBeam = [];
        beam.forEach(path => {
            const cur = path[path.length - 1];
            (adj[cur] || []).forEach(next => {
                if (!path.includes(next)) {
                    const newPath = [...path, next];
                    if (newPath.length >= 2) completed.push(newPath);
                    nextBeam.push(newPath);
                }
            });
        });

        beam = nextBeam.slice(0, bw);
        depth++;
        onStep({ depth, beam, completed: completed.slice(-15) });
        setTimeout(step, 520);
    }

    setTimeout(step, 200);
}
