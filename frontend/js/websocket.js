/**
 * websocket.js
 * ------------
 * Comunicación con el backend FastAPI.
 * API mejorada con soporte para:
 *   - Gestión de inputs (CRUD)
 *   - WebSocket raw para streaming
 *   - Análisis en tiempo real
 */

const API = (() => {
    const BASE = 'http://localhost:8000';
    const WS_BASE = 'ws://localhost:8000';

    // ──────────────────────────────────────────────────────
    // Grafo y configuración
    // ──────────────────────────────────────────────────────

    async function loadDefaultGraph() {
        try {
            const r = await fetch(`${BASE}/api/graph`);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        } catch (e) {
            console.warn('Backend offline:', e.message);
            return null;
        }
    }

    async function getModelInfo() {
        try { const r = await fetch(`${BASE}/api/model/info`); return r.ok ? r.json() : null; }
        catch { return null; }
    }

    async function getLLMStatus() {
        try { const r = await fetch(`${BASE}/api/llm/status`); return r.ok ? r.json() : null; }
        catch { return null; }
    }

    // ──────────────────────────────────────────────────────
    // Análisis con IA
    // ──────────────────────────────────────────────────────

    async function analyze(trajectories, criterion, userProfile) {
        const r = await fetch(`${BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ trajectories, criterion, user_profile: userProfile }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    }

    // ──────────────────────────────────────────────────────
    // Gestión de Inputs (BD SQLite)
    // ──────────────────────────────────────────────────────

    async function createInput(input) {
        const r = await fetch(`${BASE}/api/inputs/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(input),
        });
        return r.json();
    }

    async function listInputs() {
        const r = await fetch(`${BASE}/api/inputs/list`);
        return r.json();
    }

    async function getInput(inputId) {
        const r = await fetch(`${BASE}/api/inputs/${inputId}`);
        return r.json();
    }

    async function deleteInput(inputId) {
        const r = await fetch(`${BASE}/api/inputs/${inputId}`, { method: 'DELETE' });
        return r.json();
    }

    // ──────────────────────────────────────────────────────
    // WebSocket para streaming
    // ──────────────────────────────────────────────────────

    /**
     * WebSocket con callback raw — recibe TODOS los tipos de mensaje.
     * Ahora incluye manejo de graph_info.
     */
    function exploreWSRaw(request, onMessage, onError) {
        let ws;
        try {
            ws = new WebSocket(`${WS_BASE}/ws/explore`);

            ws.onopen = () => {
                console.log('WS connected, sending request...');
                ws.send(JSON.stringify({ type: 'start', data: request }));
            };

            ws.onmessage = e => {
                try {
                    const msg = JSON.parse(e.data);
                    console.log('WS message:', msg.type);
                    onMessage(msg);
                } catch (parseErr) {
                    console.error('WS parse error:', parseErr, e.data);
                }
            };

            ws.onerror = (ev) => {
                console.error('WS error:', ev);
                onError('WebSocket error. Check backend connection.');
            };

            ws.onclose = e => {
                console.log('WS closed:', e.code, e.reason);
                if (e.code !== 1000) onError(`WS closed unexpectedly (code ${e.code})`);
            };

        } catch (e) {
            console.error('WS creation error:', e);
            onError(e.message);
        }
    }

    // Keep old method for compatibility
    function exploreWS(request, onStep, onResult, onDone, onError) {
        exploreWSRaw(request, msg => {
            switch (msg.type) {
                case 'graph_info':
                    // Nueva: información del grafo
                    if (onStep) onStep({ ...msg, type: 'step' });
                    break;
                case 'step': onStep(msg); break;
                case 'result': onResult(msg); break;
                case 'done': onDone(); break;
                case 'error': onError(msg.msg); break;
            }
        }, onError);
    }

    return {
        loadDefaultGraph, analyze, getLLMStatus, getModelInfo,
        createInput, listInputs, getInput, deleteInput,
        exploreWSRaw, exploreWS
    };
})();