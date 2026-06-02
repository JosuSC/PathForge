/**
 * ui.js
 * -----
 * Módulo principal — gestiona estado, exploración y análisis.
 *
 */

// ── Global App State ──────────────────────────────────────────
const AppState = {
    nodes: [],
    edges: [],
    trajectories: [],
    terminalGroups: {},
    terminalsFound: [],
    exploring: false,
    showAllTraj: false,
    currentView: 'setup',
};

const PALETTE = [
    0x00c8ff, 0xb44dff, 0xffc843, 0xff6b35, 0x00ff8c, 0xff3355,
    0x4fc3f7, 0xce93d8, 0xffcc02, 0x69f0ae, 0xf48fb1, 0x80deea,
    0xff9800, 0x00bcd4, 0x9c27b0, 0x4caf50,
];
const paletteColor = i => '#' + PALETTE[i % PALETTE.length].toString(16).padStart(6, '0');
const terminalColors = [0xffd700, 0x00ff8c, 0xb44dff, 0xff6b35, 0x00c8ff, 0xff3355];

// ── Toasts ────────────────────────────────────────────────────
function toast(msg, type = 'info') {
    const c = document.getElementById('toasts');
    const el = document.createElement('div');
    el.className = `toast t-${type === 'ok' ? 'ok' : type === 'err' || type === 'error' ? 'err' : 'info'}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity .4s';
        setTimeout(() => el.remove(), 400);
    }, 3500);
}

// ── Stats ─────────────────────────────────────────────────────
function updateStats() {
    document.getElementById('hs-n').textContent = AppState.nodes.length;
    document.getElementById('hs-e').textContent = AppState.edges.length;
    document.getElementById('hs-f').textContent = AppState.trajectories.length;
    document.getElementById('hs-o').textContent = AppState.trajectories.filter(t => t.pareto_rank === 0).length;
}

// ── Node / Edge lists ─────────────────────────────────────────
function renderNodeList() {
    const list = document.getElementById('node-list');
    const sel = document.getElementById('src-sel');
    const curSrc = sel.value;
    list.innerHTML = '';
    sel.innerHTML = '<option value="">— select —</option>';

    const ef = document.getElementById('ee-from');
    const et = document.getElementById('ee-to');
    if (ef) { ef.innerHTML = ''; et.innerHTML = ''; }

    AppState.nodes.forEach((node, i) => {
        const color = paletteColor(i);
        const div = document.createElement('div');
        div.className = 'node-item' + (AppState.selectedNode === node.id ? ' sel-node' : '');
        div.innerHTML = `
            <div class="n-star" style="background:${color};color:${color}"></div>
            <div style="flex:1;min-width:0">
                <div class="n-name">${node.label}</div>
                <div class="n-sal">$${node.avg_salary.toLocaleString()}/yr</div>
            </div>
            <button class="n-rm" onclick="event.stopPropagation();removeNode('${node.id}')">✕</button>
        `;
        div.onclick = () => { AppState.selectedNode = node.id; renderNodeList(); };
        list.appendChild(div);

        sel.innerHTML += `<option value="${node.id}"${node.id === curSrc ? ' selected' : ''}>${node.label}</option>`;
        if (ef) {
            ef.innerHTML += `<option value="${node.id}">${node.label}</option>`;
            et.innerHTML += `<option value="${node.id}">${node.label}</option>`;
        }
    });
}

function renderEdgeList() {
    const el = document.getElementById('edge-list');
    el.innerHTML = '';
    if (!AppState.edges.length) { el.innerHTML = '<div class="empty-msg">No wormholes yet</div>'; return; }
    AppState.edges.forEach((e, i) => {
        const fl = AppState.nodes.find(n => n.id === e.from_node)?.label || e.from_node;
        const tl = AppState.nodes.find(n => n.id === e.to_node)?.label || e.to_node;
        const div = document.createElement('div');
        div.className = 'edge-item';
        div.innerHTML = `
            <span class="e-from">${fl}</span>
            <span class="e-arr">→</span>
            <span class="e-to">${tl}</span>
            <span style="font-size:9px;color:var(--muted);flex-shrink:0">${e.transition_years}yr</span>
            <button class="e-rm" onclick="removeEdge(${i})">✕</button>
        `;
        el.appendChild(div);
    });
}

// ── View switching ────────────────────────────────────────────
function switchView(name) {
    AppState.currentView = name;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${name}`).classList.add('active');
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${name}`).classList.add('active');

    if (name === 'universe') {
        // ✅ FIX: Usar requestAnimationFrame para asegurar dimensiones
        requestAnimationFrame(() => {
            if (!U3D._init) {
                U3D.init();
                U3D._init = true;
            } else {
                // ✅ FIX: Si ya estaba inicializado, forzar resize
                U3D.resize();
            }
            // Reconstruir si hay datos pero no estamos explorando
            if (AppState.nodes.length && !AppState.exploring) {
                U3D.rebuildAll();
            }
        });
    }
}

// ── Modals ────────────────────────────────────────────────────
function openModal(id) { if (id === 'modal-edge') renderNodeList(); document.getElementById(id).classList.add('show'); }
function closeModal(id) { document.getElementById(id).classList.remove('show'); }

// ── Range binding ─────────────────────────────────────────────
function bindRange(id, valueId, decimals = 0) {
    const el = document.getElementById(id);
    const vEl = document.getElementById(valueId);
    if (!el || !vEl) return;
    const fmt = v => decimals ? parseFloat(v).toFixed(decimals) : v;
    vEl.textContent = fmt(el.value);
    el.addEventListener('input', () => vEl.textContent = fmt(el.value));
}

// ── CRUD ──────────────────────────────────────────────────────
function confirmAddNode() {
    const id = document.getElementById('nn-id').value.trim().replace(/\s+/g, '_');
    const label = document.getElementById('nn-label').value.trim();
    const salary = parseFloat(document.getElementById('nn-salary').value) || 50000;
    const demand = parseFloat(document.getElementById('nn-dem').value);
    const sat = parseFloat(document.getElementById('nn-sat').value);
    const exp = parseInt(document.getElementById('nn-exp').value) || 0;
    const skills = document.getElementById('nn-skills').value.split(',').map(s => s.trim()).filter(Boolean);

    if (!id || !label) { toast('ID and Label required', 'err'); return; }
    if (AppState.nodes.find(n => n.id === id)) { toast('Node ID already exists', 'err'); return; }

    AppState.nodes.push({ id, label, avg_salary: salary, demand, satisfaction: sat, years_experience: exp, skills, type: 'role' });
    closeModal('modal-node');
    renderNodeList(); renderEdgeList(); Preview.draw(); updateStats();
    toast(`⭐ ${label} added`, 'ok');
}

function confirmAddEdge() {
    const from = document.getElementById('ee-from').value;
    const to = document.getElementById('ee-to').value;
    const years = parseInt(document.getElementById('ee-years').value) || 2;
    const diff = parseFloat(document.getElementById('ee-diff').value);
    const risk = parseFloat(document.getElementById('ee-risk').value);

    if (from === to) { toast('Source and destination must differ', 'err'); return; }
    if (AppState.edges.find(e => e.from_node === from && e.to_node === to)) { toast('Edge already exists', 'err'); return; }

    AppState.edges.push({ from_node: from, to_node: to, transition_years: years, difficulty: diff, risk });
    closeModal('modal-edge');
    renderEdgeList(); Preview.draw(); updateStats();
    toast('🌀 Wormhole created', 'ok');
}

function removeNode(id) {
    AppState.nodes = AppState.nodes.filter(n => n.id !== id);
    AppState.edges = AppState.edges.filter(e => e.from_node !== id && e.to_node !== id);
    renderNodeList(); renderEdgeList(); Preview.draw(); updateStats();
}

function removeEdge(i) {
    AppState.edges.splice(i, 1);
    renderEdgeList(); Preview.draw(); updateStats();
}

// ── Default graph ─────────────────────────────────────────────
async function loadDefault() {
    let data = await API.loadDefaultGraph();
    if (!data) {
        data = _getEmbeddedDefaultData();
    }

    AppState.nodes = data.nodes.map(n => ({
        id: n.id, label: n.label, avg_salary: n.avg_salary,
        demand: n.demand, satisfaction: n.satisfaction,
        years_experience: n.years_experience, skills: n.skills || [],
        type: n.type || 'role',
    }));
    AppState.edges = data.edges.map(e => ({
        from_node: e.from || e.from_node,
        to_node: e.to || e.to_node,
        transition_years: e.transition_years,
        difficulty: e.difficulty,
        risk: e.risk,
    }));

    renderNodeList(); renderEdgeList(); Preview.draw(); updateStats();
    document.getElementById('src-sel').value = 'junior_dev';
    toast('Default career graph loaded', 'ok');
}

// ── Exploration ───────────────────────────────────────────────
function launchExploration() {
    const src = document.getElementById('src-sel').value;
    if (!src) { toast('Select a starting role first', 'err'); return; }
    if (AppState.nodes.length < 2) { toast('Add at least 2 nodes', 'err'); return; }

    AppState.exploring = true;
    document.getElementById('launch-btn').disabled = true;
    toast('Launching exploration...', 'info');

    // ✅ FIX: Primero cambiar de vista
    switchView('universe');

    // ✅ FIX: Usar requestAnimationFrame encadenado para asegurar init correcto
    requestAnimationFrame(() => {
        if (!U3D._init) {
            U3D.init();
            U3D._init = true;
        }
        // Asegurar resize después del primer frame
        requestAnimationFrame(() => {
            U3D.resize();
            U3D.showSourceNode(src, []);
            requestAnimationFrame(() => _startSearch());
        });
    });
}

function _startSearch() {
    const src = document.getElementById('src-sel').value;
    const maxD = parseInt(document.getElementById('md').value);

    const overlay = document.getElementById('srch-overlay');
    const fill = document.getElementById('so-fill');
    const sub = document.getElementById('so-sub');
    const stat = document.getElementById('so-stat');
    const dots = document.getElementById('so-dots');

    overlay.classList.add('show');
    dots.innerHTML = Array.from({ length: maxD }, (_, i) => `<div class="so-d" id="sod-${i}"></div>`).join('');
    fill.style.width = '0%';

    AppState.trajectories = [];
    AppState.terminalGroups = {};
    AppState.terminalsFound = [];

    const request = {
        source: src,
        nodes: AppState.nodes,
        edges: AppState.edges,
        profile: document.getElementById('prof-sel').value,
        max_years: parseInt(document.getElementById('my').value),
        max_risk: parseFloat(document.getElementById('mr').value),
        beam_width: parseInt(document.getElementById('bw').value),
        max_depth: maxD,
        top_k: parseInt(document.getElementById('tk').value),
        user_profile: document.getElementById('user-profile').value,
        use_simulation: true,
    };

    function onGraphInfo(msg) {
        U3D.setTerminals(msg.terminals || []);
        U3D.showSourceNode(msg.source, msg.terminals || []);
        toast(`Graph loaded: ${msg.nodes.length} nodes, ${(msg.terminals || []).length} terminals`, 'info');
    }

    function onStep(msg) {
        const pct = Math.min(98, (msg.depth / maxD) * 100);
        fill.style.width = pct + '%';
        sub.textContent = `Beam Search · Depth ${msg.depth} of ${maxD} · ${msg.total_discovered || 0} nodes discovered`;
        stat.textContent = `${(msg.beam || []).length} active beams · ${(msg.completed || []).length} paths found`;

        const dd = document.getElementById(`sod-${msg.depth - 1}`);
        if (dd) dd.classList.add('on');

        document.getElementById('sb-st').textContent = 'Exploring';
        document.getElementById('sb-d').textContent = msg.depth;
        document.getElementById('sb-b').textContent = (msg.beam || []).length;
        document.getElementById('sb-p').textContent = (msg.completed || []).length;

        U3D.onBeamStep(msg);

        if (msg.terminal_reached) {
            toast(`🎯 Terminal reached: ${msg.terminal_reached}`, 'ok');
        }
    }

    function onResult(msg) {
        AppState.trajectories = msg.trajectories || [];
        AppState.terminalGroups = msg.terminal_groups || {};
        AppState.terminalsFound = msg.terminals_found || [];
        fill.style.width = '100%';
        sub.textContent = `Found ${AppState.trajectories.length} trajectories across ${AppState.terminalsFound.length} destinations`;
        updateStats();
        renderAnalysis();
        document.getElementById('tab-analysis').classList.add('has-data');
    }

    function onDone() {
        AppState.exploring = false;
        document.getElementById('launch-btn').disabled = false;
        overlay.classList.remove('show');
        document.getElementById('sb-st').textContent = 'Complete';
        document.getElementById('sb-p').textContent = AppState.trajectories.length;
        U3D.clearBeamLines();

        const nT = AppState.terminalsFound.length;
        toast(`🌌 ${AppState.trajectories.length} futures found across ${nT} possible endings!`, 'ok');

        const entries = Object.entries(AppState.terminalGroups);
        let i = 0;
        const cycle = setInterval(() => {
            if (i >= entries.length) {
                clearInterval(cycle);
                const best = AppState.trajectories[0];
                if (best) {
                    U3D.highlightTrajectory(best.nodes, 0xffd700);
                    toast('🏆 Best trajectory highlighted in gold', 'ok');
                }
                return;
            }
            const [terminal, trajs] = entries[i];
            const color = terminalColors[i % terminalColors.length];
            if (trajs.length) {
                U3D.highlightTrajectory(trajs[0].nodes, color);
                toast(`Exploring future: → ${terminal}`, 'info');
            }
            i++;
        }, 1200);
    }

    function onError(msg) {
        console.warn('WS error, fallback to demo:', msg);
        _runDemoExploration(request, onStep, onResult, onDone);
    }

    function onMessage(msg) {
        switch (msg.type) {
            case 'graph_info': onGraphInfo(msg); break;
            case 'step': onStep(msg); break;
            case 'result': onResult(msg); break;
            case 'done': onDone(); break;
            case 'error': onError(msg.msg); break;
        }
    }

    try {
        API.exploreWSRaw(request, onMessage, onError);
    } catch (e) {
        _runDemoExploration(request, onStep, onResult, onDone);
    }
}

// ── Demo fallback ─────────────────────────────────────────────
function _runDemoExploration(request, onStep, onResult, onDone) {
    toast('Running in demo mode (backend offline)', 'info');

    const terminals = AppState.edges
        .map(e => e.to_node)
        .filter(id => !AppState.edges.some(e2 => e2.from_node === id));
    U3D.setTerminals(terminals);

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
    const discoveredNodes = new Set([request.source]);
    const discoveredEdges = new Set();

    function step() {
        if (depth >= maxD || !beam.length) {
            const termGroups = {};
            const scored = completed.slice(0, 20).map((path, i) => {
                const n0 = AppState.nodes.find(n => n.id === path[0]);
                const nF = AppState.nodes.find(n => n.id === path[path.length - 1]);
                const sg = (nF && n0) ? (nF.avg_salary - n0.avg_salary) / Math.max(n0.avg_salary, 1) : Math.random() * 3;
                const term = path[path.length - 1];
                const t = {
                    nodes: path, pareto_rank: i < 3 ? 0 : i < 7 ? 1 : 2,
                    crowding_distance: Math.random() * 5,
                    terminal_node: term,
                    is_terminal_end: terminals.includes(term),
                    scores: {
                        salary_growth: +sg.toFixed(3),
                        avg_demand: +(0.6 + Math.random() * 0.35).toFixed(3),
                        avg_satisfaction: +(0.6 + Math.random() * 0.28).toFixed(3),
                        final_salary: nF ? nF.avg_salary : 50000 + Math.random() * 120000,
                        total_years: path.length * 2,
                        avg_risk: +(0.1 + Math.random() * 0.5).toFixed(3),
                        avg_difficulty: +(0.2 + Math.random() * 0.55).toFixed(3),
                        ml_success_prob: +(0.4 + Math.random() * 0.5).toFixed(3),
                        is_terminal_end: terminals.includes(term) ? 1 : 0,
                    },
                };
                if (!termGroups[term]) termGroups[term] = [];
                termGroups[term].push(t);
                return t;
            }).sort((a, b) => b.scores.salary_growth - a.scores.salary_growth);

            onResult({ trajectories: scored, terminal_groups: termGroups, terminals_found: Object.keys(termGroups) });
            onDone();
            return;
        }

        const nextBeam = [];
        const newNodes = [], newEdges = [];

        beam.forEach(path => {
            const cur = path[path.length - 1];
            (adj[cur] || []).forEach(next => {
                if (!path.includes(next)) {
                    const np = [...path, next];
                    if (np.length >= 2) completed.push(np);
                    nextBeam.push(np);

                    if (!discoveredNodes.has(next)) { newNodes.push(next); discoveredNodes.add(next); }
                    const ek = `${cur}-${next}`;
                    if (!discoveredEdges.has(ek)) { newEdges.push([cur, next]); discoveredEdges.add(ek); }
                }
            });
        });

        beam = nextBeam.slice(0, bw);
        depth++;

        const termReached = newNodes.find(id => terminals.includes(id)) || '';

        onStep({
            depth, beam, completed: completed.slice(-15),
            new_nodes: newNodes, new_edges: newEdges,
            terminal_reached: termReached,
            total_discovered: discoveredNodes.size,
        });

        setTimeout(step, 600);
    }
    setTimeout(step, 300);
}

// ── Analysis rendering ────────────────────────────────────────
function renderAnalysis() {
    const trajs = AppState.trajectories;
    document.getElementById('traj-count').textContent =
        `(${trajs.length} total · ${trajs.filter(t => t.pareto_rank === 0).length} optimal · ${AppState.terminalsFound.length} destinations)`;

    if (!trajs.length) return;

    const best = trajs[0];
    document.getElementById('best-sec').style.display = 'block';
    document.getElementById('bt-path').innerHTML = best.nodes.map((n, i) => `
        ${i > 0 ? '<span class="bt-arrow">→</span>' : ''}
        <span class="bt-node">${AppState.nodes.find(nd => nd.id === n)?.label || n}</span>
    `).join('');

    const s = best.scores;
    const simHtml = s.sim_salary_p50 ? `
        <div class="bt-m"><div class="bt-mv">${(s.sim_success_mean || 0).toFixed(0)}%</div><div class="bt-mk">Sim. Success</div></div>
    ` : '';
    document.getElementById('bt-mets').innerHTML = `
        <div class="bt-m"><div class="bt-mv">+${((s.salary_growth || 0) * 100).toFixed(0)}%</div><div class="bt-mk">Salary Growth</div></div>
        <div class="bt-m"><div class="bt-mv">$${(s.final_salary || 0).toLocaleString('en', { maximumFractionDigits: 0 })}</div><div class="bt-mk">Final Salary</div></div>
        <div class="bt-m"><div class="bt-mv">${(s.total_years || 0).toFixed(0)} yr</div><div class="bt-mk">Duration</div></div>
        <div class="bt-m"><div class="bt-mv">${((s.avg_satisfaction || 0) * 100).toFixed(0)}%</div><div class="bt-mk">Satisfaction</div></div>
        ${simHtml}
    `;

    _renderStepBreakdown(best);
    _renderTerminalGroups();
    _renderAllTrajectories(trajs);
}

function _renderTerminalGroups() {
    const container = document.getElementById('traj-grid');
    const groups = AppState.terminalGroups;
    if (!Object.keys(groups).length) return;

    let html = `<div style="grid-column:1/-1;margin-bottom:12px">
        <div class="ah" style="margin-bottom:12px">
            <h2 style="font-size:11px">ALTERNATIVE FUTURES — BY DESTINATION</h2>
            <p>Each destination represents a different type of success</p>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:10px">`;

    Object.entries(groups).forEach(([terminal, trajs], i) => {
        const best = trajs[0];
        const color = '#' + terminalColors[i % terminalColors.length].toString(16).padStart(6, '0');
        const tNode = AppState.nodes.find(n => n.id === terminal);
        const label = tNode?.label || terminal;
        const sal = best?.scores?.final_salary || 0;

        html += `<div style="border:1px solid ${color};background:${color}11;padding:10px 14px;cursor:pointer;transition:all .2s;min-width:140px"
                      onclick="highlightTerminalGroup('${terminal}', ${i})"
                      onmouseover="this.style.background='${color}22'" onmouseout="this.style.background='${color}11'">
            <div style="font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;color:${color};margin-bottom:5px">★ ${label.toUpperCase()}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--text)">$${Math.round(sal / 1000)}k/yr</div>
            <div style="font-size:10px;color:var(--muted);margin-top:2px">${trajs.length} path(s)</div>
        </div>`;
    });

    html += '</div></div>';
    container.innerHTML = html;
}

function highlightTerminalGroup(terminalId, groupIndex) {
    const trajs = AppState.terminalGroups[terminalId] || [];
    const color = terminalColors[groupIndex % terminalColors.length];
    if (trajs.length) {
        switchView('universe');
        setTimeout(() => U3D.highlightTrajectory(trajs[0].nodes, color), 100);
        toast(`Showing best path to ${terminalId}`, 'info');
    }
}

function _renderAllTrajectories(trajs) {
    const grid = document.getElementById('traj-grid');
    const frag = document.createDocumentFragment();

    trajs.forEach((t, i) => {
        const s = t.scores;
        const rnk = Math.min(t.pareto_rank, 2);
        const col = paletteColor(i);
        const rl = t.pareto_rank === 0 ? '★ OPTIMAL' : `RANK ${t.pareto_rank}`;
        const path = t.nodes.map(n => AppState.nodes.find(nd => nd.id === n)?.label || n).join(' → ');
        const termLabel = AppState.nodes.find(nd => nd.id === t.terminal_node)?.label || t.terminal_node;
        const simBadge = s.sim_success_mean
            ? `<div class="tc-m"><span class="tc-mv" style="color:var(--quasar)">${((s.sim_success_mean || 0) * 100).toFixed(0)}%</span><span class="tc-mk">sim</span></div>`
            : '';

        const card = document.createElement('div');
        card.className = `tcard${i === 0 ? ' best-card' : ''}`;
        card.style.setProperty('--cc', col);
        card.innerHTML = `
            <div class="tc-badge r${rnk}">${rl}</div>
            ${t.is_terminal_end ? `<div style="position:absolute;top:9px;left:6px;font-size:8px;color:var(--gold)">★ END</div>` : ''}
            <div class="tc-path">${path}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--gold);margin-bottom:6px">→ ${termLabel}</div>
            <div class="tc-mets">
                <div class="tc-m"><span class="tc-mv" style="color:var(--green)">+${((s.salary_growth || 0) * 100).toFixed(0)}%</span><span class="tc-mk">growth</span></div>
                <div class="tc-m"><span class="tc-mv" style="color:var(--gold)">$${Math.round((s.final_salary || 0) / 1000)}k</span><span class="tc-mk">final sal</span></div>
                <div class="tc-m"><span class="tc-mv">${(s.total_years || 0).toFixed(0)}yr</span><span class="tc-mk">duration</span></div>
                <div class="tc-m"><span class="tc-mv" style="color:${(s.avg_risk || 0) > .5 ? 'var(--red)' : 'var(--green)'}">${((s.avg_risk || 0) * 100).toFixed(0)}%</span><span class="tc-mk">risk</span></div>
                <div class="tc-m"><span class="tc-mv" style="color:var(--quasar)">${((s.avg_satisfaction || 0) * 100).toFixed(0)}%</span><span class="tc-mk">satisf.</span></div>
                <div class="tc-m"><span class="tc-mv">${((s.ml_success_prob || 0.5) * 100).toFixed(0)}%</span><span class="tc-mk">ML</span></div>
                ${simBadge}
            </div>
        `;
        card.onclick = () => {
            switchView('universe');
            const termIdx = AppState.terminalsFound.indexOf(t.terminal_node);
            const color = terminalColors[Math.max(termIdx, 0) % terminalColors.length];
            setTimeout(() => U3D.highlightTrajectory(t.nodes, color), 100);
            toast(`Highlighted: → ${termLabel}`, 'info');
        };
        frag.appendChild(card);
    });

    grid.appendChild(frag);
}

function _renderStepBreakdown(traj) {
    const sec = document.getElementById('step-sec');
    const grid = document.getElementById('step-grid');
    sec.style.display = 'block';
    grid.innerHTML = '';
    for (let i = 0; i < traj.nodes.length - 1; i++) {
        const fr = traj.nodes[i], to = traj.nodes[i + 1];
        const frN = AppState.nodes.find(n => n.id === fr);
        const toN = AppState.nodes.find(n => n.id === to);
        const edge = AppState.edges.find(e => e.from_node === fr && e.to_node === to);
        const card = document.createElement('div');
        card.className = 'step-card';
        card.innerHTML = `
            <div class="step-num">Decision ${i + 1} of ${traj.nodes.length - 1}</div>
            <div class="step-nodes">
                <span class="step-n src-n">${frN?.label || fr}</span>
                <span style="color:var(--muted);font-size:16px">→</span>
                <span class="step-n dst-n">${toN?.label || to}</span>
            </div>
            <div class="step-desc">
                ${edge ? `~${edge.transition_years} year${edge.transition_years !== 1 ? 's' : ''} · Difficulty ${(edge.difficulty * 100).toFixed(0)}% · Risk ${(edge.risk * 100).toFixed(0)}%.` : ''}
                ${toN ? `Salary → $${toN.avg_salary.toLocaleString()}/yr · ${(toN.satisfaction * 100).toFixed(0)}% satisfaction.` : ''}
            </div>
        `;
        grid.appendChild(card);
    }
}

// ── LLM Analysis ──────────────────────────────────────────────
async function requestAnalysis() {
    if (!AppState.trajectories.length) { toast('Run exploration first', 'err'); return; }

    const criterion = document.getElementById('crit-disp').value || document.getElementById('criterion').value;
    const userProfile = document.getElementById('user-profile').value;
    const llmEl = document.getElementById('llm-body');
    const btn = document.getElementById('analyze-btn');
    const provEl = document.getElementById('llm-provider');

    llmEl.textContent = '';
    llmEl.classList.add('llm-typing');
    btn.disabled = true;
    if (provEl) provEl.textContent = '';
    toast('Consulting AI...', 'info');

    try {
        const res = await API.analyze(AppState.trajectories.slice(0, 5), criterion, userProfile);
        llmEl.classList.remove('llm-typing');
        llmEl.textContent = res.analysis || 'Analysis unavailable.';
        if (provEl && res.provider_used) provEl.textContent = res.provider_used.toUpperCase();
        toast('🤖 AI analysis ready', 'ok');
    } catch (e) {
        llmEl.classList.remove('llm-typing');
        llmEl.textContent = '[Backend offline]\n\nTo enable AI analysis, run:\n  python -m backend.main_api\n\nThen click Analyze again.';
        toast('Backend offline — demo mode', 'info');
    } finally {
        btn.disabled = false;
    }
}

// ── Universe controls ─────────────────────────────────────────
function toggleRot() { const on = !U3D.autoRotate; U3D.setAutoRotate(on); document.getElementById('cb-rot').classList.toggle('on', on); }
function resetCam() { U3D.resetCamera(); }
function toggleAllTraj() {
    AppState.showAllTraj = !AppState.showAllTraj;
    document.getElementById('cb-traj').classList.toggle('on', AppState.showAllTraj);
    if (AppState.showAllTraj && AppState.trajectories.length) {
        AppState.trajectories.slice(0, 5).forEach((t, i) =>
            setTimeout(() => U3D.highlightTrajectory(t.nodes, terminalColors[i % terminalColors.length]), i * 280)
        );
    } else { U3D.clearTrajLines(); }
}
function toggleLbls() { const on = !U3D.showLbls; U3D.setShowLabels(on); document.getElementById('cb-lbl').classList.toggle('on', on); }
function doZoom(d) { U3D.zoom(d); }
function syncCrit(src) { const other = src === 'criterion' ? 'crit-disp' : 'criterion'; document.getElementById(other).value = document.getElementById(src).value; }

// ── Init ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
    SpaceBG.init();
    Preview.init();

    bindRange('bw', 'bw-v'); bindRange('md', 'md-v'); bindRange('tk', 'tk-v');
    bindRange('my', 'my-v'); bindRange('mr', 'mr-v', 2);
    bindRange('nn-dem', 'nn-dem-v', 2); bindRange('nn-sat', 'nn-sat-v', 2);
    bindRange('ee-diff', 'ee-diff-v', 2); bindRange('ee-risk', 'ee-risk-v', 2);

    document.getElementById('criterion').addEventListener('input', () => syncCrit('criterion'));
    document.getElementById('crit-disp').addEventListener('input', () => syncCrit('crit-disp'));

    setTimeout(async () => {
        const ld = document.getElementById('loading');
        ld.classList.add('out');
        setTimeout(() => ld.style.display = 'none', 800);
        await loadDefault();
        const status = await API.getLLMStatus();
        if (status?.key_count > 0) toast(`${status.key_count} API key(s) · ${status.active_provider}`, 'ok');
    }, 1600);
});

// ── Embedded default data ─────────────────────────────────────
function _getEmbeddedDefaultData() {
    return {
        nodes: [
            { id: "junior_dev", label: "Junior Dev", avg_salary: 25000, demand: .85, satisfaction: .65, years_experience: 1, skills: ["python", "git"], type: "role" },
            { id: "mid_dev", label: "Mid Developer", avg_salary: 45000, demand: .90, satisfaction: .72, years_experience: 3, skills: ["python", "design"], type: "role" },
            { id: "senior_dev", label: "Senior Dev", avg_salary: 75000, demand: .88, satisfaction: .78, years_experience: 6, skills: ["architecture", "cloud"], type: "role" },
            { id: "tech_lead", label: "Tech Lead", avg_salary: 90000, demand: .75, satisfaction: .80, years_experience: 8, skills: ["leadership"], type: "role" },
            { id: "data_scientist", label: "Data Scientist", avg_salary: 70000, demand: .92, satisfaction: .80, years_experience: 3, skills: ["ml", "python"], type: "role" },
            { id: "ml_engineer", label: "ML Engineer", avg_salary: 85000, demand: .94, satisfaction: .82, years_experience: 5, skills: ["mlops"], type: "role" },
            { id: "devops_engineer", label: "DevOps Eng", avg_salary: 72000, demand: .91, satisfaction: .74, years_experience: 4, skills: ["k8s", "docker"], type: "role" },
            { id: "engineering_manager", label: "Eng. Manager", avg_salary: 110000, demand: .65, satisfaction: .75, years_experience: 10, skills: ["hiring"], type: "role" },
            { id: "cto", label: "CTO", avg_salary: 180000, demand: .40, satisfaction: .85, years_experience: 15, skills: ["strategy"], type: "role" },
            { id: "freelancer", label: "Freelancer", avg_salary: 55000, demand: .70, satisfaction: .83, years_experience: 3, skills: ["self_mgmt"], type: "role" },
            { id: "startup_founder", label: "Founder", avg_salary: 30000, demand: .50, satisfaction: .88, years_experience: 5, skills: ["resilience"], type: "role" },
            { id: "product_manager", label: "Product Mgr", avg_salary: 95000, demand: .80, satisfaction: .77, years_experience: 5, skills: ["analytics"], type: "role" },
        ],
        edges: [
            { from: "junior_dev", to: "mid_dev", transition_years: 2, difficulty: .4, risk: .2 },
            { from: "junior_dev", to: "data_scientist", transition_years: 2, difficulty: .6, risk: .3 },
            { from: "junior_dev", to: "devops_engineer", transition_years: 2, difficulty: .5, risk: .25 },
            { from: "junior_dev", to: "freelancer", transition_years: 1, difficulty: .5, risk: .6 },
            { from: "mid_dev", to: "senior_dev", transition_years: 3, difficulty: .5, risk: .2 },
            { from: "mid_dev", to: "data_scientist", transition_years: 1, difficulty: .55, risk: .3 },
            { from: "mid_dev", to: "ml_engineer", transition_years: 2, difficulty: .65, risk: .3 },
            { from: "mid_dev", to: "freelancer", transition_years: 1, difficulty: .4, risk: .5 },
            { from: "senior_dev", to: "tech_lead", transition_years: 2, difficulty: .6, risk: .25 },
            { from: "senior_dev", to: "ml_engineer", transition_years: 1, difficulty: .6, risk: .25 },
            { from: "senior_dev", to: "product_manager", transition_years: 2, difficulty: .7, risk: .4 },
            { from: "senior_dev", to: "startup_founder", transition_years: 2, difficulty: .8, risk: .7 },
            { from: "tech_lead", to: "engineering_manager", transition_years: 2, difficulty: .55, risk: .2 },
            { from: "tech_lead", to: "cto", transition_years: 5, difficulty: .85, risk: .4 },
            { from: "data_scientist", to: "ml_engineer", transition_years: 2, difficulty: .5, risk: .2 },
            { from: "data_scientist", to: "product_manager", transition_years: 2, difficulty: .6, risk: .3 },
            { from: "ml_engineer", to: "cto", transition_years: 7, difficulty: .85, risk: .4 },
            { from: "devops_engineer", to: "tech_lead", transition_years: 3, difficulty: .65, risk: .3 },
            { from: "engineering_manager", to: "cto", transition_years: 4, difficulty: .8, risk: .35 },
            { from: "product_manager", to: "cto", transition_years: 6, difficulty: .8, risk: .4 },
            { from: "product_manager", to: "startup_founder", transition_years: 3, difficulty: .7, risk: .6 },
            { from: "freelancer", to: "startup_founder", transition_years: 2, difficulty: .65, risk: .65 },
        ],
    };
}
