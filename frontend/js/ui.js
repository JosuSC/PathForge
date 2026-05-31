/**
 * ui.js
 * -----
 * Módulo principal de la aplicación.
 * Gestiona: estado global, lógica de exploración,
 * renderizado de análisis, helpers UI, e inicialización.
 *
 * Depende de: universe.js, graph3d.js, animation.js, websocket.js
 */

// ── Global App State ──────────────────────────────────────────
const AppState = {
    nodes: [],
    edges: [],
    trajectories: [],
    exploring: false,
    showAllTraj: false,
    currentView: 'setup',
};

// Color palette (shared across modules)
const PALETTE = [
    0x00c8ff, 0xb44dff, 0xffc843, 0xff6b35, 0x00ff8c, 0xff3355,
    0x4fc3f7, 0xce93d8, 0xffcc02, 0x69f0ae, 0xf48fb1, 0x80deea,
    0xff9800, 0x00bcd4, 0x9c27b0, 0x4caf50,
];
const paletteColor = i => '#' + PALETTE[i % PALETTE.length].toString(16).padStart(6, '0');

// ── Toast notifications ───────────────────────────────────────
function toast(msg, type = 'info') {
    const container = document.getElementById('toasts');
    const el = document.createElement('div');
    el.className = `toast t-${type === 'ok' ? 'ok' : type === 'err' || type === 'error' ? 'err' : 'info'}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity .4s';
        setTimeout(() => el.remove(), 400);
    }, 3200);
}

// ── Stats bar ─────────────────────────────────────────────────
function updateStats() {
    document.getElementById('hs-n').textContent = AppState.nodes.length;
    document.getElementById('hs-e').textContent = AppState.edges.length;
    document.getElementById('hs-f').textContent = AppState.trajectories.length;
    document.getElementById('hs-o').textContent = AppState.trajectories.filter(t => t.pareto_rank === 0).length;
}

// ── Node & Edge lists ─────────────────────────────────────────
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

        // Sidebar list item
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

        // Source selector
        sel.innerHTML += `<option value="${node.id}"${node.id === curSrc ? ' selected' : ''}>${node.label}</option>`;

        // Edge modal selectors
        if (ef) {
            ef.innerHTML += `<option value="${node.id}">${node.label}</option>`;
            et.innerHTML += `<option value="${node.id}">${node.label}</option>`;
        }
    });
}

function renderEdgeList() {
    const el = document.getElementById('edge-list');
    el.innerHTML = '';
    if (!AppState.edges.length) {
        el.innerHTML = '<div class="empty-msg">No wormholes yet</div>';
        return;
    }
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
        setTimeout(() => {
            if (!U3D._init) { U3D.init(); U3D._init = true; }
            U3D.rebuildAll();
        }, 60);
    }
}

// ── Modals ────────────────────────────────────────────────────
function openModal(id) {
    if (id === 'modal-edge') renderNodeList(); // refresh selects
    document.getElementById(id).classList.add('show');
}
function closeModal(id) {
    document.getElementById(id).classList.remove('show');
}

// ── Range binding ─────────────────────────────────────────────
function bindRange(id, valueId, decimals = 0) {
    const el = document.getElementById(id);
    const vEl = document.getElementById(valueId);
    if (!el || !vEl) return;
    const fmt = v => decimals ? parseFloat(v).toFixed(decimals) : v;
    vEl.textContent = fmt(el.value);
    el.addEventListener('input', () => vEl.textContent = fmt(el.value));
}

// ── CRUD: nodes & edges ───────────────────────────────────────
function confirmAddNode() {
    const id = document.getElementById('nn-id').value.trim().replace(/\s+/g, '_');
    const label = document.getElementById('nn-label').value.trim();
    const salary = parseFloat(document.getElementById('nn-salary').value) || 50000;
    const demand = parseFloat(document.getElementById('nn-dem').value);
    const sat = parseFloat(document.getElementById('nn-sat').value);
    const exp = parseInt(document.getElementById('nn-exp').value) || 0;
    const skills = document.getElementById('nn-skills').value.split(',').map(s => s.trim()).filter(Boolean);

    if (!id || !label) { toast('ID and Label are required', 'err'); return; }
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
        // Embedded demo data (stays in sync with backend/data/careers.json)
        data = {
            nodes: [
                { id: "junior_dev", label: "Junior Dev", avg_salary: 25000, demand: .85, satisfaction: .65, years_experience: 1, skills: ["python", "git"], type: "role" },
                { id: "mid_dev", label: "Mid Developer", avg_salary: 45000, demand: .90, satisfaction: .72, years_experience: 3, skills: ["python", "design"], type: "role" },
                { id: "senior_dev", label: "Senior Dev", avg_salary: 75000, demand: .88, satisfaction: .78, years_experience: 6, skills: ["architecture"], type: "role" },
                { id: "tech_lead", label: "Tech Lead", avg_salary: 90000, demand: .75, satisfaction: .80, years_experience: 8, skills: ["leadership"], type: "role" },
                { id: "data_scientist", label: "Data Scientist", avg_salary: 70000, demand: .92, satisfaction: .80, years_experience: 3, skills: ["ml", "python"], type: "role" },
                { id: "ml_engineer", label: "ML Engineer", avg_salary: 85000, demand: .94, satisfaction: .82, years_experience: 5, skills: ["mlops", "cloud"], type: "role" },
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

    AppState.nodes = data.nodes;
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
    switchView('universe');
    setTimeout(() => {
        if (!U3D._init) { U3D.init(); U3D._init = true; }
        U3D.rebuildAll();
        setTimeout(_startSearch, 200);
    }, 100);
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
    };

    // ── Callbacks ──────────────────────────────────────────────
    function onStep(msg) {
        const pct = Math.min(98, (msg.depth / maxD) * 100);
        fill.style.width = pct + '%';
        sub.textContent = `Beam Search · Depth ${msg.depth} of ${maxD}`;
        stat.textContent = `${msg.beam.length} active beams · ${msg.completed.length} paths discovered`;

        const dd = document.getElementById(`sod-${msg.depth - 1}`);
        if (dd) dd.classList.add('on');

        document.getElementById('sb-st').textContent = 'Exploring';
        document.getElementById('sb-d').textContent = msg.depth;
        document.getElementById('sb-b').textContent = msg.beam.length;
        document.getElementById('sb-p').textContent = msg.completed.length;

        U3D.animateBeamStep(msg.beam, msg.completed);
    }

    function onResult(msg) {
        AppState.trajectories = msg.trajectories || [];
        fill.style.width = '100%';
        sub.textContent = 'Processing results...';
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
        toast(`🌌 ${AppState.trajectories.length} futures discovered!`, 'ok');

        // Doctor Strange moment: cycle top 3 then settle on best in gold
        const top = AppState.trajectories.slice(0, Math.min(3, AppState.trajectories.length));
        const cols = [0xffd700, 0xb44dff, 0x00c8ff];
        let i = 0;
        const cycle = setInterval(() => {
            if (i >= top.length) {
                clearInterval(cycle);
                if (AppState.trajectories.length) {
                    U3D.highlightTrajectory(AppState.trajectories[0].nodes, 0xffd700);
                    toast('🏆 Best trajectory highlighted in gold', 'ok');
                }
                return;
            }
            U3D.highlightTrajectory(top[i].nodes, cols[i % 3]);
            i++;
        }, 1100);
    }

    function onError(msg) {
        console.warn('WebSocket error, switching to demo mode:', msg);
        runDemoExploration(request, onStep, onResult, onDone);
    }

    // Try real backend first, fall back to demo
    try {
        API.exploreWS(request, onStep, onResult, onDone, onError);
    } catch (e) {
        runDemoExploration(request, onStep, onResult, onDone);
    }
}

// ── Analysis rendering ────────────────────────────────────────
function renderAnalysis() {
    const trajs = AppState.trajectories;
    document.getElementById('traj-count').textContent =
        `(${trajs.length} total · ${trajs.filter(t => t.pareto_rank === 0).length} optimal)`;

    // Best trajectory section
    if (trajs.length) {
        const best = trajs[0];
        document.getElementById('best-sec').style.display = 'block';

        // Path nodes
        document.getElementById('bt-path').innerHTML = best.nodes.map((n, i) => `
      ${i > 0 ? '<span class="bt-arrow">→</span>' : ''}
      <span class="bt-node">${AppState.nodes.find(nd => nd.id === n)?.label || n}</span>
    `).join('');

        // Metrics
        const s = best.scores;
        document.getElementById('bt-mets').innerHTML = `
      <div class="bt-m"><div class="bt-mv">+${((s.salary_growth || 0) * 100).toFixed(0)}%</div><div class="bt-mk">Salary Growth</div></div>
      <div class="bt-m"><div class="bt-mv">$${(s.final_salary || 0).toLocaleString('en', { maximumFractionDigits: 0 })}</div><div class="bt-mk">Final Salary</div></div>
      <div class="bt-m"><div class="bt-mv">${(s.total_years || 0).toFixed(0)} yr</div><div class="bt-mk">Duration</div></div>
      <div class="bt-m"><div class="bt-mv">${((s.avg_satisfaction || 0) * 100).toFixed(0)}%</div><div class="bt-mk">Satisfaction</div></div>
    `;

        _renderStepBreakdown(best);
    }

    // All trajectories grid
    const grid = document.getElementById('traj-grid');
    const cols = [0xffd700, 0xb44dff, 0x00c8ff, 0xff6b35, 0x00ff8c, 0xff3355];
    grid.innerHTML = '';

    trajs.forEach((t, i) => {
        const s = t.scores;
        const rnk = Math.min(t.pareto_rank, 2);
        const col = paletteColor(i);
        const rl = t.pareto_rank === 0 ? '★ OPTIMAL' : `RANK ${t.pareto_rank}`;
        const path = t.nodes.map(n => AppState.nodes.find(nd => nd.id === n)?.label || n).join(' → ');

        const card = document.createElement('div');
        card.className = `tcard${i === 0 ? ' best-card' : ''}`;
        card.style.setProperty('--cc', col);
        card.innerHTML = `
      <div class="tc-badge r${rnk}">${rl}</div>
      <div class="tc-path">${path}</div>
      <div class="tc-mets">
        <div class="tc-m"><span class="tc-mv" style="color:var(--green)">+${((s.salary_growth || 0) * 100).toFixed(0)}%</span><span class="tc-mk">growth</span></div>
        <div class="tc-m"><span class="tc-mv" style="color:var(--gold)">$${Math.round((s.final_salary || 0) / 1000)}k</span><span class="tc-mk">final sal</span></div>
        <div class="tc-m"><span class="tc-mv">${(s.total_years || 0).toFixed(0)}yr</span><span class="tc-mk">duration</span></div>
        <div class="tc-m"><span class="tc-mv" style="color:${(s.avg_risk || 0) > .5 ? 'var(--red)' : 'var(--green)'}">${((s.avg_risk || 0) * 100).toFixed(0)}%</span><span class="tc-mk">risk</span></div>
        <div class="tc-m"><span class="tc-mv" style="color:var(--quasar)">${((s.avg_satisfaction || 0) * 100).toFixed(0)}%</span><span class="tc-mk">satisf.</span></div>
        <div class="tc-m"><span class="tc-mv">${((s.avg_demand || 0) * 100).toFixed(0)}%</span><span class="tc-mk">demand</span></div>
      </div>
    `;
        card.onclick = () => {
            switchView('universe');
            setTimeout(() => U3D.highlightTrajectory(t.nodes, cols[i % cols.length]), 100);
            toast(`Highlighted: ${t.nodes.slice(0, 3).join(' → ')}...`, 'info');
        };
        grid.appendChild(card);
    });
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
        ${toN ? `Salary becomes $${toN.avg_salary.toLocaleString()}/yr · ${(toN.satisfaction * 100).toFixed(0)}% satisfaction.` : ''}
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
function toggleRot() {
    const on = !U3D.autoRotate;
    U3D.setAutoRotate(on);
    document.getElementById('cb-rot').classList.toggle('on', on);
}
function resetCam() { U3D.resetCamera(); }
function toggleAllTraj() {
    AppState.showAllTraj = !AppState.showAllTraj;
    document.getElementById('cb-traj').classList.toggle('on', AppState.showAllTraj);
    if (AppState.showAllTraj && AppState.trajectories.length) {
        const cols = [0xffd700, 0xb44dff, 0x00c8ff, 0xff6b35, 0x00ff8c];
        AppState.trajectories.slice(0, 5).forEach((t, i) => setTimeout(() => U3D.highlightTrajectory(t.nodes, cols[i]), i * 280));
    } else {
        U3D.clearTrajLines();
    }
}
function toggleLbls() {
    const on = !U3D.showLbls;
    U3D.setShowLabels(on);
    document.getElementById('cb-lbl').classList.toggle('on', on);
}
function doZoom(d) { U3D.zoom(d); }

// Sync criterion between Setup and Analysis tabs
function syncCrit(src) {
    const other = src === 'criterion' ? 'crit-disp' : 'criterion';
    document.getElementById(other).value = document.getElementById(src).value;
}

// ── Init ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
    // Start background universe
    SpaceBG.init();
    // Start 2D preview
    Preview.init();

    // Bind all range sliders
    bindRange('bw', 'bw-v');
    bindRange('md', 'md-v');
    bindRange('tk', 'tk-v');
    bindRange('my', 'my-v');
    bindRange('mr', 'mr-v', 2);
    bindRange('nn-dem', 'nn-dem-v', 2);
    bindRange('nn-sat', 'nn-sat-v', 2);
    bindRange('ee-diff', 'ee-diff-v', 2);
    bindRange('ee-risk', 'ee-risk-v', 2);

    // Sync criterion fields
    document.getElementById('criterion').addEventListener('input', () => syncCrit('criterion'));
    document.getElementById('crit-disp').addEventListener('input', () => syncCrit('crit-disp'));

    // Fade out loading screen and load default graph
    setTimeout(async () => {
        const ld = document.getElementById('loading');
        ld.classList.add('out');
        setTimeout(() => ld.style.display = 'none', 800);
        await loadDefault();

        // Check LLM status and show in UI
        const status = await API.getLLMStatus();
        if (status && status.key_count > 0) {
            toast(`${status.key_count} API key(s) loaded · Active: ${status.active_provider}`, 'ok');
        }
    }, 1600);
});
