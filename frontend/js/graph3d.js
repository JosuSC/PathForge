/**
 * graph3d.js
 * ----------
 * Universo 3D con Three.js — construcción PROGRESIVA del grafo.
 *
 */

const U3D = (() => {
    let scene, camera, renderer;
    let raycaster, mouse;

    const NM = {};   // nodeId → THREE.Mesh
    const EL = {};   // "from-to" → THREE.Line
    const TL = [];   // trajectory highlight lines
    const BL = [];   // beam search active lines

    const CURVE_CACHE = {};

    let phi = Math.PI / 2.5, theta = 0, radius = 30;
    let isDrag = false, pmx = 0, pmy = 0;
    let autoRot = true, showLbls = true;
    let t3 = 0;
    let terminalNodes = new Set();
    let sourceNodeId = null;

    const PAL = [
        0x00c8ff, 0xb44dff, 0xffc843, 0xff6b35, 0x00ff8c, 0xff3355,
        0x4fc3f7, 0xce93d8, 0xffcc02, 0x69f0ae, 0xf48fb1, 0x80deea,
        0xff9800, 0x00bcd4, 0x9c27b0, 0x4caf50,
    ];

    // ── Init ──────────────────────────────────────────────────
    function init() {
        const cv = document.getElementById('three-canvas');
        if (!cv) { console.error('❌ three-canvas not found'); return; }

        // ✅ FIX: Esperar a que el canvas tenga dimensiones válidas
        if (cv.clientWidth === 0 || cv.clientHeight === 0) {
            console.warn('⚠️ Canvas has 0 dimensions, retrying in 100ms...');
            setTimeout(() => init(), 100);
            return;
        }

        scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x00000a, 0.006);

        camera = new THREE.PerspectiveCamera(55, cv.clientWidth / cv.clientHeight, 0.1, 800);
        _updCam();

        try {
            renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: false });
            renderer.setSize(cv.clientWidth, cv.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.1;
            console.log('✅ WebGLRenderer created:', cv.clientWidth, 'x', cv.clientHeight);
        } catch (e) {
            console.error('❌ WebGLRenderer failed:', e);
            return;
        }

        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();

        // Lighting
        scene.add(new THREE.AmbientLight(0x0a1628, 3));
        const pl = new THREE.PointLight(0x00c8ff, 4, 100);
        pl.position.set(0, 0, 25); scene.add(pl);
        const pl2 = new THREE.PointLight(0xb44dff, 2, 80);
        pl2.position.set(-20, 10, -10); scene.add(pl2);

        _buildDeepSpaceParticles();
        _buildNebulaClouds();

        // Events
        cv.addEventListener('mousedown', e => { isDrag = true; pmx = e.clientX; pmy = e.clientY; });
        window.addEventListener('mouseup', () => isDrag = false);
        cv.addEventListener('mousemove', _onMouseMove);
        cv.addEventListener('click', _onMouseClick);
        cv.addEventListener('wheel', e => { radius = Math.max(8, Math.min(70, radius + e.deltaY * 0.04)); });
        window.addEventListener('resize', _onResize);

        // ✅ FIX: ResizeObserver para detectar cambios de tamaño del contenedor
        if (typeof ResizeObserver !== 'undefined') {
            const ro = new ResizeObserver(() => _onResize());
            ro.observe(cv);
        }

        _animate();
        console.log('✅ U3D initialized successfully');
    }

    // ✅ FIX: Función resize() exportable para forzar redimensionamiento
    function resize() {
        if (!renderer || !camera) return;
        const cv = renderer.domElement;
        if (cv.clientWidth === 0 || cv.clientHeight === 0) return;
        renderer.setSize(cv.clientWidth, cv.clientHeight);
        camera.aspect = cv.clientWidth / cv.clientHeight;
        camera.updateProjectionMatrix();
    }

    function _onResize() {
        resize();
    }

    // ── Background ────────────────────────────────────────────
    function _buildDeepSpaceParticles() {
        const n = 4000;
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(n * 3);
        const col = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
            pos[i * 3] = (Math.random() - 0.5) * 600;
            pos[i * 3 + 1] = (Math.random() - 0.5) * 600;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 600;
            const t = Math.random();
            if (t < 0.55) { col[i * 3] = .55; col[i * 3 + 1] = .75; col[i * 3 + 2] = 1.0; }
            else if (t < 0.80) { col[i * 3] = 1.0; col[i * 3 + 1] = .90; col[i * 3 + 2] = .50; }
            else { col[i * 3] = .70; col[i * 3 + 1] = .40; col[i * 3 + 2] = 1.0; }
        }
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
            size: 0.15, vertexColors: true, transparent: true, opacity: 0.75, sizeAttenuation: true,
        })));
    }

    function _buildNebulaClouds() {
        [[0x0a1628, .06], [0x1a0a3a, .05], [0x0a2a1a, .04]].forEach(([cl, op]) => {
            const m = new THREE.Mesh(
                new THREE.PlaneGeometry(200, 140),
                new THREE.MeshBasicMaterial({ color: cl, transparent: true, opacity: op, side: THREE.DoubleSide })
            );
            m.position.set((Math.random() - 0.5) * 50, (Math.random() - 0.5) * 30, -60 - Math.random() * 40);
            m.rotation.z = Math.random() * Math.PI;
            scene.add(m);
        });
    }

    // ── Camera ────────────────────────────────────────────────
    function _updCam() {
        camera.position.set(
            radius * Math.sin(phi) * Math.sin(theta),
            radius * Math.cos(phi),
            radius * Math.sin(phi) * Math.cos(theta)
        );
        camera.lookAt(0, 0, 0);
    }

    // ── Node positioning: Fibonacci sphere ────────────────────
    function _nodePos(nodeId) {
        const nodes = AppState.nodes;
        const idx = nodes.findIndex(n => n.id === nodeId);
        const total = nodes.length;
        const node = nodes[idx] || { avg_salary: 50000 };

        const golden = Math.PI * (3 - Math.sqrt(5));
        const y = 1 - (idx / Math.max(total - 1, 1)) * 2;
        const r2 = Math.sqrt(1 - y * y);
        const ang = golden * idx;
        const sp = 10 + (node.avg_salary / 18000);
        return new THREE.Vector3(Math.cos(ang) * r2 * sp, y * sp, Math.sin(ang) * r2 * sp);
    }

    // ── Deterministic curve ───────────────────────────────────
    function _curv(a, b, edgeKey, segments = 18) {
        if (CURVE_CACHE[edgeKey]) return CURVE_CACHE[edgeKey];

        let hash = 0;
        for (let i = 0; i < edgeKey.length; i++) {
            hash = ((hash << 5) - hash) + edgeKey.charCodeAt(i);
            hash |= 0;
        }
        const ox = ((hash % 100) / 100 - 0.5) * 4;
        const oy = (((hash >> 8) % 100) / 100 - 0.5) * 4;

        const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
        mid.x += ox;
        mid.y += oy;
        const points = new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(segments);
        CURVE_CACHE[edgeKey] = points;
        return points;
    }

    // ── Add a single node ─────────────────────────────────────
    function addNodeToScene(nodeId, options = {}) {
        if (NM[nodeId]) return;

        const idx = AppState.nodes.findIndex(n => n.id === nodeId);
        const node = AppState.nodes[idx] || { id: nodeId, label: nodeId, avg_salary: 50000, satisfaction: 0.7, demand: 0.7, years_experience: 1, skills: [] };
        const color = PAL[Math.max(idx, 0) % PAL.length];
        const pos = _nodePos(nodeId);

        const isTerminal = terminalNodes.has(nodeId);
        const isSource = nodeId === sourceNodeId;
        let size = 0.27 + (node.avg_salary / 200000) * 0.5;
        if (isTerminal) size *= 1.4;
        if (isSource) size *= 1.2;

        const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(size, 20, 20),
            new THREE.MeshStandardMaterial({
                color, emissive: color, emissiveIntensity: 0.55,
                metalness: 0.1, roughness: 0.15,
            })
        );
        mesh.position.copy(pos);
        mesh.userData = { nodeId, baseScale: 0.01, targetScale: 1, phase: idx * 0.8, color, isTerminal, isSource };
        scene.add(mesh);
        NM[nodeId] = mesh;

        // Glow halo
        const glow = new THREE.Mesh(
            new THREE.SphereGeometry(size * 3, 16, 16),
            new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.11, side: THREE.BackSide })
        );
        mesh.add(glow);
        mesh.userData.glow = glow;

        // Star spikes
        const spkMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 });
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach(ax => {
            mesh.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(-ax[0] * size * 4, -ax[1] * size * 4, -ax[2] * size * 4),
                    new THREE.Vector3(ax[0] * size * 4, ax[1] * size * 4, ax[2] * size * 4),
                ]),
                spkMat
            ));
        });

        // Terminal ring
        if (isTerminal) {
            const ringGeo = new THREE.TorusGeometry(size * 2.2, 0.05, 8, 32);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0xffd700, transparent: true, opacity: 0.7 });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            mesh.add(ring);
            mesh.userData.ring = ring;
        }

        // Source crown
        if (isSource) {
            const crownGeo = new THREE.TorusGeometry(size * 1.8, 0.07, 8, 32);
            const crownMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
            const crown = new THREE.Mesh(crownGeo, crownMat);
            crown.rotation.x = Math.PI / 2;
            mesh.add(crown);
        }

        if (showLbls) _addLabel(node, mesh, color);

        mesh.userData.baseScale = 0.01;
        mesh.userData.targetScale = 1;
    }

    function _addLabel(node, mesh, color) {
        const cv2 = document.createElement('canvas');
        cv2.width = 256; cv2.height = 60;
        const c = cv2.getContext('2d');
        c.clearRect(0, 0, 256, 60);
        c.font = 'bold 20px Rajdhani, sans-serif';
        c.fillStyle = 'rgba(0,0,0,.5)';
        c.fillRect(0, 18, 256, 28);
        c.fillStyle = '#' + color.toString(16).padStart(6, '0');
        c.textAlign = 'center';
        c.fillText(node.label, 128, 38);
        const sprite = new THREE.Sprite(
            new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv2), transparent: true })
        );
        const s = mesh.geometry.parameters.radius;
        sprite.scale.set(3.8, 0.95, 1);
        sprite.position.set(0, s + 0.75, 0);
        mesh.add(sprite);
        mesh.userData.lbl = sprite;
    }

    // ── Add a single edge ─────────────────────────────────────
    function addEdgeToScene(fromId, toId) {
        const key = `${fromId}-${toId}`;
        if (EL[key]) return;

        const a = NM[fromId], b = NM[toId];
        if (!a || !b) return;

        const edgeData = AppState.edges.find(e => e.from_node === fromId && e.to_node === toId);
        const risk = edgeData?.risk || 0.3;
        const opacity = 0.22 + (0.6 - risk) * 0.3;

        const points = _curv(a.position, b.position, key, 18);
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color: 0x0a3060, transparent: true, opacity })
        );
        scene.add(line);
        EL[key] = line;

        line.material.opacity = 0;
        const targetOpacity = opacity;
        let progress = 0;
        const fadeIn = () => {
            progress += 0.05;
            line.material.opacity = Math.min(progress, targetOpacity);
            if (progress < targetOpacity) requestAnimationFrame(fadeIn);
        };
        requestAnimationFrame(fadeIn);
    }

    // ── Animation loop ────────────────────────────────────────
    function _animate() {
        requestAnimationFrame(_animate);
        t3 += 0.005;
        if (autoRot) theta += 0.0025;
        _updCam();

        Object.values(NM).forEach(m => {
            if (m.userData.targetScale && m.userData.baseScale < m.userData.targetScale) {
                m.userData.baseScale = Math.min(
                    m.userData.baseScale + 0.08,
                    m.userData.targetScale
                );
            }

            const bs = m.userData.baseScale || 1;
            const pl = 1 + Math.sin(t3 * 2 + m.userData.phase) * 0.07;
            m.scale.setScalar(bs * pl);

            if (m.userData.glow) {
                m.userData.glow.material.opacity = 0.09 + Math.sin(t3 * 2 + m.userData.phase) * 0.04;
            }

            if (m.userData.ring) {
                m.userData.ring.rotation.z += 0.01;
                m.userData.ring.material.opacity = 0.5 + Math.sin(t3 * 3 + m.userData.phase) * 0.2;
            }
        });

        renderer && renderer.render(scene, camera);
    }

    // ── Mouse interaction ─────────────────────────────────────
    function _onMouseMove(e) {
        if (isDrag) {
            const dx = (e.clientX - pmx) * 0.005;
            const dy = (e.clientY - pmy) * 0.005;
            theta -= dx;
            phi = Math.max(0.2, Math.min(Math.PI - 0.2, phi + dy));
            pmx = e.clientX; pmy = e.clientY;
        }

        if (!renderer) return;
        const cv2 = renderer.domElement;
        const r = cv2.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
        mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const meshes = Object.values(NM);
        const hits = raycaster.intersectObjects(meshes);

        meshes.forEach(m => { if (!m.userData.selected) m.material.emissiveIntensity = 0.55; });
        const popup = document.getElementById('node-popup');
        if (popup) popup.classList.remove('show');
        cv2.style.cursor = 'default';

        if (hits.length) {
            hits[0].object.material.emissiveIntensity = 1.3;
            cv2.style.cursor = 'pointer';
            const node = AppState.nodes.find(n => n.id === hits[0].object.userData.nodeId);
            if (node) _showPopup(node, e.clientX, e.clientY);
        }
    }

    function _onMouseClick(e) {
        if (!renderer) return;
        const cv2 = renderer.domElement;
        const r = cv2.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
        mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects(Object.values(NM));
        if (hits.length) {
            const node = AppState.nodes.find(n => n.id === hits[0].object.userData.nodeId);
            if (node) {
                const isT = terminalNodes.has(node.id) ? ' [TERMINAL]' : '';
                toast(`${node.label}${isT} · $${node.avg_salary.toLocaleString()}/yr · ${(node.satisfaction * 100).toFixed(0)}% satisfaction`, 'info');
            }
        }
    }

    function _showPopup(node, x, y) {
        const popup = document.getElementById('node-popup');
        if (!popup) return;
        const isT = terminalNodes.has(node.id);
        document.getElementById('np-title').textContent = node.label.toUpperCase() + (isT ? ' ★' : '');
        document.getElementById('np-body').innerHTML = `
            <div class="np-row"><span class="np-k">Type</span><span class="np-v" style="color:${isT ? 'var(--gold)' : 'var(--pulsar)'}">${isT ? 'TERMINAL ★' : 'Waypoint'}</span></div>
            <div class="np-row"><span class="np-k">Salary</span><span class="np-v">$${node.avg_salary.toLocaleString()}/yr</span></div>
            <div class="np-row"><span class="np-k">Demand</span><span class="np-v">${(node.demand * 100).toFixed(0)}%</span></div>
            <div class="np-row"><span class="np-k">Satisfaction</span><span class="np-v">${(node.satisfaction * 100).toFixed(0)}%</span></div>
            <div class="np-row"><span class="np-k">Experience</span><span class="np-v">${node.years_experience} yrs</span></div>
            <div class="np-row"><span class="np-k">Skills</span><span class="np-v">${(node.skills || []).slice(0, 3).join(', ')}</span></div>
        `;
        popup.style.left = Math.min(x + 14, window.innerWidth - 245) + 'px';
        popup.style.top = Math.min(y + 14, window.innerHeight - 200) + 'px';
        popup.classList.add('show');
    }

    // ── Public API ────────────────────────────────────────────
    function clearGraph() {
        Object.values(NM).forEach(m => scene.remove(m));
        Object.keys(NM).forEach(k => delete NM[k]);
        Object.values(EL).forEach(l => scene.remove(l));
        Object.keys(EL).forEach(k => delete EL[k]);
        Object.keys(CURVE_CACHE).forEach(k => delete CURVE_CACHE[k]);
        clearTrajLines();
        clearBeamLines();
    }

    function showSourceNode(nodeId, terminals = []) {
        clearGraph();
        sourceNodeId = nodeId;
        terminalNodes = new Set(terminals);
        addNodeToScene(nodeId);
    }

    function onBeamStep(step) {
        (step.new_nodes || []).forEach(nodeId => {
            addNodeToScene(nodeId);
            const m = NM[nodeId];
            if (m) {
                m.material.emissiveIntensity = 2.0;
                setTimeout(() => { if (m) m.material.emissiveIntensity = 0.55; }, 400);
            }
        });

        (step.new_edges || []).forEach(([fromId, toId]) => {
            addEdgeToScene(fromId, toId);
        });

        clearBeamLines();
        (step.beam || []).forEach(path => {
            for (let i = 0; i < path.length - 1; i++) {
                const a = NM[path[i]], b = NM[path[i + 1]];
                if (!a || !b) continue;
                const pts = _curv(a.position, b.position, `beam-${path[i]}-${path[i + 1]}`, 10);
                const line = new THREE.Line(
                    new THREE.BufferGeometry().setFromPoints(pts),
                    new THREE.LineBasicMaterial({ color: 0xb44dff, transparent: true, opacity: 0.85 })
                );
                scene.add(line);
                BL.push(line);
            }
        });

        if (step.terminal_reached) {
            const m = NM[step.terminal_reached];
            if (m) {
                m.userData.baseScale = 2.5;
                setTimeout(() => { if (m) m.userData.baseScale = 1.5; }, 600);
            }
        }
    }

    function rebuildAll() {
        clearGraph();
        AppState.nodes.forEach((n) => addNodeToScene(n.id));
        AppState.edges.forEach(e => addEdgeToScene(e.from_node, e.to_node));
    }

    function clearTrajLines() {
        TL.forEach(l => scene.remove(l)); TL.length = 0;
        Object.values(NM).forEach(m => {
            m.userData.selected = false;
            m.userData.baseScale = 1;
            m.material.emissiveIntensity = 0.55;
        });
    }

    function clearBeamLines() {
        BL.forEach(l => scene.remove(l)); BL.length = 0;
    }

    function highlightTrajectory(nodes, color = 0xffd700) {
        clearTrajLines();
        for (let i = 0; i < nodes.length - 1; i++) {
            const a = NM[nodes[i]], b = NM[nodes[i + 1]];
            if (!a || !b) continue;
            const pts = _curv(a.position, b.position, `traj-${nodes[i]}-${nodes[i + 1]}`, 24);
            const line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(pts),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 })
            );
            scene.add(line); TL.push(line);
        }
        nodes.forEach(id => {
            const m = NM[id];
            if (m) { m.userData.selected = true; m.userData.baseScale = 1.5; m.material.emissiveIntensity = 1.2; }
        });
    }

    return {
        init,
        resize,  // ✅ FIX: Nueva función exportada
        clearGraph,
        showSourceNode,
        onBeamStep,
        rebuildAll,
        clearTrajLines,
        clearBeamLines,
        highlightTrajectory,
        addNodeToScene,
        addEdgeToScene,
        setAutoRotate: v => autoRot = v,
        setShowLabels: v => { showLbls = v; Object.values(NM).forEach(m => { if (m.userData.lbl) m.userData.lbl.visible = v; }); },
        setTerminals: t => terminalNodes = new Set(t),
        resetCamera: () => { phi = Math.PI / 2.5; theta = 0; radius = 30; },
        zoom: d => radius = Math.max(8, Math.min(70, radius + d)),
        get autoRotate() { return autoRot; },
    };
})();
