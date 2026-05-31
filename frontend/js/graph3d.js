/**
 * graph3d.js
 * ----------
 * Módulo U3D: universo 3D con Three.js.
 * Gestiona nodos-estrella, aristas-wormhole, etiquetas,
 * trayectorias iluminadas y animación del beam search.
 *
 * Dependencias: Three.js (r128), universe.js
 * Exporta globalmente: U3D
 */

const U3D = (() => {
    // ── Privados ───────────────────────────────────────────────
    let scene, camera, renderer;
    let raycaster, mouse;
    const NM = {};   // nodeId → THREE.Mesh
    const EL = [];   // edge lines
    const TL = [];   // trajectory highlight lines
    const BL = [];   // beam search lines
    const LS = [];   // label sprites

    let phi = Math.PI / 2.5, theta = 0, radius = 30;
    let isDrag = false, pmx = 0, pmy = 0;
    let autoRot = true, showLbls = true;
    let t3 = 0;

    // Color palette (sync con ui.js)
    const PAL = [
        0x00c8ff, 0xb44dff, 0xffc843, 0xff6b35, 0x00ff8c, 0xff3355,
        0x4fc3f7, 0xce93d8, 0xffcc02, 0x69f0ae, 0xf48fb1, 0x80deea,
        0xff9800, 0x00bcd4, 0x9c27b0, 0x4caf50,
    ];

    // ── Init ──────────────────────────────────────────────────
    function init() {
        const cv = document.getElementById('three-canvas');
        if (!cv) return;

        scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x00000a, 0.007);

        camera = new THREE.PerspectiveCamera(55, cv.clientWidth / cv.clientHeight, 0.1, 800);
        _updCam();

        renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: false });
        renderer.setSize(cv.clientWidth, cv.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;

        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();

        // Lighting
        scene.add(new THREE.AmbientLight(0x0a1628, 3));
        const pl = new THREE.PointLight(0x00c8ff, 4, 100);
        pl.position.set(0, 0, 25);
        scene.add(pl);
        const pl2 = new THREE.PointLight(0xb44dff, 2, 80);
        pl2.position.set(-20, 10, -10);
        scene.add(pl2);

        _buildDeepSpaceParticles();
        _buildNebulaClouds();

        // Orbit
        cv.addEventListener('mousedown', e => { isDrag = true; pmx = e.clientX; pmy = e.clientY; });
        window.addEventListener('mouseup', () => isDrag = false);
        cv.addEventListener('mousemove', _onMouseMove);
        cv.addEventListener('click', _onMouseClick);
        cv.addEventListener('wheel', e => radius = Math.max(8, Math.min(70, radius + e.deltaY * 0.04)));
        window.addEventListener('resize', () => {
            if (!renderer) return;
            renderer.setSize(cv.clientWidth, cv.clientHeight);
            camera.aspect = cv.clientWidth / cv.clientHeight;
            camera.updateProjectionMatrix();
        });

        _animate();
    }

    // ── Background particles & nebula ─────────────────────────
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
            if (t < 0.55) { col[i * 3] = .55; col[i * 3 + 1] = .75; col[i * 3 + 2] = 1; }
            else if (t < 0.80) { col[i * 3] = 1; col[i * 3 + 1] = .9; col[i * 3 + 2] = .5; }
            else { col[i * 3] = .7; col[i * 3 + 1] = .4; col[i * 3 + 2] = 1; }
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

    // ── Node position (Fibonacci sphere) ─────────────────────
    function _nodePos(node, index, total) {
        const golden = Math.PI * (3 - Math.sqrt(5));
        const y = 1 - (index / Math.max(total - 1, 1)) * 2;
        const r2 = Math.sqrt(1 - y * y);
        const ang = golden * index;
        const sp = 10 + (node.avg_salary / 18000);
        return new THREE.Vector3(Math.cos(ang) * r2 * sp, y * sp, Math.sin(ang) * r2 * sp);
    }

    // ── Node mesh ─────────────────────────────────────────────
    function _addNode(node, index) {
        const total = AppState.nodes.length;
        const pos = _nodePos(node, index, total);
        const color = PAL[index % PAL.length];
        const size = 0.27 + (node.avg_salary / 200000) * 0.5;

        const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(size, 20, 20),
            new THREE.MeshStandardMaterial({
                color, emissive: color, emissiveIntensity: 0.55,
                metalness: 0.1, roughness: 0.15,
            })
        );
        mesh.position.copy(pos);
        mesh.userData = { nodeId: node.id, baseScale: 1, phase: index * 0.8, color };
        scene.add(mesh);
        NM[node.id] = mesh;

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

        // Label sprite
        if (showLbls) _addLabel(node, mesh, color);
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
        LS.push(sprite);
    }

    // ── Curved line helper ───────────────────────────────────
    function _curv(a, b, segments = 14) {
        const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
        mid.x += (Math.random() - 0.5) * 3;
        mid.y += (Math.random() - 0.5) * 3;
        return new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(segments);
    }

    // ── Animation loop ────────────────────────────────────────
    function _animate() {
        requestAnimationFrame(_animate);
        t3 += 0.005;
        if (autoRot) theta += 0.0025;
        _updCam();

        Object.values(NM).forEach(m => {
            const bs = m.userData.baseScale || 1;
            const pl = 1 + Math.sin(t3 * 2 + m.userData.phase) * 0.07;
            m.scale.setScalar(bs * pl);
            if (m.userData.glow) {
                m.userData.glow.material.opacity = 0.09 + Math.sin(t3 * 2 + m.userData.phase) * 0.04;
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

        const cv2 = renderer.domElement;
        const r = cv2.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
        mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const meshes = Object.values(NM);
        const hits = raycaster.intersectObjects(meshes);

        meshes.forEach(m => { if (!m.userData.selected) m.material.emissiveIntensity = 0.55; });
        document.getElementById('node-popup').classList.remove('show');
        cv2.style.cursor = 'default';

        if (hits.length) {
            hits[0].object.material.emissiveIntensity = 1.3;
            cv2.style.cursor = 'pointer';
            const node = AppState.nodes.find(n => n.id === hits[0].object.userData.nodeId);
            if (node) _showPopup(node, e.clientX, e.clientY);
        }
    }

    function _onMouseClick(e) {
        const cv2 = renderer.domElement;
        const r = cv2.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
        mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects(Object.values(NM));
        if (hits.length) {
            const node = AppState.nodes.find(n => n.id === hits[0].object.userData.nodeId);
            if (node) toast(`${node.label} · $${node.avg_salary.toLocaleString()}/yr · ${(node.satisfaction * 100).toFixed(0)}% satisfaction`, 'info');
        }
    }

    function _showPopup(node, x, y) {
        document.getElementById('np-title').textContent = node.label.toUpperCase();
        document.getElementById('np-body').innerHTML = `
      <div class="np-row"><span class="np-k">Salary</span><span class="np-v">$${node.avg_salary.toLocaleString()}/yr</span></div>
      <div class="np-row"><span class="np-k">Demand</span><span class="np-v">${(node.demand * 100).toFixed(0)}%</span></div>
      <div class="np-row"><span class="np-k">Satisfaction</span><span class="np-v">${(node.satisfaction * 100).toFixed(0)}%</span></div>
      <div class="np-row"><span class="np-k">Experience</span><span class="np-v">${node.years_experience} yrs</span></div>
      <div class="np-row"><span class="np-k">Skills</span><span class="np-v">${(node.skills || []).slice(0, 3).join(', ')}</span></div>
    `;
        const popup = document.getElementById('node-popup');
        popup.style.left = Math.min(x + 14, window.innerWidth - 245) + 'px';
        popup.style.top = Math.min(y + 14, window.innerHeight - 175) + 'px';
        popup.classList.add('show');
    }

    // ── Public API ────────────────────────────────────────────
    function rebuildAll() {
        Object.keys(NM).forEach(id => { scene.remove(NM[id]); delete NM[id]; });
        EL.forEach(l => scene.remove(l)); EL.length = 0;
        LS.length = 0;
        AppState.nodes.forEach((n, i) => _addNode(n, i));
        buildEdges();
    }

    function buildEdges() {
        EL.forEach(l => scene.remove(l)); EL.length = 0;
        AppState.edges.forEach(e => {
            const a = NM[e.from_node], b = NM[e.to_node];
            if (!a || !b) return;
            const line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(_curv(a.position, b.position, 14)),
                new THREE.LineBasicMaterial({ color: 0x0a2040, transparent: true, opacity: 0.22 + (0.6 - e.risk) * 0.3 })
            );
            scene.add(line); EL.push(line);
        });
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
            const line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(_curv(a.position, b.position, 22)),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 })
            );
            scene.add(line); TL.push(line);
        }
        nodes.forEach(id => {
            const m = NM[id];
            if (m) { m.userData.selected = true; m.userData.baseScale = 1.5; m.material.emissiveIntensity = 1.2; }
        });
    }

    function animateBeamStep(beam, completed) {
        clearBeamLines();
        beam.forEach(path => {
            for (let i = 0; i < path.length - 1; i++) {
                const a = NM[path[i]], b = NM[path[i + 1]];
                if (!a || !b) continue;
                const line = new THREE.Line(
                    new THREE.BufferGeometry().setFromPoints(_curv(a.position, b.position, 9)),
                    new THREE.LineBasicMaterial({ color: 0xb44dff, transparent: true, opacity: 0.78 })
                );
                scene.add(line); BL.push(line);
            }
        });
        // Flash last completed nodes
        completed.slice(-3).forEach(path => path.forEach(id => {
            const m = NM[id];
            if (m) {
                const b = m.userData.baseScale || 1;
                m.userData.baseScale = 1.8;
                setTimeout(() => m.userData.baseScale = b, 320);
            }
        }));
    }

    return {
        init,
        rebuildAll,
        buildEdges,
        clearTrajLines,
        clearBeamLines,
        highlightTrajectory,
        animateBeamStep,
        setAutoRotate: v => autoRot = v,
        setShowLabels: v => { showLbls = v; Object.values(NM).forEach(m => { if (m.userData.lbl) m.userData.lbl.visible = v; }); },
        resetCamera: () => { phi = Math.PI / 2.5; theta = 0; radius = 30; },
        zoom: d => radius = Math.max(8, Math.min(70, radius + d)),
        get autoRotate() { return autoRot; },
    };
})();
