/**
 * Intelligent CC Suggestion Tool — Frontend
 */

let currentJobId = null;
let allEvents = [];
let videoDuration = 0;
let currentFilter = 'all';

// ── Upload ──

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please upload a video file.');
        return;
    }

    showSection('processing');
    updateProgress(2, `Uploading ${file.name}…`);

    try {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch('/api/upload', { method: 'POST', body: form });
        if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');

        const data = await res.json();
        currentJobId = data.job_id;
        updateProgress(5, 'Starting pipeline…');

        await fetch(`/api/process/${currentJobId}`, { method: 'POST' });
        pollStatus();
    } catch (err) {
        alert(err.message);
        showSection('upload');
    }
}

// ── Status Polling ──

async function pollStatus() {
    if (!currentJobId) return;
    try {
        const res = await fetch(`/api/status/${currentJobId}`);
        const data = await res.json();
        updateProgress(data.progress, data.stage);

        if (data.status === 'complete') await loadResults();
        else if (data.status === 'error') { alert(`Error: ${data.stage}`); showSection('upload'); }
        else setTimeout(pollStatus, 700);
    } catch { setTimeout(pollStatus, 2000); }
}

function updateProgress(pct, stage) {
    document.getElementById('progress-percent').textContent = `${pct}%`;
    document.getElementById('progress-bar').style.width = `${pct}%`;
    document.getElementById('progress-stage').textContent = stage || '';
}

// ── Results ──

async function loadResults() {
    const res = await fetch(`/api/events/${currentJobId}`);
    const data = await res.json();
    allEvents = data.events;

    const video = document.getElementById('video-player');
    video.src = `/api/video/${currentJobId}`;
    video.addEventListener('loadedmetadata', () => { videoDuration = video.duration; renderTimeline(); });
    video.addEventListener('timeupdate', updatePlayhead);

    updateStats();
    renderEvents();
    renderSRT();
    showSection('results');
    document.getElementById('status-badge').textContent = `${allEvents.filter(e => e.accepted).length} captions`;
}

function updateStats() {
    const acc = allEvents.filter(e => e.accepted).length;
    const rej = allEvents.length - acc;
    const rate = allEvents.length ? Math.round((rej / allEvents.length) * 100) : 0;
    document.getElementById('stat-total').textContent = allEvents.length;
    document.getElementById('stat-accepted').textContent = acc;
    document.getElementById('stat-rejected').textContent = rej;
    document.getElementById('stat-rate').textContent = `${rate}%`;
}

// ── Timeline ──

function renderTimeline() {
    const track = document.getElementById('timeline-track');
    const playhead = document.getElementById('timeline-playhead');
    track.innerHTML = '';
    track.appendChild(playhead);
    if (videoDuration <= 0) return;

    allEvents.forEach(ev => {
        const el = document.createElement('div');
        el.className = `timeline-event ${ev.accepted ? 'accepted' : 'rejected'}`;
        el.style.left = `${(ev.start_time / videoDuration) * 100}%`;
        el.style.width = `${Math.max(((ev.end_time - ev.start_time) / videoDuration) * 100, 0.6)}%`;
        el.title = `${ev.cc_text}  ${ts(ev.start_time)}`;
        el.addEventListener('click', () => seekTo(ev.start_time));
        track.appendChild(el);
    });
}

function updatePlayhead() {
    const v = document.getElementById('video-player');
    if (videoDuration > 0) {
        document.getElementById('timeline-playhead').style.left = `${(v.currentTime / videoDuration) * 100}%`;
    }
    updateCaptionOverlay(v.currentTime);
}

function updateCaptionOverlay(currentTime) {
    const overlay = document.getElementById('cc-overlay');
    const LINGER = 2.0;
    const active = allEvents.find(e =>
        e.accepted &&
        currentTime >= e.start_time &&
        currentTime <= e.end_time + LINGER
    );

    if (active) {
        const cat = active.category || 'default';
        const icons = {
            high_impact: '💥', interactive: '🔔', social: '👥',
            ambient: '🌿', default: '🔊'
        };
        const icon = icons[cat] || icons.default;
        const conf = Math.round(active.confidence * 100);

        overlay.innerHTML = `
            <div class="cc-badge ${cat.replace('_', '-')}"
                 style="font-family:${ccFont}; font-size:${ccSize}px; color:${ccColor};">
                <div class="cc-badge-icon">${icon}</div>
                <div class="cc-badge-content">
                    <div class="cc-badge-label">${active.cc_text}</div>
                    <div class="cc-badge-meta">${cat.replace('_', ' ')}</div>
                </div>
                <div class="cc-badge-confidence">${conf}%</div>
            </div>`;
        overlay.style.bottom = ccPosition === 'auto' ? '' : ccPosition;
        overlay.style.top = ccPosition === 'auto' ? '48px' : '';
        overlay.classList.add('visible');

        // Highlight the active event card
        document.querySelectorAll('.event-card').forEach(card => {
            card.classList.remove('cc-active');
        });
        const activeCard = document.querySelector(`.event-card[data-event-id="${active.id}"]`);
        if (activeCard) {
            activeCard.classList.add('cc-active');
            activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    } else {
        overlay.classList.remove('visible');
        document.querySelectorAll('.event-card').forEach(card => {
            card.classList.remove('cc-active');
        });
    }
}

function seekTo(t) {
    const v = document.getElementById('video-player');
    v.currentTime = t;
    v.play();
}

// ── Events List ──

function renderEvents() {
    const list = document.getElementById('events-list');
    list.innerHTML = '';

    const filtered = allEvents.filter(e => {
        if (currentFilter === 'accepted') return e.accepted;
        if (currentFilter === 'rejected') return !e.accepted;
        return true;
    });

    filtered.forEach(ev => {
        const card = document.createElement('div');
        card.className = `event-card ${ev.accepted ? 'accepted-event' : 'rejected-event'}`;
        card.setAttribute('data-event-id', ev.id);
        card.onclick = () => seekTo(ev.start_time);

        const sceneTag = ev.on_scene_cut ? '  ·  scene cut' : '';
        const speechTag = ev.speech_paused ? '  ·  speech paused' : '';

        card.innerHTML = `
            <div class="event-id">${ev.id}</div>
            <div class="event-info">
                <h4>${ev.cc_text}</h4>
                <div class="event-meta">
                    <span>${ts(ev.start_time)} → ${ts(ev.end_time)}</span>
                    <span>${ev.label}${sceneTag}${speechTag}</span>
                </div>
            </div>
            <div class="event-scores">
                <span class="score-pill">A ${(ev.confidence * 100).toFixed(0)}%</span>
                <span class="score-pill">V ${(ev.reaction_score * 100).toFixed(0)}%</span>
            </div>
            <div class="category-badge ${ev.category}">${ev.category.replace('_', ' ')}</div>
            <div class="event-toggle ${ev.accepted ? 'on' : ''}" 
                 onclick="toggleEvent(event, ${ev.id})"
                 title="${ev.accepted ? 'Reject' : 'Accept'}"></div>
        `;
        list.appendChild(card);
    });
}

async function toggleEvent(clickEv, id) {
    clickEv.stopPropagation();
    try {
        const res = await fetch(`/api/toggle/${currentJobId}/${id}`, { method: 'POST' });
        const data = await res.json();
        const ev = allEvents.find(e => e.id === id);
        if (ev) ev.accepted = data.accepted;
        updateStats();
        renderEvents();
        renderTimeline();
        renderSRT();
        document.getElementById('status-badge').textContent = `${allEvents.filter(e => e.accepted).length} captions`;
    } catch (err) { console.error(err); }
}

function filterEvents(f, btn) {
    currentFilter = f;
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    renderEvents();
}

// ── SRT ──

function renderSRT() {
    const acc = allEvents.filter(e => e.accepted).sort((a, b) => a.start_time - b.start_time);
    let srt = '';
    acc.forEach((e, i) => {
        srt += `${i + 1}\n${srtTs(e.start_time)} --> ${srtTs(e.end_time)}\n${e.cc_text}\n\n`;
    });
    document.getElementById('srt-preview').textContent = srt || 'No accepted events.';
}

function exportSRT() {
    if (currentJobId) window.location.href = `/api/export/${currentJobId}`;
}

function exportSLS() {
    if (currentJobId) window.location.href = `/api/export-sls/${currentJobId}`;
}

// ── Caption Style Customizer ──

let ccFont = "'Inter', system-ui, sans-serif";
let ccSize = 15;
let ccColor = '#ffffff';
let ccPosition = '48px';
let ccBgOpacity = 0.78;

function toggleCustomizer() {
    const body = document.getElementById('customizer-body');
    const arrow = document.getElementById('customizer-arrow');
    if (body.style.display === 'none') {
        body.style.display = 'flex';
        arrow.textContent = '▾';
    } else {
        body.style.display = 'none';
        arrow.textContent = '▸';
    }
}

function applyCaptionStyle() {
    ccFont = document.getElementById('cc-font').value;
    ccSize = parseInt(document.getElementById('cc-size').value);
    ccPosition = document.getElementById('cc-position').value;
    ccBgOpacity = parseInt(document.getElementById('cc-bg-opacity').value) / 100;

    document.getElementById('cc-size-val').textContent = ccSize + 'px';
    document.getElementById('cc-bg-val').textContent = Math.round(ccBgOpacity * 100) + '%';

    // Apply live to any visible caption
    const overlay = document.getElementById('cc-overlay');
    const badge = overlay.querySelector('.cc-badge');
    if (badge) {
        badge.style.fontFamily = ccFont;
        badge.style.fontSize = ccSize + 'px';
        badge.style.color = ccColor;
        badge.style.setProperty('--cc-bg-alpha', ccBgOpacity);
    }

    // Save to CSS custom properties for future captions
    document.documentElement.style.setProperty('--cc-font', ccFont);
    document.documentElement.style.setProperty('--cc-size', ccSize + 'px');
    document.documentElement.style.setProperty('--cc-color', ccColor);
    document.documentElement.style.setProperty('--cc-bg-alpha', ccBgOpacity);
}

function setCCColor(el) {
    document.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
    el.classList.add('active');
    ccColor = el.dataset.color;
    applyCaptionStyle();
}

// ── Keyboard Shortcuts ──

document.addEventListener('keydown', (e) => {
    // Don't interfere with inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    const video = document.getElementById('video-player');
    if (!video || !currentJobId) return;

    switch (e.key) {
        case ' ':
            e.preventDefault();
            video.paused ? video.play() : video.pause();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            video.currentTime = Math.max(0, video.currentTime - 5);
            break;
        case 'ArrowRight':
            e.preventDefault();
            video.currentTime = Math.min(video.duration, video.currentTime + 5);
            break;
        case 'j':
        case 'J':
            e.preventDefault();
            jumpToEvent(-1);
            break;
        case 'k':
        case 'K':
            e.preventDefault();
            jumpToEvent(1);
            break;
    }
});

function jumpToEvent(direction) {
    const accepted = allEvents.filter(e => e.accepted).sort((a, b) => a.start_time - b.start_time);
    if (!accepted.length) return;

    const video = document.getElementById('video-player');
    const ct = video.currentTime;

    if (direction > 0) {
        // Next event
        const next = accepted.find(e => e.start_time > ct + 0.5);
        if (next) seekTo(next.start_time);
    } else {
        // Previous event
        const prev = [...accepted].reverse().find(e => e.start_time < ct - 0.5);
        if (prev) seekTo(prev.start_time);
    }
}

// ── Navigation ──

function showSection(name) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(`section-${name}`).classList.add('active');
}

function resetApp() {
    currentJobId = null;
    allEvents = [];
    videoDuration = 0;
    document.getElementById('status-badge').textContent = 'Ready';
    showSection('upload');
}

// ── Helpers ──

function ts(s) {
    const m = Math.floor(s / 60);
    return `${m}:${(s % 60).toFixed(1).padStart(4, '0')}`;
}

function srtTs(s) {
    if (s < 0) s = 0;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    const ms = Math.round((s % 1) * 1000);
    return `${p(h)}:${p(m)}:${p(sec)},${String(ms).padStart(3, '0')}`;
}

function p(n) { return String(n).padStart(2, '0'); }

