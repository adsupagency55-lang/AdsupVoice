/* ═══════════════════════════════════════════════════════════
   AdsupVoice — app.js
   Handles all UI interactions and API calls for EverAI-style UI
═══════════════════════════════════════════════════════════ */
'use strict';

const API = '';         // same origin
let selectedVoiceId = null;
let selectedVoiceName = 'Chưa chọn';
let allVoices = [];
let carouselOffset = 0;
const CARDS_VISIBLE = 5;
let cloneAudioFile = null;
let pollTimer = null;
let history = JSON.parse(localStorage.getItem('adsup_history') || '[]');

/* ─────────────────────────────── HELPERS ─────────────────────────────── */
function $(id) { return document.getElementById(id); }

function toast(msg, type = 'info') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 3000);
}

function setOverlay(visible, label = 'Đang tạo giọng nói…') {
  $('gen-overlay').classList.toggle('hidden', !visible);
  $('gen-label').textContent = label;
}

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

/* ─────────────────────────────── STATUS POLLING ─────────────────────── */
async function fetchStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const s = await r.json();
    const dot  = $('status-dot');
    const txt  = $('status-text');
    const msBb = $('ms-backbone');
    const msCo = $('ms-codec');
    const msSt = $('ms-status');
    const footDot = $('footer-status-dot');
    const footTxt = $('footer-status-text');

    if (s.model_loaded) {
      dot.className  = 'status-dot ready';
      txt.textContent = 'Sẵn sàng';
      if (footDot) footDot.className = 'status-dot ready';
      if (footTxt) footTxt.textContent = 'Sẵn sàng';

      msBb.textContent = s.backbone || '—';
      msCo.textContent = s.codec    || '—';
      msSt.textContent = '✅ Đã tải';
      msSt.style.color = '#22c55e';
      $('btn-generate').disabled = !selectedVoiceId;
      $('btn-generate-clone').disabled = false;
      // Fetch voices once loaded
      if (allVoices.length === 0) fetchVoices();
      return true;
    } else {
      dot.className  = 'status-dot loading';
      txt.textContent = 'Đang tải model…';
      if (footDot) footDot.className = 'status-dot loading';
      if (footTxt) footTxt.textContent = 'Đang tải model…';
      msSt.textContent = '⏳ Loading';
      msSt.style.color = '#f59e0b';
      return false;
    }
  } catch(e) {
    $('status-dot').className = 'status-dot error';
    $('status-text').textContent = 'Lỗi kết nối';
    return false;
  }
}

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const ready = await fetchStatus();
    if (ready) clearInterval(pollTimer);
  }, 2000);
}

/* ─────────────────────────────── MODEL LOADING ─────────────────────── */
async function loadModels() {
  try {
    const r  = await fetch(`${API}/api/models`);
    const data = await r.json();

    const bbSel = $('setting-backbone');
    const coSel = $('setting-codec');
    bbSel.innerHTML = data.backbones.map(b =>
      `<option value="${b.id}">${b.id}</option>`
    ).join('');
    coSel.innerHTML = data.codecs.map(c =>
      `<option value="${c.id}">${c.id}</option>`
    ).join('');

    // Defaults
    const defaultBb = 'VieNeu-TTS-0.3B-q4-gguf';
    const defaultCo = 'NeuCodec ONNX (Fast CPU)';
    [...bbSel.options].forEach(o => { if (o.value === defaultBb) o.selected = true; });
    [...coSel.options].forEach(o => { if (o.value === defaultCo) o.selected = true; });
  } catch(e) {
    console.error('Could not load model list:', e);
  }
}

$('btn-load-model').addEventListener('click', async () => {
  const backbone     = $('setting-backbone').value;
  const codec        = $('setting-codec').value;
  const device       = $('setting-device').value;
  const use_lmdeploy = $('setting-lmdeploy').checked;

  $('btn-load-model').disabled = true;
  $('load-model-label').textContent = '⏳ Đang tải…';
  $('status-dot').className = 'status-dot loading';
  $('status-text').textContent = 'Đang tải model…';
  $('ms-backbone').textContent = backbone;
  $('ms-codec').textContent    = codec;
  $('ms-status').textContent   = '⏳ Loading';
  allVoices = [];

  try {
    await fetch(`${API}/api/load_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backbone, codec, device, use_lmdeploy }),
    });
    startPolling();
    toast('Bắt đầu tải model, vui lòng đợi…');
  } catch(e) {
    toast('Lỗi khi gửi yêu cầu tải model', 'error');
  } finally {
    $('btn-load-model').disabled = false;
    $('load-model-label').textContent = '🔄 Tải Model';
  }
});

/* ─────────────────────────────── VOICE CAROUSEL ─────────────────────── */
const VOICE_EMOJIS = ['🎤','🎧','👩','👨','🌺','🌊','🏔️','🌸','🎼','🎙️'];

function renderCarousel() {
  const track = $('carousel-track');
  if (allVoices.length === 0) {
    track.innerHTML = '<div class="carousel-loading">Tải model để xem danh sách giọng</div>';
    return;
  }
  const visible = allVoices.slice(carouselOffset, carouselOffset + CARDS_VISIBLE);
  track.innerHTML = visible.map((v, i) => {
    const emoji = VOICE_EMOJIS[(carouselOffset + i) % VOICE_EMOJIS.length];
    const isSelected = v.id === selectedVoiceId;
    const nameParts = v.name.split(' ');
    const shortName = nameParts[0];
    const subName   = nameParts.slice(1).join(' ') || 'Vietnamese';
    return `
      <div class="voice-card${isSelected ? ' selected' : ''}" data-id="${v.id}" data-name="${v.name}">
        <div class="vc-avatar">${emoji}</div>
        <div class="vc-name">${shortName}</div>
        <div class="vc-sub">${subName}</div>
        <div class="vc-play">▶</div>
      </div>`;
  }).join('');

  track.querySelectorAll('.voice-card').forEach(card => {
    const playBtn = card.querySelector('.vc-play');
    const voiceId = card.dataset.id;

    card.addEventListener('click', () => {
      selectedVoiceId   = voiceId;
      selectedVoiceName = card.dataset.name;
      updateSelectedVoiceUI();
      renderCarousel();
    });

    playBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      playPreview(voiceId, playBtn);
    });
  });
}

const audioPreview = new Audio();
async function playPreview(voiceId, btn) {
  if (btn.textContent === '⏳') return;
  const oldTxt = btn.textContent;
  btn.textContent = '⏳';
  
  try {
    const r = await fetch(`${API}/api/preview_voice?voice_id=${encodeURIComponent(voiceId)}`);
    if (!r.ok) throw new Error('Preview failed');
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    audioPreview.src = url;
    audioPreview.play();
  } catch(e) {
    toast('Không thể tải bản nghe thử', 'error');
  } finally {
    btn.textContent = oldTxt;
  }
}

function updateSelectedVoiceUI() {
  const nameParts = selectedVoiceName.split(' ');
  $('sel-name').textContent  = selectedVoiceName || 'Chưa chọn';
  $('sel-style').textContent = 'Vietnamese TTS';
  $('sel-avatar').textContent = selectedVoiceName ? selectedVoiceName[0].toUpperCase() : 'V';
  if ($('style-chip-preset')) {
    $('style-chip-preset').textContent = `Giọng: ${selectedVoiceName || 'Chưa chọn'}`;
  }
  $('btn-generate').disabled = !selectedVoiceId;
}

async function fetchVoices() {
  try {
    const r = await fetch(`${API}/api/voices`);
    allVoices = await r.json();
    carouselOffset = 0;
    renderCarousel();
    // Auto-select first voice
    if (allVoices.length > 0 && !selectedVoiceId) {
      selectedVoiceId   = allVoices[0].id;
      selectedVoiceName = allVoices[0].name;
      updateSelectedVoiceUI();
      renderCarousel();
    }
  } catch(e) { console.error(e); }
}

$('carousel-prev').addEventListener('click', () => {
  if (carouselOffset > 0) { carouselOffset = Math.max(0, carouselOffset - 1); renderCarousel(); }
});
$('carousel-next').addEventListener('click', () => {
  if (carouselOffset + CARDS_VISIBLE < allVoices.length) { carouselOffset++; renderCarousel(); }
});

/* ─────────────────────────────── CHAR COUNTER ─────────────────────── */
$('text-input').addEventListener('input', () => {
  $('char-count').textContent = $('text-input').value.length;
});
$('text-input-clone').addEventListener('input', () => {
  $('char-count-clone').textContent = $('text-input-clone').value.length;
});
$('btn-clear-text').addEventListener('click', () => {
  $('text-input').value = '';
  $('char-count').textContent = '0';
});
$('btn-clear-clone').addEventListener('click', () => {
  $('text-input-clone').value = '';
  $('char-count-clone').textContent = '0';
});

/* ─────────────────────────────── LANG BUTTONS ─────────────────────── */
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

/* ─────────────────────────────── MODE TABS ─────────────────────── */
document.querySelectorAll('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const mode = tab.dataset.mode;
    $('panel-preset').classList.toggle('hidden', mode !== 'preset');
    $('panel-clone').classList.toggle('hidden',  mode !== 'clone');
  });
});

/* ─────────────────────────────── RIGHT TABS ─────────────────────── */
document.querySelectorAll('.right-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.right-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.dataset.rtab;
    $('rtab-settings').classList.toggle('hidden', target !== 'settings');
    $('rtab-history').classList.toggle('hidden',  target !== 'history');
  });
});

/* ─────────────────────────────── SETTINGS SLIDERS ─────────────────────── */
$('setting-temp').addEventListener('input', () => {
  $('temp-val').textContent = parseFloat($('setting-temp').value).toFixed(1);
});
$('setting-chunk').addEventListener('input', () => {
  $('chunk-val').textContent = $('setting-chunk').value;
});

/* ─────────────────────────────── GENERATE (PRESET) ─────────────────────── */
$('btn-generate').addEventListener('click', async () => {
  const text = $('text-input').value.trim();
  if (!text) { toast('Vui lòng nhập văn bản!', 'error'); return; }
  if (!selectedVoiceId) { toast('Vui lòng chọn giọng!', 'error'); return; }

  setOverlay(true);
  $('btn-generate').disabled = true;

  try {
    const r = await fetch(`${API}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        voice_id: selectedVoiceId,
        temperature: parseFloat($('setting-temp').value),
        max_chars_chunk: parseInt($('setting-chunk').value),
      }),
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || 'Lỗi tạo giọng nói');
    }

    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    addHistoryItem(text, selectedVoiceName, url);
    switchToHistory();
    toast('✅ Tạo xong! Nghe trong History ➡', 'success');
  } catch(e) {
    toast(`❌ ${e.message}`, 'error');
  } finally {
    setOverlay(false);
    $('btn-generate').disabled = false;
  }
});

/* ─────────────────────────────── CLONE AUDIO UPLOAD ─────────────────────── */
const cloneInput = $('clone-audio-input');
const cloneDrop  = $('clone-drop-area');

$('clone-browse').addEventListener('click', () => cloneInput.click());
cloneInput.addEventListener('change', () => {
  if (cloneInput.files[0]) {
    cloneAudioFile = cloneInput.files[0];
    $('clone-filename').textContent = `✅ ${cloneAudioFile.name}`;
  }
});
cloneDrop.addEventListener('dragover', e => { e.preventDefault(); cloneDrop.classList.add('dragging'); });
cloneDrop.addEventListener('dragleave', () => cloneDrop.classList.remove('dragging'));
cloneDrop.addEventListener('drop', e => {
  e.preventDefault(); cloneDrop.classList.remove('dragging');
  if (e.dataTransfer.files[0]) {
    cloneAudioFile = e.dataTransfer.files[0];
    cloneInput.files = e.dataTransfer.files;
    $('clone-filename').textContent = `✅ ${cloneAudioFile.name}`;
  }
});

/* ─────────────────────────────── GENERATE (CLONE) ─────────────────────── */
$('btn-generate-clone').addEventListener('click', async () => {
  const text    = $('text-input-clone').value.trim();
  const refText = $('clone-ref-text').value.trim();
  if (!text)          { toast('Vui lòng nhập văn bản để đọc!', 'error'); return; }
  if (!cloneAudioFile){ toast('Vui lòng chọn file audio mẫu!', 'error'); return; }

  setOverlay(true, 'Đang clone giọng nói…');
  $('btn-generate-clone').disabled = true;

  try {
    const fd = new FormData();
    fd.append('text',          text);
    fd.append('ref_text',      refText);
    fd.append('temperature',   $('setting-temp').value);
    fd.append('max_chars_chunk', $('setting-chunk').value);
    fd.append('ref_audio',     cloneAudioFile);

    const r = await fetch(`${API}/api/generate_clone`, { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || 'Lỗi clone giọng nói');
    }

    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    addHistoryItem(text, `Clone: ${cloneAudioFile.name}`, url);
    switchToHistory();
    toast('✅ Clone xong! Nghe trong History ➡', 'success');
  } catch(e) {
    toast(`❌ ${e.message}`, 'error');
  } finally {
    setOverlay(false);
    $('btn-generate-clone').disabled = false;
  }
});

/* ─────────────────────────────── HISTORY ─────────────────────── */
function addHistoryItem(text, voiceName, audioUrl) {
  const item = { id: Date.now(), text, voiceName, audioUrl, ts: Date.now() };
  history.unshift(item);
  if (history.length > 50) history.pop();
  renderHistory();
}

function renderHistory() {
  const list = $('history-list');
  if (history.length === 0) {
    list.innerHTML = '<div class="history-empty">Chưa có lịch sử</div>';
    return;
  }
  list.innerHTML = history.map(item => `
    <div class="history-item" id="hi-${item.id}">
      <div class="hi-meta">
        <span class="hi-text" title="${item.text}">${item.text}</span>
        <span class="hi-time">${formatTime(item.ts)}</span>
      </div>
      <div class="hi-voice">🎤 ${item.voiceName}</div>
      <audio class="hi-player" controls src="${item.audioUrl}"></audio>
      <div class="hi-actions">
        <button class="hi-btn" onclick="downloadAudio('${item.audioUrl}', ${item.id})">⬇ Tải</button>
        <button class="hi-btn danger" onclick="removeHistory(${item.id})">✕</button>
      </div>
    </div>
  `).join('');
}

window.downloadAudio = (url, id) => {
  const a = document.createElement('a');
  a.href = url;
  a.download = `adsupvoice_${id}.wav`;
  a.click();
};

window.removeHistory = (id) => {
  history = history.filter(h => h.id !== id);
  renderHistory();
};

$('btn-clear-history').addEventListener('click', () => {
  history = [];
  renderHistory();
  toast('Đã xoá lịch sử');
});

function switchToHistory() {
  document.querySelectorAll('.right-tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-rtab="history"]').classList.add('active');
  $('rtab-settings').classList.add('hidden');
  $('rtab-history').classList.remove('hidden');
}

/* ─────────────────────────────── SIDEBAR NAV ─────────────────────── */
document.querySelectorAll('.nav-item[data-page]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    a.classList.add('active');
    const page = a.dataset.page;
    if (page === 'history') {
      document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
      switchToHistory();
    } else if (page === 'clone') {
      document.querySelector('[data-mode="clone"]')?.click();
    } else if (page === 'tts') {
      document.querySelector('[data-mode="preset"]')?.click();
    }
  });
});

/* ─────────────────────────────── INIT ─────────────────────── */
async function init() {
  await loadModels();
  await fetchStatus();
  renderHistory();
}

init();
