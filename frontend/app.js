const STORAGE_KEYS = {
  sessionId: "finbot.session_id",
  chatHistory: "finbot.chat_history",
  lastResponse: "finbot.last_response",
  languagePreference: "finbot.language_preference"
};

const API_BASE = window.APP_API_BASE || "http://127.0.0.1:8000";
const TURN_URL = `${API_BASE}/chat/turn`;
const MODEL_LOAD_URL = `${API_BASE}/model/load`;

const THINKING_STEP_MS = 2000;
const THINKING_STEPS_I18N = {
  en: [
    "Analyzing your profile and preferences …",
    "Applying policy rules …",
    "Preparing response …"
  ],
  vi: [
    "Đang phân tích hồ sơ và sở thích của bạn …",
    "Đang áp dụng các quy tắc chính sách …",
    "Đang chuẩn bị phản hồi …"
  ],
  zh: [
    "正在分析您的资料和偏好 …",
    "正在应用政策规则 …",
    "正在准备回复 …"
  ]
};
function getThinkingSteps() {
  return THINKING_STEPS_I18N[getLanguageHint()] || THINKING_STEPS_I18N.en;
}

/** @type {Map<string, ReturnType<typeof setInterval>>} */
const thinkingTimers = new Map();

const chatList      = document.getElementById("chatList");
const chatForm      = document.getElementById("chatForm");
const chatInput     = document.getElementById("chatInput");
const sendBtn       = document.getElementById("sendBtn");
const resetBtn      = document.getElementById("resetBtn");
const modelSelect   = document.getElementById("modelSelect");
const langSelect    = document.getElementById("langSelect");

const sessionBadge  = document.getElementById("sessionBadge");
const apiBaseLabel  = document.getElementById("apiBaseLabel");
const stateValue    = document.getElementById("stateValue");
const taskModeValue = document.getElementById("taskModeValue");
const nextItemValue = document.getElementById("nextItemValue");
const readyValue    = document.getElementById("readyValue");
const langValue     = document.getElementById("langValue");
const collectedJson = document.getElementById("collectedJson");

apiBaseLabel.textContent = API_BASE;

// ── Storage helpers ─────────────────────────────────────────────────────────

function loadSessionId() { return localStorage.getItem(STORAGE_KEYS.sessionId); }
function saveSessionId(id) { if (id) localStorage.setItem(STORAGE_KEYS.sessionId, id); }

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEYS.chatHistory) || "[]"); }
  catch { return []; }
}
function saveHistory(history) { localStorage.setItem(STORAGE_KEYS.chatHistory, JSON.stringify(history)); }
function saveLastResponse(data) { localStorage.setItem(STORAGE_KEYS.lastResponse, JSON.stringify(data)); }

function getLanguageHint() { return langSelect?.value || "en"; }
function getModelId() { return modelSelect?.value || "Qwen/Qwen2.5-1.5B-Instruct"; }

function restoreLanguagePreference() {
  const saved = localStorage.getItem(STORAGE_KEYS.languagePreference);
  if (!saved || !langSelect) return;
  if (["en", "vi", "zh"].includes(saved)) langSelect.value = saved;
}

// ── HTML escaping ────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ── Avatar SVG helpers ───────────────────────────────────────────────────────

function botAvatarHtml() {
  return `<div class="avatar avatar-bot" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16">
      <circle cx="12" cy="8" r="4"/>
      <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
    </svg>
  </div>`;
}

function userAvatarHtml() {
  return `<div class="avatar avatar-user" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14">
      <circle cx="12" cy="8" r="4"/>
      <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
    </svg>
  </div>`;
}

// ── Bubble rendering ─────────────────────────────────────────────────────────

function stopThinkingLabel(id) {
  const timer = thinkingTimers.get(id);
  if (timer !== undefined) {
    clearInterval(timer);
    thinkingTimers.delete(id);
  }
}

function startThinkingLabel(id) {
  stopThinkingLabel(id);
  let stepIndex = 0;

  const tick = () => {
    const row = chatList.querySelector(`[data-id="${id}"]`);
    if (!row) {
      stopThinkingLabel(id);
      return;
    }
    const label = row.querySelector(".loading-label");
    if (!label) {
      stopThinkingLabel(id);
      return;
    }
    const steps = getThinkingSteps();
    updateLoadingBubbleText(id, steps[stepIndex % steps.length]);
    stepIndex += 1;
  };

  tick();
  thinkingTimers.set(id, setInterval(tick, THINKING_STEP_MS));
}

function appendBubble(role, text, { id = null, loading = false, recommendation = null } = {}) {
  const isUser = role === "user";
  const row = document.createElement("div");
  row.className = `msg-row${isUser ? " user" : ""}`;
  if (id) row.dataset.id = id;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${isUser ? "bubble-user" : "bubble-assistant"}`;

  if (loading) {
    bubble.innerHTML = `
      <div style="display:flex;align-items:center;">
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span class="loading-label">${escapeHtml(text)}</span>
      </div>`;
  } else {
    renderAssistantContent(bubble, text, recommendation);
  }

  row.innerHTML = isUser ? userAvatarHtml() : botAvatarHtml();
  if (isUser) {
    row.insertBefore(bubble, row.firstChild);
  } else {
    row.appendChild(bubble);
  }

  chatList.appendChild(row);
  chatList.scrollTop = chatList.scrollHeight;
}

function replaceBubble(id, text, recommendation = null) {
  stopThinkingLabel(id);
  const row = chatList.querySelector(`[data-id="${id}"]`);
  if (!row) return;
  const bubble = row.querySelector(".bubble");
  if (bubble) renderAssistantContent(bubble, text, recommendation);
}

function updateLoadingBubbleText(id, text) {
  const row = chatList.querySelector(`[data-id="${id}"]`);
  if (!row) return;
  const label = row.querySelector(".loading-label");
  if (label) label.textContent = text;
}

// ── Recommendation card ──────────────────────────────────────────────────────

function createRecommendationCard(rec) {
  if (!rec) return null;

  const safeSources = Array.isArray(rec.sources)
    ? rec.sources.filter((s) => s && String(s).trim()) : [];
  const sourceChips = safeSources.length > 0
    ? safeSources.map((s) => `<span class="source-chip">${escapeHtml(s)}</span>`).join("")
    : `<span class="source-chip">N/A</span>`;

  const card = document.createElement("div");
  card.className = "recommendation-card";
  card.innerHTML = `
    <div class="recommendation-header">
      <h5>Personalized Recommendation</h5>
      <p class="recommendation-subtitle">Generated from your profile inputs and current session context.</p>
    </div>
    <div class="recommendation-body">
      <div class="recommendation-grid">
        <section class="recommendation-item recommendation-item-wide">
          <h6>Profile Summary</h6>
          <p>${escapeHtml(rec.profile_summary || "N/A")}</p>
        </section>
        <section class="recommendation-item recommendation-item-wide">
          <h6>Recommendation</h6>
          <p>${escapeHtml(rec.recommendation || "N/A")}</p>
        </section>
        <section class="recommendation-item">
          <h6>Reasoning</h6>
          <p>${escapeHtml(rec.reasoning || "N/A")}</p>
        </section>
        <section class="recommendation-item">
          <h6>Risks &amp; Caveats</h6>
          <p>${escapeHtml(rec.risks_caveats || "N/A")}</p>
        </section>
        <section class="recommendation-item recommendation-item-wide">
          <h6>Sources</h6>
          <div class="source-badges">${sourceChips}</div>
        </section>
        <section class="recommendation-item recommendation-item-wide">
          <h6>Disclaimer</h6>
          <p>${escapeHtml(rec.disclaimer || "N/A")}</p>
        </section>
      </div>
    </div>`;
  return card;
}

function renderAssistantContent(target, text, recommendation = null) {
  target.innerHTML = "";
  const p = document.createElement("p");
  p.style.marginBottom = "0";
  p.textContent = text.trim();
  target.appendChild(p);

  const card = createRecommendationCard(recommendation);
  if (card) {
    target.classList.add("has-recommendation");
    target.appendChild(card);
  } else {
    target.classList.remove("has-recommendation");
  }
}

// ── History rendering ────────────────────────────────────────────────────────

function renderHistory() {
  chatList.innerHTML = "";
  loadHistory().forEach((item) =>
    appendBubble(item.role, item.text, { recommendation: item.recommendation || null })
  );
}

function buildAssistantDialogueText(data) {
  return data.assistant_message || "(empty response)";
}

// ── Side panel ───────────────────────────────────────────────────────────────

function updateSidePanel(data) {
  if (data.session_id) {
    sessionBadge.textContent = `${data.session_id.slice(0, 8)}…`;
  } else {
    sessionBadge.textContent = "No session";
  }
  stateValue.textContent    = data.state      || "—";
  taskModeValue.textContent = data.task_mode  || "—";
  nextItemValue.textContent = data.next_item  || "—";
  langValue.textContent     = data.detected_language || "—";

  const ready = Boolean(data.ready_for_recommendation);
  readyValue.textContent = String(ready);
  readyValue.className   = `badge-pill ${ready ? "badge-ready" : "badge-pending"}`;

  collectedJson.textContent = JSON.stringify(data.collected || {}, null, 2);
}

// ── Send state ───────────────────────────────────────────────────────────────

function setSending(isSending) {
  chatInput.disabled = isSending;
  sendBtn.disabled   = isSending;
}

// ── Init chat ────────────────────────────────────────────────────────────────

async function initializeChat() {
  const existingHistory = loadHistory();
  if (existingHistory.length > 0) return;

  const loadingId = `loading-init-${Date.now()}`;
  appendBubble("assistant", "Loading model…", { id: loadingId, loading: true });
  startThinkingLabel(loadingId);
  setSending(true);

  const payload = {
    session_id:    loadSessionId(),
    message:       "__INIT__",
    model_id:      getModelId(),
    language_hint: getLanguageHint()
  };

  try {
    const warmupRes = await fetch(MODEL_LOAD_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: getModelId() })
    });
    if (!warmupRes.ok) {
      const body = await warmupRes.text();
      throw new Error(`Model load failed (${warmupRes.status}): ${body}`);
    }

    const res = await fetch(TURN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      let errorMsg = `HTTP ${res.status}`;
      try { const body = await res.json(); errorMsg = body?.error?.message || errorMsg; }
      catch { errorMsg = (await res.text()) || errorMsg; }
      throw new Error(errorMsg);
    }

    const data = await res.json();
    saveSessionId(data.session_id);
    saveLastResponse(data);

    const assistantText = buildAssistantDialogueText(data);
    replaceBubble(loadingId, assistantText, data.recommendation);

    saveHistory([{ role: "assistant", text: assistantText, recommendation: data.recommendation || null }]);
    updateSidePanel(data);
  } catch (err) {
    replaceBubble(loadingId, `Failed to initialise chat. ${err.message}`);
  } finally {
    setSending(false);
    chatInput.focus();
  }
}

// ── Send message ─────────────────────────────────────────────────────────────

async function sendMessage(message) {
  const history = loadHistory();
  history.push({ role: "user", text: message });
  saveHistory(history);
  appendBubble("user", message);

  const loadingId = `loading-${Date.now()}`;
  appendBubble("assistant", "Thinking…", { id: loadingId, loading: true });
  startThinkingLabel(loadingId);
  setSending(true);

  const payload = {
    session_id:    loadSessionId(),
    message,
    model_id:      getModelId(),
    language_hint: getLanguageHint()
  };

  try {
    const res = await fetch(TURN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      let errorMsg = `HTTP ${res.status}`;
      try { const body = await res.json(); errorMsg = body?.error?.message || errorMsg; }
      catch { errorMsg = (await res.text()) || errorMsg; }
      throw new Error(errorMsg);
    }

    const data = await res.json();
    saveSessionId(data.session_id);
    saveLastResponse(data);

    const assistantText = buildAssistantDialogueText(data);
    replaceBubble(loadingId, assistantText, data.recommendation);

    history.push({ role: "assistant", text: assistantText, recommendation: data.recommendation || null });
    saveHistory(history);
    updateSidePanel(data);
  } catch (err) {
    replaceBubble(loadingId, `Request failed. ${err.message}`);
  } finally {
    setSending(false);
    chatInput.focus();
  }
}

// ── Event listeners ──────────────────────────────────────────────────────────

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = "";
  await sendMessage(msg);
});

resetBtn.addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEYS.sessionId);
  localStorage.removeItem(STORAGE_KEYS.chatHistory);
  localStorage.removeItem(STORAGE_KEYS.lastResponse);
  renderHistory();
  updateSidePanel({
    state: null,
    task_mode: null,
    next_item: null,
    ready_for_recommendation: false,
    detected_language: null,
    collected: {}
  });
  sessionBadge.textContent = "No session";
  initializeChat();
});

if (langSelect) {
  langSelect.addEventListener("change", () => {
    localStorage.setItem(STORAGE_KEYS.languagePreference, langSelect.value);
  });
}

// ── Boot ─────────────────────────────────────────────────────────────────────

(function init() {
  restoreLanguagePreference();
  renderHistory();
  const lastResponseRaw = localStorage.getItem(STORAGE_KEYS.lastResponse);
  if (lastResponseRaw) {
    try { updateSidePanel(JSON.parse(lastResponseRaw)); } catch { /* ignore */ }
  }
  if (loadSessionId()) {
    sessionBadge.textContent = `${loadSessionId().slice(0, 8)}…`;
  }
  initializeChat();
})();
