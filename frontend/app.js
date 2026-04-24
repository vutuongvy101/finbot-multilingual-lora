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
const MODEL_LOADING_STEPS_I18N = {
  en: [
    "Loading model …",
    "Warming up model …",
    "Preparing chat session …"
  ],
  vi: [
    "Đang tải mô hình …",
    "Đang khởi động mô hình …",
    "Đang chuẩn bị phiên chat …"
  ],
  zh: [
    "正在加载模型 …",
    "正在预热模型 …",
    "正在准备聊天会话 …"
  ]
};
const UI_I18N = {
  en: {
    noSession: "No session",
    loadingModel: "Loading model…",
    activatingModel: "Activating model",
    thinking: "Thinking…",
    initFailed: "Failed to initialise chat.",
    requestFailed: "Request failed.",
    emptyResponse: "(empty response)",
    recommendationTitle: "Personalized Recommendation",
    recommendationSubtitle: "Generated from your profile inputs and current session context.",
    profileSummary: "Profile Summary",
    recommendation: "Recommendation",
    reasoning: "Reasoning",
    risksCaveats: "Risks & Caveats",
    sources: "Sources",
    disclaimer: "Disclaimer",
    na: "N/A"
  },
  vi: {
    noSession: "Chưa có phiên",
    loadingModel: "Đang tải mô hình…",
    activatingModel: "Đang kích hoạt mô hình",
    thinking: "Đang suy luận…",
    initFailed: "Khởi tạo hội thoại thất bại.",
    requestFailed: "Yêu cầu thất bại.",
    emptyResponse: "(không có phản hồi)",
    recommendationTitle: "Khuyến nghị cá nhân hóa",
    recommendationSubtitle: "Được tạo từ thông tin hồ sơ và ngữ cảnh phiên hiện tại.",
    profileSummary: "Tóm tắt hồ sơ",
    recommendation: "Khuyến nghị",
    reasoning: "Lý do",
    risksCaveats: "Rủi ro & lưu ý",
    sources: "Nguồn",
    disclaimer: "Miễn trừ trách nhiệm",
    na: "N/A"
  },
  zh: {
    noSession: "暂无会话",
    loadingModel: "正在加载模型…",
    activatingModel: "正在激活模型",
    thinking: "正在思考…",
    initFailed: "初始化会话失败。",
    requestFailed: "请求失败。",
    emptyResponse: "（无回复）",
    recommendationTitle: "个性化建议",
    recommendationSubtitle: "基于您的资料输入和当前会话上下文生成。",
    profileSummary: "资料摘要",
    recommendation: "建议",
    reasoning: "理由",
    risksCaveats: "风险与注意事项",
    sources: "来源",
    disclaimer: "免责声明",
    na: "N/A"
  }
};
function getThinkingSteps(labelMode = "thinking") {
  const language = getLanguageHint();
  if (labelMode === "model-loading") {
    return MODEL_LOADING_STEPS_I18N[language] || MODEL_LOADING_STEPS_I18N.en;
  }
  return THINKING_STEPS_I18N[language] || THINKING_STEPS_I18N.en;
}
function uiText(key) {
  const language = getLanguageHint();
  const pack = UI_I18N[language] || UI_I18N.en;
  return pack[key] || UI_I18N.en[key] || "";
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
const settingsState = {
  modelId: getModelId(),
  language: getLanguageHint()
};
let conversationVersion = 0;

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
function loadLastResponse() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEYS.lastResponse) || "null"); }
  catch { return null; }
}

async function parseErrorMessage(res, fallbackPrefix = "HTTP") {
  let errorMsg = `${fallbackPrefix} ${res.status}`;
  try {
    const body = await res.json();
    return body?.error?.message || errorMsg;
  } catch {
    const text = await res.text();
    return text || errorMsg;
  }
}

async function warmupModel(modelId) {
  const warmupRes = await fetch(MODEL_LOAD_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId })
  });
  if (!warmupRes.ok) {
    const body = await warmupRes.text();
    throw new Error(`Model load failed (${warmupRes.status}): ${body}`);
  }
}

async function postTurn(payload) {
  const res = await fetch(TURN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    throw new Error(await parseErrorMessage(res));
  }
  return res.json();
}

function getLanguageHint() { return langSelect?.value || "en"; }
function getModelId() { return modelSelect?.value || "lora-qwen25-1p5b-finbot-v2"; }
function getModelDisplayName() {
  if (!modelSelect) return getModelId();
  const selected = modelSelect.options[modelSelect.selectedIndex];
  return selected?.text?.trim() || getModelId();
}

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

function startThinkingLabel(id, labelMode = "thinking") {
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
    const steps = getThinkingSteps(labelMode);
    updateLoadingBubbleText(id, steps[stepIndex % steps.length]);
    stepIndex += 1;
  };

  tick();
  thinkingTimers.set(id, setInterval(tick, THINKING_STEP_MS));
}

function appendBubble(
  role,
  text,
  { id = null, loading = false, recommendation = null, loadingTitle = "" } = {}
) {
  const isUser = role === "user";
  const row = document.createElement("div");
  row.className = `msg-row${isUser ? " user" : ""}`;
  if (id) row.dataset.id = id;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${isUser ? "bubble-user" : "bubble-assistant"}`;

  if (loading) {
    const safeTitle = loadingTitle ? `<div class="loading-title">${escapeHtml(loadingTitle)}</div>` : "";
    bubble.innerHTML = `
      ${safeTitle}
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

function appendContextMarker(text) {
  const row = document.createElement("div");
  row.className = "context-marker-row";

  const marker = document.createElement("div");
  marker.className = "context-marker";
  marker.textContent = text;

  row.appendChild(marker);
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
    : `<span class="source-chip">${escapeHtml(uiText("na"))}</span>`;

  const card = document.createElement("div");
  card.className = "recommendation-card";
  card.innerHTML = `
    <div class="recommendation-header">
      <h5>${escapeHtml(uiText("recommendationTitle"))}</h5>
      <p class="recommendation-subtitle">${escapeHtml(uiText("recommendationSubtitle"))}</p>
    </div>
    <div class="recommendation-body">
      <div class="recommendation-grid">
        <section class="recommendation-item recommendation-item-wide">
          <h6>${escapeHtml(uiText("profileSummary"))}</h6>
          <p>${escapeHtml(rec.profile_summary || uiText("na"))}</p>
        </section>
        <section class="recommendation-item recommendation-item-wide">
          <h6>${escapeHtml(uiText("recommendation"))}</h6>
          <p>${escapeHtml(rec.recommendation || uiText("na"))}</p>
        </section>
        <section class="recommendation-item">
          <h6>${escapeHtml(uiText("reasoning"))}</h6>
          <p>${escapeHtml(rec.reasoning || uiText("na"))}</p>
        </section>
        <section class="recommendation-item">
          <h6>${escapeHtml(uiText("risksCaveats"))}</h6>
          <p>${escapeHtml(rec.risks_caveats || uiText("na"))}</p>
        </section>
        <section class="recommendation-item recommendation-item-wide">
          <h6>${escapeHtml(uiText("sources"))}</h6>
          <div class="source-badges">${sourceChips}</div>
        </section>
        <section class="recommendation-item recommendation-item-wide">
          <h6>${escapeHtml(uiText("disclaimer"))}</h6>
          <p>${escapeHtml(rec.disclaimer || uiText("na"))}</p>
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
  loadHistory().forEach((item) => {
    if (item.role === "meta") {
      appendContextMarker(item.text);
      return;
    }
    appendBubble(item.role, item.text, { recommendation: item.recommendation || null });
  });
}

function buildAssistantDialogueText(data) {
  return data.assistant_message || uiText("emptyResponse");
}

// ── Side panel ───────────────────────────────────────────────────────────────

function updateSidePanel(data) {
  if (data.session_id) {
    sessionBadge.textContent = `${data.session_id.slice(0, 8)}…`;
  } else {
    sessionBadge.textContent = uiText("noSession");
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

function resetConversation() {
  conversationVersion += 1;
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
  sessionBadge.textContent = uiText("noSession");
  initializeChat({ contextInLoadingBubble: true });
}

function buildContextMarkerText() {
  const lang = getLanguageHint();
  const model = getModelDisplayName();
  if (lang === "vi") return `Đang dùng mô hình: ${model} | Ngôn ngữ: ${lang}`;
  if (lang === "zh") return `当前模型：${model} | 语言：${lang}`;
  return `Now using model: ${model} | Language: ${lang}`;
}

function recordContextChangeMarker() {
  const history = loadHistory();
  if (history.length === 0) return;

  const text = buildContextMarkerText();
  history.push({
    role: "meta",
    text,
    model_id: getModelId(),
    language: getLanguageHint(),
    timestamp: new Date().toISOString()
  });
  saveHistory(history);
  appendContextMarker(text);
}

// ── Init chat ────────────────────────────────────────────────────────────────

async function initializeChat({ contextInLoadingBubble = false } = {}) {
  const requestVersion = conversationVersion;
  const existingHistory = loadHistory();
  if (existingHistory.length > 0) return;

  const initialContextText = buildContextMarkerText();
  if (contextInLoadingBubble) {
    saveHistory([]);
  } else {
    saveHistory([
      {
        role: "meta",
        text: initialContextText,
        model_id: getModelId(),
        language: getLanguageHint(),
        timestamp: new Date().toISOString()
      }
    ]);
    appendContextMarker(initialContextText);
  }

  const loadingId = `loading-init-${Date.now()}`;
  appendBubble("assistant", uiText("loadingModel"), {
    id: loadingId,
    loading: true,
    loadingTitle: contextInLoadingBubble
      ? initialContextText
      : `${uiText("activatingModel")}: ${getModelDisplayName()}`
  });
  startThinkingLabel(loadingId, "model-loading");
  setSending(true);

  const payload = {
    session_id:    loadSessionId(),
    message:       "__INIT__",
    model_id:      getModelId(),
    language_hint: getLanguageHint()
  };

  try {
    await warmupModel(getModelId());
    const data = await postTurn(payload);
    if (requestVersion !== conversationVersion) return;
    saveSessionId(data.session_id);
    saveLastResponse(data);

    const assistantText = buildAssistantDialogueText(data);
    replaceBubble(loadingId, assistantText, data.recommendation);

    const history = loadHistory();
    history.push({ role: "assistant", text: assistantText, recommendation: data.recommendation || null });
    saveHistory(history);
    updateSidePanel(data);
  } catch (err) {
    if (requestVersion !== conversationVersion) return;
    replaceBubble(loadingId, `${uiText("initFailed")} ${err.message}`);
  } finally {
    if (requestVersion !== conversationVersion) return;
    setSending(false);
    chatInput.focus();
  }
}

// ── Send message ─────────────────────────────────────────────────────────────

async function sendMessage(message) {
  const requestVersion = conversationVersion;
  
  const history = loadHistory();
  history.push({ role: "user", text: message });
  saveHistory(history);
  appendBubble("user", message);

  const loadingId = `loading-${Date.now()}`;
  appendBubble("assistant", uiText("thinking"), { id: loadingId, loading: true });
  startThinkingLabel(loadingId);
  setSending(true);

  const payload = {
    session_id:    loadSessionId(),
    message:       message,
    model_id:      getModelId(),
    language_hint: getLanguageHint()
  };

  try {
    const data = await postTurn(payload);
    if (requestVersion !== conversationVersion) return;
    saveSessionId(data.session_id);
    saveLastResponse(data);

    const assistantText = buildAssistantDialogueText(data);
    replaceBubble(loadingId, assistantText, data.recommendation);

    history.push({ role: "assistant", text: assistantText, recommendation: data.recommendation || null });
    saveHistory(history);
    updateSidePanel(data);
  } catch (err) {
    if (requestVersion !== conversationVersion) return;
    replaceBubble(loadingId, `${uiText("requestFailed")} ${err.message}`);
  } finally {
    if (requestVersion !== conversationVersion) return;
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
  resetConversation();
});

if (langSelect) {
  langSelect.addEventListener("change", () => {
    localStorage.setItem(STORAGE_KEYS.languagePreference, langSelect.value);
    const nextLanguage = getLanguageHint();
    if (nextLanguage === settingsState.language) return;
    settingsState.language = nextLanguage;
    // Start a fresh session so prompts and assistant language switch immediately.
    resetConversation();
  });
}

if (modelSelect) {
  modelSelect.addEventListener("change", () => {
    const nextModelId = getModelId();
    if (nextModelId === settingsState.modelId) return;
    settingsState.modelId = nextModelId;
    recordContextChangeMarker();
  });
}

// ── Boot ─────────────────────────────────────────────────────────────────────

(function init() {
  restoreLanguagePreference();
  settingsState.language = getLanguageHint();
  settingsState.modelId = getModelId();
  renderHistory();
  const lastResponseRaw = localStorage.getItem(STORAGE_KEYS.lastResponse);
  if (lastResponseRaw) {
    try { updateSidePanel(JSON.parse(lastResponseRaw)); } catch { /* ignore */ }
  }
  if (loadSessionId()) {
    sessionBadge.textContent = `${loadSessionId().slice(0, 8)}…`;
  } else {
    sessionBadge.textContent = uiText("noSession");
  }
  initializeChat();
})();
