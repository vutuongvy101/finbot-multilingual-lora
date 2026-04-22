const STORAGE_KEYS = {
    sessionId: "finbot.session_id",
    chatHistory: "finbot.chat_history",
    lastResponse: "finbot.last_response",
    languagePreference: "finbot.language_preference"
  };
  
  const API_BASE = window.APP_API_BASE || "http://127.0.0.1:8000";
  const TURN_URL = `${API_BASE}/chat/turn`;
  
  const chatList = document.getElementById("chatList");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const recommendationArea = document.getElementById("recommendationArea");
  const langSelect = document.getElementById("langSelect");
  
  const sessionBadge = document.getElementById("sessionBadge");
  const apiBaseLabel = document.getElementById("apiBaseLabel");
  const stateValue = document.getElementById("stateValue");
  const taskModeValue = document.getElementById("taskModeValue");
  const nextItemValue = document.getElementById("nextItemValue");
  const readyValue = document.getElementById("readyValue");
  const langValue = document.getElementById("langValue");
  const collectedJson = document.getElementById("collectedJson");
  
  apiBaseLabel.textContent = API_BASE;
  
  function loadSessionId() {
    return localStorage.getItem(STORAGE_KEYS.sessionId);
  }
  
  function saveSessionId(id) {
    if (id) localStorage.setItem(STORAGE_KEYS.sessionId, id);
  }
  
  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEYS.chatHistory) || "[]");
    } catch {
      return [];
    }
  }
  
  function saveHistory(history) {
    localStorage.setItem(STORAGE_KEYS.chatHistory, JSON.stringify(history));
  }
  
  function saveLastResponse(data) {
    localStorage.setItem(STORAGE_KEYS.lastResponse, JSON.stringify(data));
  }

  function getLanguageHint() {
    return langSelect?.value || "en";
  }

  function restoreLanguagePreference() {
    const saved = localStorage.getItem(STORAGE_KEYS.languagePreference);
    if (!saved || !langSelect) return;
    if (["en", "vi", "zh"].includes(saved)) {
      langSelect.value = saved;
    }
  }
  
  function appendBubble(role, text, { id = null, loading = false } = {}) {
    const div = document.createElement("div");
    div.className = `bubble ${role === "user" ? "bubble-user" : "bubble-assistant"}`;
    if (id) div.dataset.id = id;
  
    if (loading) {
      div.innerHTML = `
        <div class="d-flex align-items-center gap-2">
          <div class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></div>
          <span>${text}</span>
        </div>
      `;
    } else {
      div.textContent = text;
    }
  
    chatList.appendChild(div);
    chatList.scrollTop = chatList.scrollHeight;
  }
  
  function replaceBubble(id, text) {
    const target = chatList.querySelector(`[data-id="${id}"]`);
    if (!target) return;
    target.innerHTML = "";
    target.textContent = text;
  }
  
  function renderHistory() {
    chatList.innerHTML = "";
    const history = loadHistory();
    history.forEach((item) => appendBubble(item.role, item.text));
  }
  
  function renderRecommendation(rec) {
    recommendationArea.innerHTML = "";
    if (!rec) return;
  
    const card = document.createElement("div");
    card.className = "card shadow-sm recommendation-card";
    card.innerHTML = `
      <div class="card-body">
        <h5 class="card-title mb-3">Recommendation</h5>
        <p><strong>Profile Summary:</strong><br>${escapeHtml(rec.profile_summary)}</p>
        <p><strong>Recommendation:</strong><br>${escapeHtml(rec.recommendation)}</p>
        <p><strong>Reasoning:</strong><br>${escapeHtml(rec.reasoning)}</p>
        <p><strong>Risks & Caveats:</strong><br>${escapeHtml(rec.risks_caveats)}</p>
        <p><strong>Sources:</strong><br>${escapeHtml((rec.sources || []).join(", ") || "N/A")}</p>
        <p class="mb-0"><strong>Disclaimer:</strong><br>${escapeHtml(rec.disclaimer)}</p>
      </div>
    `;
    recommendationArea.appendChild(card);
  }
  
  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }
  
  function updateSidePanel(data) {
    sessionBadge.textContent = data.session_id ? `Session: ${data.session_id.slice(0, 8)}...` : "No session";
    stateValue.textContent = data.state || "-";
    taskModeValue.textContent = data.task_mode || "-";
    nextItemValue.textContent = data.next_item || "-";
    readyValue.textContent = String(Boolean(data.ready_for_recommendation));
    readyValue.className = `badge ${data.ready_for_recommendation ? "text-bg-success" : "text-bg-secondary"}`;
    langValue.textContent = data.detected_language || "-";
    collectedJson.textContent = JSON.stringify(data.collected || {}, null, 2);
  }
  
  function setSending(isSending) {
    chatInput.disabled = isSending;
    sendBtn.disabled = isSending;
  }

  async function initializeChat() {
    const existingHistory = loadHistory();
    if (existingHistory.length > 0) return;

    const loadingId = `loading-init-${Date.now()}`;
    appendBubble("assistant", "Starting chat...", { id: loadingId, loading: true });
    setSending(true);

    const payload = {
      session_id: loadSessionId(),
      message: "__INIT__",
      model_id: "qwen2.5-1.5b-instruct",
      language_hint: getLanguageHint()
    };

    try {
      const res = await fetch(TURN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`HTTP ${res.status}: ${body}`);
      }

      const data = await res.json();
      saveSessionId(data.session_id);
      saveLastResponse(data);
      replaceBubble(loadingId, data.assistant_message || "(empty response)");

      const history = [{ role: "assistant", text: data.assistant_message || "(empty response)" }];
      saveHistory(history);

      updateSidePanel(data);
      renderRecommendation(data.recommendation);
    } catch (err) {
      replaceBubble(loadingId, `Failed to initialize chat. ${err.message}`);
    } finally {
      setSending(false);
      chatInput.focus();
    }
  }
  
  async function sendMessage(message) {
    const history = loadHistory();
    history.push({ role: "user", text: message });
    saveHistory(history);
    appendBubble("user", message);
  
    const loadingId = `loading-${Date.now()}`;
    appendBubble("assistant", "Thinking...", { id: loadingId, loading: true });
    setSending(true);
  
    const payload = {
      session_id: loadSessionId(),
      message,
      model_id: "qwen2.5-1.5b-instruct",
      language_hint: getLanguageHint()
    };
  
    try {
      const res = await fetch(TURN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
  
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`HTTP ${res.status}: ${body}`);
      }
  
      const data = await res.json();
      saveSessionId(data.session_id);
      saveLastResponse(data);
  
      replaceBubble(loadingId, data.assistant_message || "(empty response)");
  
      history.push({ role: "assistant", text: data.assistant_message || "(empty response)" });
      saveHistory(history);
  
      updateSidePanel(data);
      renderRecommendation(data.recommendation);
    } catch (err) {
      replaceBubble(loadingId, `Request failed. ${err.message}`);
    } finally {
      setSending(false);
      chatInput.focus();
    }
  }
  
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
    recommendationArea.innerHTML = "";
    renderHistory();
    updateSidePanel({
      state: "-",
      task_mode: null,
      next_item: null,
      ready_for_recommendation: false,
      detected_language: "-",
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
  
  (function init() {
    restoreLanguagePreference();
    renderHistory();
    const lastResponseRaw = localStorage.getItem(STORAGE_KEYS.lastResponse);
    if (lastResponseRaw) {
      try {
        const data = JSON.parse(lastResponseRaw);
        updateSidePanel(data);
        renderRecommendation(data.recommendation);
      } catch {
        // ignore malformed cache
      }
    }
    if (loadSessionId()) {
      sessionBadge.textContent = `Session: ${loadSessionId().slice(0, 8)}...`;
    }
    initializeChat();
  })();