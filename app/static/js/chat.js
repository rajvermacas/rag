(() => {
  "use strict";

  if (typeof window.RagCommon === "undefined") {
    throw new Error("RagCommon is required before chat.js");
  }

  const { requireElement, requireString, requireNumber, requireErrorMessage, renderMarkdown, removeCitationArtifacts, escapeHtml } = window.RagCommon;
  const CHAT_STORAGE_KEY = "rag-chat-sessions";
  const CHAT_MODEL_SELECTION_STORAGE_KEY = "rag-chat-model-selection";
  const DEFAULT_CHAT_GREETING = "Hello! How can I assist you today?";
  const EMPTY_ASSISTANT_RESPONSE_MESSAGE =
    "I could not generate a response from the current context. Please rephrase your question.";
  const EMPTY_RESPONSE_SENTINELS = new Set(["empty response", "empty reponse"]);
  const CHAT_MODELS_ENDPOINT = "/models/chat";
  const MODEL_SELECTION_SEPARATOR = "||";

  const uploadForm = requireElement("upload-form");
  const uploadButton = requireElement("upload-button");
  const uploadLoader = requireElement("upload-loader");
  const uploadButtonLabel = requireElement("upload-button-label");
  const uploadStatus = requireElement("upload-status");
  const refreshDocuments = requireElement("refresh-documents");
  const documentsList = requireElement("documents-list");
  const documentsStatus = requireElement("documents-status");
  const chatForm = requireElement("chat-form");
  const chatButton = requireElement("chat-button");
  const chatWindow = requireElement("chat-window");
  const chatModelSelect = requireElement("chat-model-select");
  const chatHistorySelect = requireElement("chat-history-select");
  const clearChatButton = requireElement("clear-chat");

  const chatSessions = [];
  let activeSessionId = "";
  let conversationHistory = [];
  let restoredChatModelSelectionValue = "";
  syncChatSubmitEnabledState();

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    uploadButton.disabled = true;
    setUploadButtonLoadingState(true);
    setUploadStatus("hidden", "");
    try {
      const formData = new FormData(uploadForm);
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        setUploadStatus("error", `Upload failed: ${requireString(payload, "detail", "upload error")}`);
        return;
      }
      requireString(payload, "doc_id", "upload response");
      requireNumber(payload, "chunks_indexed", "upload response");
      setUploadStatus("success", "Document uploaded successfully.");
      uploadForm.reset();
      await loadDocuments();
    } finally {
      uploadButton.disabled = false;
      setUploadButtonLoadingState(false);
    }
  });

  refreshDocuments.addEventListener("click", async () => {
    await loadDocuments();
  });

  clearChatButton.addEventListener("click", () => {
    clearActiveChat();
  });

  chatHistorySelect.addEventListener("change", () => {
    const selectedSessionId = chatHistorySelect.value;
    setActiveSession(selectedSessionId);
    renderChatHistoryOptions();
    renderActiveSessionMessages();
    persistChatState();
  });

  chatModelSelect.addEventListener("change", () => {
    syncChatSubmitEnabledState();
    persistSelectedChatModelSelection();
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const formData = new FormData(chatForm);
      const rawMessage = formData.get("message");
      if (typeof rawMessage !== "string") {
        throw new Error("message field is required");
      }
      const message = rawMessage.trim();
      if (message === "") {
        return;
      }
      const selectedModel = readSelectedChatModelSelection();

      appendUserMessage(message);
      conversationHistory.push({ role: "user", message });
      persistChatState();
      chatForm.reset();
      syncChatSubmitEnabledState();

      const thinkingMessageId = appendThinkingMessage();
      try {
        const answer = await streamAssistantResponse(
          message,
          conversationHistory,
          selectedModel,
          thinkingMessageId
        );
        conversationHistory.push({ role: "assistant", message: answer });
        persistChatState();
      } catch (error) {
        removeMessageById(thinkingMessageId);
        appendSystemError(requireErrorMessage(error));
      }
    } catch (error) {
      appendSystemError(requireErrorMessage(error));
    }
  });

  window.addEventListener("load", async () => {
    initializeChatSessions();
    restorePersistedChatModelSelection();
    await loadChatModels();
    await loadDocuments();
  });

  function syncChatSubmitEnabledState() {
    if (typeof chatModelSelect.value !== "string") {
      throw new Error("chat model select value must be a string");
    }
    chatButton.disabled = chatModelSelect.value.trim() === "";
  }

  function readSelectedChatModelSelection() {
    if (typeof chatModelSelect.value !== "string") {
      throw new Error("chat model select value must be a string");
    }
    const normalizedSelectionValue = chatModelSelect.value.trim();
    if (normalizedSelectionValue === "") {
      throw new Error("chat model must be selected");
    }
    return parseChatModelSelectionValue(normalizedSelectionValue);
  }

  function parseChatModelSelectionValue(value) {
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error("chat model selection value must be a non-empty string");
    }
    const parts = value.split(MODEL_SELECTION_SEPARATOR);
    if (parts.length !== 2) {
      throw new Error("chat model selection value is malformed");
    }
    const backendId = parts[0].trim();
    const model = parts[1].trim();
    if (backendId === "" || model === "") {
      throw new Error("chat model selection value contains empty backend_id or model");
    }
    return { backend_id: backendId, model };
  }

  async function loadChatModels() {
    const response = await fetch(CHAT_MODELS_ENDPOINT);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(requireString(payload, "detail", "chat model list error"));
    }
    if (!("models" in payload) || !Array.isArray(payload.models)) {
      throw new Error("missing 'models' array in chat model list response");
    }
    renderChatModelOptions(payload.models);
  }

  function renderChatModelOptions(models) {
    validateChatModelList(models);
    chatModelSelect.innerHTML = "";
    chatModelSelect.append(createChatModelPlaceholderOption());
    models.forEach((model, index) => {
      chatModelSelect.append(createChatModelOption(model, index));
    });
    applyRestoredChatModelSelection();
    syncChatSubmitEnabledState();
  }

  function restorePersistedChatModelSelection() {
    const persistedSelectionValue = localStorage.getItem(CHAT_MODEL_SELECTION_STORAGE_KEY);
    if (persistedSelectionValue === null) {
      restoredChatModelSelectionValue = "";
      return;
    }
    const normalizedSelectionValue = persistedSelectionValue.trim();
    if (normalizedSelectionValue === "") {
      throw new Error("persisted chat model selection must not be empty");
    }
    parseChatModelSelectionValue(normalizedSelectionValue);
    restoredChatModelSelectionValue = normalizedSelectionValue;
  }

  function applyRestoredChatModelSelection() {
    if (restoredChatModelSelectionValue === "") {
      chatModelSelect.value = "";
      return;
    }
    chatModelSelect.value = restoredChatModelSelectionValue;
    if (chatModelSelect.value !== restoredChatModelSelectionValue) {
      throw new Error("persisted chat model selection is not available in chat model list");
    }
  }

  function persistSelectedChatModelSelection() {
    if (typeof chatModelSelect.value !== "string") {
      throw new Error("chat model select value must be a string");
    }
    const normalizedSelectionValue = chatModelSelect.value.trim();
    if (normalizedSelectionValue === "") {
      restoredChatModelSelectionValue = "";
      localStorage.removeItem(CHAT_MODEL_SELECTION_STORAGE_KEY);
      return;
    }
    parseChatModelSelectionValue(normalizedSelectionValue);
    restoredChatModelSelectionValue = normalizedSelectionValue;
    localStorage.setItem(CHAT_MODEL_SELECTION_STORAGE_KEY, normalizedSelectionValue);
  }

  function validateChatModelList(models) {
    if (models.length === 0) {
      throw new Error("chat model list must include at least one model");
    }
    const seen = new Set();
    models.forEach((modelOption, index) => {
      const normalizedOption = normalizeChatModelOption(modelOption, index);
      const modelKey = encodeChatModelSelectionValue(
        normalizedOption.backend_id,
        normalizedOption.model
      );
      if (seen.has(modelKey)) {
        throw new Error(`chat model list contains duplicate backend/model pair: ${modelKey}`);
      }
      seen.add(modelKey);
    });
  }

  function createChatModelPlaceholderOption() {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Select model";
    option.selected = true;
    return option;
  }

  function createChatModelOption(modelOption, index) {
    const normalizedOption = normalizeChatModelOption(modelOption, index);
    const option = document.createElement("option");
    option.value = encodeChatModelSelectionValue(
      normalizedOption.backend_id,
      normalizedOption.model
    );
    option.textContent = normalizedOption.label;
    return option;
  }

  function normalizeChatModelOption(modelOption, index) {
    const context = `models[${index}]`;
    if (typeof modelOption !== "object" || modelOption === null || Array.isArray(modelOption)) {
      throw new Error(`chat model option must be an object at ${context}`);
    }
    const backendId = requireString(modelOption, "backend_id", context).trim();
    const provider = requireString(modelOption, "provider", context).trim();
    const model = requireString(modelOption, "model", context).trim();
    const label = requireString(modelOption, "label", context).trim();
    if (backendId === "") {
      throw new Error(`chat model backend_id must not be empty at ${context}`);
    }
    if (provider === "") {
      throw new Error(`chat model provider must not be empty at ${context}`);
    }
    if (model === "") {
      throw new Error(`chat model id must not be empty at ${context}`);
    }
    if (label === "") {
      throw new Error(`chat model label must not be empty at ${context}`);
    }
    return { backend_id: backendId, provider, model, label };
  }

  function encodeChatModelSelectionValue(backendId, model) {
    if (typeof backendId !== "string" || backendId.trim() === "") {
      throw new Error("chat model backend_id must be a non-empty string");
    }
    if (typeof model !== "string" || model.trim() === "") {
      throw new Error("chat model id must be a non-empty string");
    }
    if (backendId.includes(MODEL_SELECTION_SEPARATOR)) {
      throw new Error(`chat model backend_id must not include '${MODEL_SELECTION_SEPARATOR}'`);
    }
    if (model.includes(MODEL_SELECTION_SEPARATOR)) {
      throw new Error(`chat model id must not include '${MODEL_SELECTION_SEPARATOR}'`);
    }
    return `${backendId}${MODEL_SELECTION_SEPARATOR}${model}`;
  }

  function setUploadButtonLoadingState(isLoading) {
    if (typeof isLoading !== "boolean") {
      throw new Error("upload loading state must be a boolean");
    }
    if (isLoading) {
      uploadLoader.classList.remove("hidden");
      uploadButtonLabel.textContent = "Uploading...";
      return;
    }
    uploadLoader.classList.add("hidden");
    uploadButtonLabel.textContent = "Upload";
  }

  function setUploadStatus(statusType, message) {
    if (typeof statusType !== "string") {
      throw new Error("upload status type must be a string");
    }
    if (typeof message !== "string") {
      throw new Error("upload status message must be a string");
    }
    if (statusType === "hidden") {
      uploadStatus.className = "mt-3 hidden rounded-lg border px-3 py-2 text-sm font-medium";
      uploadStatus.textContent = "";
      return;
    }
    if (statusType === "success") {
      uploadStatus.className = "mt-3 rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm font-medium text-emerald-800";
      uploadStatus.textContent = message;
      return;
    }
    if (statusType === "error") {
      uploadStatus.className = "mt-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm font-medium text-red-700";
      uploadStatus.textContent = message;
      return;
    }
    if (statusType === "info") {
      uploadStatus.className = "mt-3 rounded-lg border border-zinc-300 bg-zinc-100 px-3 py-2 text-sm font-medium text-zinc-700";
      uploadStatus.textContent = message;
      return;
    }
    throw new Error(`unsupported upload status type: ${statusType}`);
  }

  function initializeChatSessions() {
    const storedState = loadPersistedChatState();
    if (storedState !== null) {
      restoreChatSessions(storedState);
      renderChatHistoryOptions();
      renderActiveSessionMessages();
      return;
    }

    const session = createChatSession("Current Chat");
    chatSessions.push(session);
    setActiveSession(session.id);
    appendAssistantMessage(DEFAULT_CHAT_GREETING);
    renderChatHistoryOptions();
    renderActiveSessionMessages();
    persistChatState();
  }

  function createChatSession(label) {
    if (typeof label !== "string" || label.trim() === "") {
      throw new Error("chat session label is required");
    }
    return {
      id: createSessionId(),
      label,
      history: [],
      messages: [],
    };
  }

  function createSessionId() {
    if (typeof crypto === "undefined" || typeof crypto.randomUUID !== "function") {
      throw new Error("crypto.randomUUID is required for chat sessions");
    }
    return crypto.randomUUID();
  }

  function loadPersistedChatState() {
    const rawState = localStorage.getItem(CHAT_STORAGE_KEY);
    if (rawState === null) {
      return null;
    }
    const parsedState = JSON.parse(rawState);
    validatePersistedChatState(parsedState);
    return parsedState;
  }

  function validatePersistedChatState(parsedState) {
    if (typeof parsedState !== "object" || parsedState === null) {
      throw new Error("persisted chat state must be an object");
    }
    if (!("activeSessionId" in parsedState)) {
      throw new Error("persisted chat state missing activeSessionId");
    }
    if (!("sessions" in parsedState)) {
      throw new Error("persisted chat state missing sessions");
    }
    if (typeof parsedState.activeSessionId !== "string") {
      throw new Error("persisted activeSessionId must be a string");
    }
    if (!Array.isArray(parsedState.sessions)) {
      throw new Error("persisted sessions must be an array");
    }
  }

  function restoreChatSessions(state) {
    state.sessions.forEach((sessionRecord, index) => {
      validateSessionRecord(sessionRecord, index);
      chatSessions.push({
        id: sessionRecord.id,
        label: sessionRecord.label,
        history: sessionRecord.history,
        messages: sessionRecord.messages,
      });
    });
    if (chatSessions.length === 0) {
      throw new Error("persisted sessions must include at least one chat");
    }
    setActiveSession(state.activeSessionId);
  }

  function validateSessionRecord(sessionRecord, index) {
    const context = `session[${index}]`;
    requireString(sessionRecord, "id", context);
    requireString(sessionRecord, "label", context);
    if (!("history" in sessionRecord) || !Array.isArray(sessionRecord.history)) {
      throw new Error(`missing 'history' array in ${context}`);
    }
    if (!("messages" in sessionRecord) || !Array.isArray(sessionRecord.messages)) {
      throw new Error(`missing 'messages' array in ${context}`);
    }
    sessionRecord.history.forEach((turn, turnIndex) => {
      const turnContext = `${context}.history[${turnIndex}]`;
      const role = requireString(turn, "role", turnContext);
      requireString(turn, "message", turnContext);
      if (role !== "user" && role !== "assistant") {
        throw new Error(`history role must be user or assistant in ${turnContext}`);
      }
    });
    sessionRecord.messages.forEach((entry, messageIndex) => {
      const messageContext = `${context}.messages[${messageIndex}]`;
      const role = requireString(entry, "role", messageContext);
      requireString(entry, "text", messageContext);
      if (!["user", "assistant", "error"].includes(role)) {
        throw new Error(`unsupported message role in ${messageContext}: ${role}`);
      }
    });
  }

  function persistChatState() {
    const payload = {
      activeSessionId,
      sessions: buildPersistableSessions(),
    };
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(payload));
  }

  function buildPersistableSessions() {
    return chatSessions.map((session, index) => {
      const persistableSession = {
        id: session.id,
        label: session.label,
        history: session.history,
        messages: session.messages.filter((entry) => entry.role !== "thinking"),
      };
      validateSessionRecord(persistableSession, index);
      return persistableSession;
    });
  }

  function clearActiveChat() {
    const activeSession = getActiveSession();
    if (isEmptyOrPristineGreetingSession(activeSession)) {
      return;
    }
    activeSession.label = `Chat ${new Date().toLocaleString()}`;
    const newSession = createChatSession("Current Chat");
    chatSessions.unshift(newSession);
    setActiveSession(newSession.id);
    appendAssistantMessage(DEFAULT_CHAT_GREETING);
    renderChatHistoryOptions();
    renderActiveSessionMessages();
    persistChatState();
  }

  function isEmptyOrPristineGreetingSession(session) {
    const hasNoMessages = session.messages.length === 0;
    const hasNoHistory = session.history.length === 0;
    if (hasNoMessages && hasNoHistory) {
      return true;
    }
    if (!hasNoHistory || session.messages.length !== 1) {
      return false;
    }
    const [onlyMessage] = session.messages;
    return onlyMessage.role === "assistant" && onlyMessage.text === DEFAULT_CHAT_GREETING;
  }

  function renderChatHistoryOptions() {
    chatHistorySelect.innerHTML = "";
    chatSessions.forEach((session) => {
      const option = document.createElement("option");
      option.value = session.id;
      option.textContent = session.label;
      if (session.id === activeSessionId) {
        option.selected = true;
      }
      chatHistorySelect.append(option);
    });
  }

  function setActiveSession(sessionId) {
    const session = getSessionById(sessionId);
    activeSessionId = session.id;
    conversationHistory = session.history;
  }

  function getSessionById(sessionId) {
    const session = chatSessions.find((item) => item.id === sessionId);
    if (typeof session === "undefined") {
      throw new Error(`chat session not found: ${sessionId}`);
    }
    return session;
  }

  function getActiveSession() {
    if (activeSessionId === "") {
      throw new Error("active chat session is not set");
    }
    return getSessionById(activeSessionId);
  }

  function appendUserMessage(message) {
    addMessageToActiveSession("user", message);
  }

  function appendAssistantMessage(markdownAnswer) {
    addMessageToActiveSession("assistant", removeCitationArtifacts(markdownAnswer));
  }

  function appendThinkingMessage() {
    return addMessageToActiveSession("thinking", "Thinking...");
  }

  function appendSystemError(errorText) {
    addMessageToActiveSession("error", errorText);
  }

  function addMessageToActiveSession(role, text) {
    if (typeof role !== "string") {
      throw new Error("message role must be a string");
    }
    if (typeof text !== "string") {
      throw new Error("message text must be a string");
    }
    const session = getActiveSession();
    const messageId = createSessionId();
    session.messages.push({ id: messageId, role, text });
    renderActiveSessionMessages();
    persistChatState();
    return messageId;
  }

  function removeMessageById(messageId) {
    const session = getActiveSession();
    const entryIndex = session.messages.findIndex((entry) => entry.id === messageId);
    if (entryIndex === -1) {
      throw new Error(`chat message not found: ${messageId}`);
    }
    session.messages.splice(entryIndex, 1);
    renderActiveSessionMessages();
    persistChatState();
  }

  function upsertStreamedAssistantMessage(messageId, rawText) {
    const session = getActiveSession();
    const entry = session.messages.find((item) => item.id === messageId);
    if (typeof entry === "undefined") {
      throw new Error(`chat message not found: ${messageId}`);
    }
    entry.role = "assistant";
    entry.text = removeCitationArtifacts(rawText);
    renderActiveSessionMessages();
    persistChatState();
  }

  function renderActiveSessionMessages() {
    const session = getActiveSession();
    chatWindow.innerHTML = "";
    session.messages.forEach((entry) => {
      chatWindow.append(buildMessageElement(entry));
    });
    scrollChatToBottom();
  }

  function buildMessageElement(entry) {
    if (!("role" in entry) || !("text" in entry)) {
      throw new Error("chat entry must include role and text");
    }
    if (entry.role === "user") {
      return buildUserMessageElement(entry.text);
    }
    if (entry.role === "assistant") {
      return buildAssistantMessageElement(entry.text);
    }
    if (entry.role === "thinking") {
      return buildThinkingMessageElement();
    }
    if (entry.role === "error") {
      return buildErrorMessageElement(entry.text);
    }
    throw new Error(`unsupported chat entry role: ${entry.role}`);
  }

  function buildUserMessageElement(message) {
    const row = document.createElement("div");
    row.className = "mb-3 flex justify-end";
    const wrapper = document.createElement("article");
    wrapper.className = "max-w-[85%] rounded-xl border border-zinc-300 bg-zinc-200/90 p-3";
    wrapper.innerHTML = `<p class="text-right text-xs font-semibold uppercase tracking-wide text-zinc-600">You</p><p class="mt-1 text-right text-sm text-zinc-900">${escapeHtml(message)}</p>`;
    row.append(wrapper);
    return row;
  }

  function buildAssistantMessageElement(markdownAnswer) {
    const row = document.createElement("div");
    row.className = "mb-3 flex justify-start";
    const wrapper = document.createElement("article");
    wrapper.className = "max-w-[85%] rounded-xl border border-red-200 bg-white p-3 shadow-sm";
    const markdownHtml = renderMarkdown(removeCitationArtifacts(markdownAnswer));
    wrapper.innerHTML = `<p class="text-xs font-semibold uppercase tracking-wide text-red-600">Assistant</p><div class="markdown-body mt-2 text-sm text-zinc-800">${markdownHtml}</div>`;
    row.append(wrapper);
    return row;
  }

  function buildThinkingMessageElement() {
    const row = document.createElement("div");
    row.className = "mb-3 flex justify-start";
    const wrapper = document.createElement("article");
    wrapper.className = "max-w-[85%] rounded-xl border border-zinc-200 bg-white p-3 shadow-sm";
    wrapper.innerHTML = "<p class=\"text-xs font-semibold uppercase tracking-wide text-zinc-500\">Assistant</p><p class=\"mt-2 text-sm text-zinc-600 animate-pulse\">Thinking...</p>";
    row.append(wrapper);
    return row;
  }

  function buildErrorMessageElement(errorText) {
    const row = document.createElement("div");
    row.className = "mb-3 flex justify-start";
    const wrapper = document.createElement("article");
    wrapper.className = "max-w-[85%] rounded-xl border border-red-300 bg-red-50 p-3";
    wrapper.innerHTML = `<p class="text-xs font-semibold uppercase tracking-wide text-red-600">Error</p><p class="mt-1 text-sm text-red-700">${escapeHtml(errorText)}</p>`;
    row.append(wrapper);
    return row;
  }

  async function streamAssistantResponse(message, history, modelSelection, thinkingMessageId) {
    if (typeof modelSelection !== "object" || modelSelection === null || Array.isArray(modelSelection)) {
      throw new Error("chat model selection must be an object");
    }
    const backendId = requireString(modelSelection, "backend_id", "chat model selection");
    const model = requireString(modelSelection, "model", "chat model selection");
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history, backend_id: backendId, model }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(requireString(payload, "detail", "chat stream error"));
    }
    if (response.body === null) {
      throw new Error("chat stream response body is missing");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      const chunk = decoder.decode(value, { stream: true });
      if (chunk === "") {
        continue;
      }
      accumulatedText += chunk;
      upsertStreamedAssistantMessage(thinkingMessageId, accumulatedText);
    }

    const trailingChunk = decoder.decode();
    if (trailingChunk !== "") {
      accumulatedText += trailingChunk;
      upsertStreamedAssistantMessage(thinkingMessageId, accumulatedText);
    }

    const finalAnswer = removeCitationArtifacts(accumulatedText).trim();
    if (!isEffectivelyEmptyAssistantAnswer(finalAnswer)) {
      return finalAnswer;
    }
    upsertStreamedAssistantMessage(thinkingMessageId, EMPTY_ASSISTANT_RESPONSE_MESSAGE);
    return EMPTY_ASSISTANT_RESPONSE_MESSAGE;
  }

  function isEffectivelyEmptyAssistantAnswer(text) {
    if (typeof text !== "string") {
      throw new Error("assistant answer must be a string");
    }
    const normalized = text.trim().toLowerCase();
    return normalized === "" || EMPTY_RESPONSE_SENTINELS.has(normalized);
  }

  async function loadDocuments() {
    documentsStatus.textContent = "Refreshing indexed documents...";
    const response = await fetch("/documents");
    const payload = await response.json();
    if (!response.ok) {
      documentsStatus.textContent = `Failed to load documents: ${requireString(payload, "detail", "document list error")}`;
      return;
    }
    if (!("documents" in payload) || !Array.isArray(payload.documents)) {
      throw new Error("missing 'documents' array in document list response");
    }
    renderDocuments(payload.documents);
  }

  function renderDocuments(documents) {
    documentsList.innerHTML = "";
    if (documents.length === 0) {
      const emptyItem = document.createElement("li");
      emptyItem.className = "rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-500";
      emptyItem.textContent = "No indexed documents yet.";
      documentsList.append(emptyItem);
      documentsStatus.textContent = "0 indexed documents.";
      return;
    }

    documents.forEach((documentRecord, index) => {
      const context = `document[${index}]`;
      const filename = requireString(documentRecord, "filename", context);
      const docId = requireString(documentRecord, "doc_id", context);
      const row = document.createElement("li");
      row.className = "flex items-center justify-between gap-3 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2";
      row.innerHTML = `
        <span class="truncate text-sm font-medium text-zinc-800">${escapeHtml(filename)}</span>
        <button
          class="delete-document rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-red-700 transition hover:bg-red-50"
          type="button"
          data-doc-id="${escapeHtml(docId)}"
        >
          Delete
        </button>
      `;
      const deleteButton = row.querySelector(".delete-document");
      if (deleteButton === null) {
        throw new Error("missing delete button in document row");
      }
      deleteButton.addEventListener("click", async () => {
        await deleteDocument(docId, filename);
      });
      documentsList.append(row);
    });
    documentsStatus.textContent = `${documents.length} indexed document(s).`;
  }

  async function deleteDocument(docId, filename) {
    const shouldDelete = window.confirm(`Delete '${filename}' and remove its indexed chunks?`);
    if (!shouldDelete) {
      return;
    }
    documentsStatus.textContent = `Deleting ${filename}...`;
    const response = await fetch(`/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
      documentsStatus.textContent = `Delete failed: ${requireString(payload, "detail", "delete document error")}`;
      return;
    }
    const chunksDeleted = requireNumber(payload, "chunks_deleted", "delete document response");
    documentsStatus.textContent = `Deleted ${filename}. Removed ${chunksDeleted} chunk(s) from index.`;
    await loadDocuments();
  }

  function scrollChatToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
})();
