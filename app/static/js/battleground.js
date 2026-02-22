(() => {
  "use strict";

  if (typeof window.RagCommon === "undefined") {
    throw new Error("RagCommon is required before battleground.js");
  }

  const {
    requireElement,
    requireString,
    requireErrorMessage,
    removeCitationArtifacts,
  } = window.RagCommon;
  const logger = console;
  const BATTLEGROUND_STORAGE_KEY = "rag-battleground-state";

  const navChat = requireElement("nav-chat");
  const navBattleground = requireElement("nav-battleground");
  const chatSection = requireElement("chat-section");
  const battlegroundSection = requireElement("battleground-section");
  const battlegroundForm = requireElement("battleground-form");
  const battlegroundMessage = requireElement("battleground-message");
  const battlegroundSubmit = requireElement("battleground-submit");
  const clearBattlegroundChatButton = requireElement("clear-battleground-chat");
  const modelASelect = requireElement("model-a-select");
  const modelBSelect = requireElement("model-b-select");
  const battlegroundStatus = requireElement("battleground-status");
  const modelAOutput = requireElement("battleground-model-a-output");
  const modelBOutput = requireElement("battleground-model-b-output");

  const state = createInitialState();

  initializeTabNavigation();
  void initializeBattleground().catch(handleInitializationFailure);

  function createInitialState() {
    return {
      selectedModelA: "",
      selectedModelB: "",
      isModelSelectionLocked: false,
      historyA: [],
      historyB: [],
      messagesA: [],
      messagesB: [],
      isSubmitting: false,
    };
  }

  function initializeTabNavigation() {
    navChat.addEventListener("click", () => {
      setActiveTab("chat");
    });
    navBattleground.addEventListener("click", () => {
      setActiveTab("battleground");
    });
    setActiveTab("chat");
  }

  function setActiveTab(tabId) {
    if (tabId === "chat") {
      navChat.setAttribute("aria-selected", "true");
      navChat.className = "rounded-xl border border-zinc-900 bg-zinc-900 px-4 py-2 text-sm font-semibold text-white transition";
      navBattleground.setAttribute("aria-selected", "false");
      navBattleground.className = "rounded-xl border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:border-red-500 hover:text-red-600";
      chatSection.classList.remove("hidden");
      battlegroundSection.classList.add("hidden");
      return;
    }
    if (tabId === "battleground") {
      navChat.setAttribute("aria-selected", "false");
      navChat.className = "rounded-xl border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:border-red-500 hover:text-red-600";
      navBattleground.setAttribute("aria-selected", "true");
      navBattleground.className = "rounded-xl border border-zinc-900 bg-zinc-900 px-4 py-2 text-sm font-semibold text-white transition";
      chatSection.classList.add("hidden");
      battlegroundSection.classList.remove("hidden");
      return;
    }
    throw new Error(`unsupported tab id: ${tabId}`);
  }

  async function initializeBattleground() {
    ensureSelectHasPlaceholder(modelASelect);
    ensureSelectHasPlaceholder(modelBSelect);
    battlegroundForm.addEventListener("submit", handleCompareSubmit);
    clearBattlegroundChatButton.addEventListener("click", handleClearBattlegroundChat);
    modelASelect.addEventListener("change", handleModelSelectChange);
    modelBSelect.addEventListener("change", handleModelSelectChange);
    await loadModelOptions();
    restorePersistedBattlegroundState();
    renderOutputs();
  }

  function handleInitializationFailure(error) {
    const message = requireErrorMessage(error);
    setBattlegroundStatus(message);
    logger.error("battleground_initialization_failed error=%s", message);
  }

  async function loadModelOptions() {
    logger.info("battleground_models_loading_started");
    setBattlegroundStatus("Loading battleground models...");
    const response = await fetch("/models/battleground");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(requireString(payload, "detail", "battleground model list error"));
    }
    if (!("models" in payload) || !Array.isArray(payload.models)) {
      throw new Error("missing 'models' array in battleground model list response");
    }
    const models = payload.models.map((item, index) => {
      if (typeof item !== "string" || item.trim() === "") {
        throw new Error(`model id at index ${index} must be a non-empty string`);
      }
      return item;
    });
    populateModelSelect(modelASelect, "Select model A", models);
    populateModelSelect(modelBSelect, "Select model B", models);
    setBattlegroundStatus("Models loaded. Start comparison.");
    logger.info("battleground_models_loading_completed model_count=%s", models.length);
  }

  function populateModelSelect(selectElement, placeholderLabel, models) {
    if (typeof placeholderLabel !== "string" || placeholderLabel.trim() === "") {
      throw new Error("model placeholder label must be a non-empty string");
    }
    if (!Array.isArray(models)) {
      throw new Error("models must be an array");
    }
    selectElement.innerHTML = "";
    const placeholderOption = document.createElement("option");
    placeholderOption.value = "";
    placeholderOption.textContent = placeholderLabel;
    placeholderOption.selected = true;
    selectElement.append(placeholderOption);
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      selectElement.append(option);
    });
    selectElement.value = "";
  }

  function handleModelSelectChange() {
    if (state.isModelSelectionLocked) {
      return;
    }
    state.selectedModelA = modelASelect.value.trim();
    state.selectedModelB = modelBSelect.value.trim();
    persistBattlegroundState();
  }

  function restorePersistedBattlegroundState() {
    const persistedState = loadPersistedBattlegroundState();
    if (persistedState === null) {
      return;
    }
    validatePersistedBattlegroundState(persistedState);
    state.selectedModelA = persistedState.selectedModelA;
    state.selectedModelB = persistedState.selectedModelB;
    state.isModelSelectionLocked = persistedState.isModelSelectionLocked;
    state.historyA = persistedState.historyA;
    state.historyB = persistedState.historyB;
    state.messagesA = persistedState.messagesA;
    state.messagesB = persistedState.messagesB;
    applyRestoredModelState();
    logger.info(
      "battleground_state_restored history_turns_a=%s history_turns_b=%s",
      state.historyA.length,
      state.historyB.length
    );
  }

  function loadPersistedBattlegroundState() {
    const rawState = localStorage.getItem(BATTLEGROUND_STORAGE_KEY);
    if (rawState === null) {
      return null;
    }
    const parsedState = JSON.parse(rawState);
    if (typeof parsedState !== "object" || parsedState === null) {
      throw new Error("persisted battleground state must be an object");
    }
    return parsedState;
  }

  function validatePersistedBattlegroundState(parsedState) {
    requireString(parsedState, "selectedModelA", "persisted battleground state");
    requireString(parsedState, "selectedModelB", "persisted battleground state");
    if (!("isModelSelectionLocked" in parsedState)) {
      throw new Error("persisted battleground state missing isModelSelectionLocked");
    }
    if (typeof parsedState.isModelSelectionLocked !== "boolean") {
      throw new Error("persisted battleground state isModelSelectionLocked must be a boolean");
    }
    validateTurnList(parsedState, "historyA");
    validateTurnList(parsedState, "historyB");
    validateMessageList(parsedState, "messagesA");
    validateMessageList(parsedState, "messagesB");
  }

  function validateTurnList(stateObject, key) {
    if (!(key in stateObject) || !Array.isArray(stateObject[key])) {
      throw new Error(`persisted battleground state missing array '${key}'`);
    }
    stateObject[key].forEach((turn, index) => {
      const context = `${key}[${index}]`;
      const role = requireString(turn, "role", context);
      const message = requireString(turn, "message", context).trim();
      if (!["user", "assistant"].includes(role)) {
        throw new Error(`unsupported role in ${context}: ${role}`);
      }
      if (message === "") {
        throw new Error(`message in ${context} must not be empty`);
      }
    });
  }

  function validateMessageList(stateObject, key) {
    if (!(key in stateObject) || !Array.isArray(stateObject[key])) {
      throw new Error(`persisted battleground state missing array '${key}'`);
    }
    stateObject[key].forEach((entry, index) => {
      const context = `${key}[${index}]`;
      const role = requireString(entry, "role", context);
      const text = requireString(entry, "text", context).trim();
      if (!["user", "assistant", "error"].includes(role)) {
        throw new Error(`unsupported message role in ${context}: ${role}`);
      }
      if (text === "") {
        throw new Error(`message text in ${context} must not be empty`);
      }
    });
  }

  function applyRestoredModelState() {
    if (state.selectedModelA !== "") {
      ensureSelectIncludesValue(modelASelect, state.selectedModelA);
      modelASelect.value = state.selectedModelA;
    }
    if (state.selectedModelB !== "") {
      ensureSelectIncludesValue(modelBSelect, state.selectedModelB);
      modelBSelect.value = state.selectedModelB;
    }
    applyModelLockState();
  }

  function ensureSelectIncludesValue(selectElement, value) {
    const optionValues = Array.from(selectElement.options).map((option) => option.value);
    if (!optionValues.includes(value)) {
      throw new Error(`select '${selectElement.id}' missing option value '${value}'`);
    }
  }

  async function handleCompareSubmit(event) {
    event.preventDefault();
    if (state.isSubmitting) {
      throw new Error("battleground compare request already in progress");
    }
    state.isSubmitting = true;
    battlegroundSubmit.disabled = true;
    try {
      const message = readRequiredMessage();
      const modelA = readRequiredModel(modelASelect, "Choose a model for Model A.");
      const modelB = readRequiredModel(modelBSelect, "Choose a model for Model B.");
      validateModelPair(modelA, modelB);
      lockModelsIfNeeded(modelA, modelB);
      appendUserTurn(message);
      appendThinkingMessages();
      setBattlegroundStatus("Comparing models...");
      logger.info(
        "battleground_turn_started model_a=%s model_b=%s history_turns_a=%s history_turns_b=%s",
        modelA,
        modelB,
        state.historyA.length,
        state.historyB.length
      );
      const terminalState = await streamComparison(message, modelA, modelB);
      const erroredSides = listErroredSides(terminalState);
      finalizeTurnHistories(terminalState, erroredSides);
      setBattlegroundStatus(buildCompletionStatus(erroredSides));
      logger.info("battleground_turn_completed errors=%s", erroredSides.join(","));
    } catch (error) {
      removeThinkingMessages();
      const message = requireErrorMessage(error);
      setBattlegroundStatus(message);
      logger.error("battleground_turn_failed error=%s", message);
    } finally {
      state.isSubmitting = false;
      battlegroundSubmit.disabled = false;
      persistBattlegroundState();
      renderOutputs();
    }
  }

  function readRequiredMessage() {
    if (typeof battlegroundMessage.value !== "string") {
      throw new Error("battleground message input value must be a string");
    }
    const message = battlegroundMessage.value.trim();
    if (message === "") {
      throw new Error("Enter a prompt before starting comparison.");
    }
    return message;
  }

  function readRequiredModel(selectElement, emptyErrorMessage) {
    if (typeof emptyErrorMessage !== "string" || emptyErrorMessage.trim() === "") {
      throw new Error("model empty error message must be a non-empty string");
    }
    if (typeof selectElement.value !== "string") {
      throw new Error(`select '${selectElement.id}' value must be a string`);
    }
    const model = selectElement.value.trim();
    if (model === "") {
      throw new Error(emptyErrorMessage);
    }
    return model;
  }

  function validateModelPair(modelA, modelB) {
    if (modelA === modelB) {
      throw new Error("Model A and Model B must be different.");
    }
    if (
      state.isModelSelectionLocked &&
      (state.selectedModelA !== modelA || state.selectedModelB !== modelB)
    ) {
      throw new Error("Model selection is locked. Start a new battleground chat to change models.");
    }
  }

  function lockModelsIfNeeded(modelA, modelB) {
    if (state.isModelSelectionLocked) {
      return;
    }
    state.selectedModelA = modelA;
    state.selectedModelB = modelB;
    state.isModelSelectionLocked = true;
    applyModelLockState();
  }

  function applyModelLockState() {
    modelASelect.disabled = state.isModelSelectionLocked;
    modelBSelect.disabled = state.isModelSelectionLocked;
  }

  function appendUserTurn(message) {
    const userTurn = { role: "user", message };
    state.historyA.push(userTurn);
    state.historyB.push(userTurn);
    state.messagesA.push({ role: "user", text: message });
    state.messagesB.push({ role: "user", text: message });
    battlegroundMessage.value = "";
    persistBattlegroundState();
    renderOutputs();
  }

  function appendThinkingMessages() {
    state.messagesA.push({ role: "thinking", text: "Thinking..." });
    state.messagesB.push({ role: "thinking", text: "Thinking..." });
    renderOutputs();
  }

  function removeThinkingMessages() {
    state.messagesA = state.messagesA.filter((entry) => entry.role !== "thinking");
    state.messagesB = state.messagesB.filter((entry) => entry.role !== "thinking");
  }

  async function streamComparison(message, modelA, modelB) {
    const terminalState = createTerminalState();
    const response = await fetch("/battleground/compare/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildComparePayload(message, modelA, modelB)),
    });
    if (!response.ok) {
      const errorPayload = await response.json();
      throw new Error(requireString(errorPayload, "detail", "battleground compare error"));
    }
    if (response.body === null) {
      throw new Error("battleground compare response body is missing");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    while (true) {
      const readResult = await reader.read();
      if (readResult.done) {
        break;
      }
      buffered += decoder.decode(readResult.value, { stream: true });
      buffered = consumeNdjsonBuffer(buffered, false, terminalState);
    }
    buffered += decoder.decode();
    consumeNdjsonBuffer(buffered, true, terminalState);
    assertTerminalStateComplete(terminalState);
    return terminalState;
  }

  function buildComparePayload(message, modelA, modelB) {
    return {
      message,
      history_a: cloneHistory(state.historyA),
      history_b: cloneHistory(state.historyB),
      model_a: modelA,
      model_b: modelB,
    };
  }

  function cloneHistory(history) {
    return history.map((turn) => ({
      role: turn.role,
      message: turn.message,
    }));
  }

  function consumeNdjsonBuffer(buffered, allowTrailingLine, terminalState) {
    if (typeof buffered !== "string") {
      throw new Error("ndjson buffer must be a string");
    }
    if (typeof allowTrailingLine !== "boolean") {
      throw new Error("allowTrailingLine must be a boolean");
    }
    requireTerminalState(terminalState);
    const lines = buffered.split("\n");
    let trailing = lines.pop();
    if (typeof trailing !== "string") {
      throw new Error("ndjson trailing segment must be a string");
    }
    if (allowTrailingLine && trailing.trim() !== "") {
      lines.push(trailing);
      trailing = "";
    }
    lines.forEach((line) => {
      if (line.trim() === "") {
        return;
      }
      applyStreamEvent(line, terminalState);
    });
    return trailing;
  }

  function applyStreamEvent(rawLine, terminalState) {
    requireTerminalState(terminalState);
    let event;
    try {
      event = JSON.parse(rawLine);
    } catch (error) {
      throw new Error(`invalid battleground NDJSON line: ${rawLine}`);
    }
    if (typeof event !== "object" || event === null || Array.isArray(event)) {
      throw new Error("battleground event must be an object");
    }
    const side = requireString(event, "side", "battleground event");
    const hasChunk = Object.prototype.hasOwnProperty.call(event, "chunk");
    const hasDone = Object.prototype.hasOwnProperty.call(event, "done");
    const hasError = Object.prototype.hasOwnProperty.call(event, "error");
    if ([hasChunk, hasDone, hasError].filter((value) => value).length !== 1) {
      throw new Error("battleground event must include exactly one of chunk, done, or error");
    }
    if (hasChunk) {
      const chunk = requireString(event, "chunk", "battleground chunk event");
      const cleanedChunk = upsertStreamingMessage(side, chunk);
      appendStreamResponse(terminalState, side, cleanedChunk);
      return;
    }
    if (hasDone) {
      if (event.done !== true) {
        throw new Error("battleground done event must set done=true");
      }
      terminalState[side] = true;
      return;
    }
    terminalState[side] = true;
    markSideAsErrored(terminalState, side);
    upsertErrorMessage(side, requireString(event, "error", "battleground error event"));
  }

  function upsertStreamingMessage(side, chunk) {
    const cleanedChunk = removeCitationArtifacts(chunk);
    if (cleanedChunk === "") {
      return "";
    }
    const sideMessages = getMessagesForSide(side);
    const lastEntry = getLastEntry(sideMessages);
    if (lastEntry === null) {
      sideMessages.push({ role: "assistant", text: cleanedChunk });
      renderOutputs();
      return cleanedChunk;
    }
    if (lastEntry.role === "thinking") {
      lastEntry.role = "assistant";
      lastEntry.text = cleanedChunk;
      renderOutputs();
      return cleanedChunk;
    }
    if (lastEntry.role === "assistant") {
      lastEntry.text = `${lastEntry.text}${cleanedChunk}`;
      renderOutputs();
      return cleanedChunk;
    }
    sideMessages.push({ role: "assistant", text: cleanedChunk });
    renderOutputs();
    return cleanedChunk;
  }

  function upsertErrorMessage(side, errorText) {
    const sideMessages = getMessagesForSide(side);
    const lastEntry = getLastEntry(sideMessages);
    if (lastEntry !== null && ["thinking", "assistant"].includes(lastEntry.role)) {
      lastEntry.role = "error";
      lastEntry.text = errorText;
      renderOutputs();
      return;
    }
    sideMessages.push({ role: "error", text: errorText });
    renderOutputs();
  }

  function getMessagesForSide(side) {
    if (side === "A") {
      return state.messagesA;
    }
    if (side === "B") {
      return state.messagesB;
    }
    throw new Error(`unsupported battleground side: ${side}`);
  }

  function createTerminalState() {
    return {
      A: false,
      B: false,
      errorA: false,
      errorB: false,
      responseA: "",
      responseB: "",
    };
  }

  function requireTerminalState(terminalState) {
    if (typeof terminalState !== "object" || terminalState === null || Array.isArray(terminalState)) {
      throw new Error("terminal state must be an object");
    }
    if (typeof terminalState.A !== "boolean" || typeof terminalState.B !== "boolean") {
      throw new Error("terminal state must include boolean A and B values");
    }
    if (typeof terminalState.errorA !== "boolean" || typeof terminalState.errorB !== "boolean") {
      throw new Error("terminal state must include boolean errorA and errorB values");
    }
    if (typeof terminalState.responseA !== "string" || typeof terminalState.responseB !== "string") {
      throw new Error("terminal state must include string responseA and responseB values");
    }
  }

  function markSideAsErrored(terminalState, side) {
    requireTerminalState(terminalState);
    if (side === "A") {
      terminalState.errorA = true;
      return;
    }
    if (side === "B") {
      terminalState.errorB = true;
      return;
    }
    throw new Error(`unsupported battleground side: ${side}`);
  }

  function listErroredSides(terminalState) {
    requireTerminalState(terminalState);
    const erroredSides = [];
    if (terminalState.errorA) {
      erroredSides.push("A");
    }
    if (terminalState.errorB) {
      erroredSides.push("B");
    }
    return erroredSides;
  }

  function appendStreamResponse(terminalState, side, cleanedChunk) {
    requireTerminalState(terminalState);
    if (typeof cleanedChunk !== "string") {
      throw new Error("cleaned stream chunk must be a string");
    }
    if (cleanedChunk === "") {
      return;
    }
    if (side === "A") {
      terminalState.responseA = `${terminalState.responseA}${cleanedChunk}`;
      return;
    }
    if (side === "B") {
      terminalState.responseB = `${terminalState.responseB}${cleanedChunk}`;
      return;
    }
    throw new Error(`unsupported battleground side: ${side}`);
  }

  function assertTerminalStateComplete(terminalState) {
    requireTerminalState(terminalState);
    const incompleteSides = [];
    if (!terminalState.A) {
      incompleteSides.push("A");
    }
    if (!terminalState.B) {
      incompleteSides.push("B");
    }
    if (incompleteSides.length > 0) {
      throw new Error(`Battleground stream ended before terminal events for side(s): ${incompleteSides.join(", ")}.`);
    }
  }

  function finalizeTurnHistories(terminalState, erroredSides) {
    requireTerminalState(terminalState);
    removeThinkingMessages();
    if (erroredSides.includes("A")) {
      logger.info("battleground_side_history_skipped side=A reason=error");
    } else {
      state.historyA.push({
        role: "assistant",
        message: requireAssistantResponse(terminalState.responseA, "A"),
      });
    }
    if (erroredSides.includes("B")) {
      logger.info("battleground_side_history_skipped side=B reason=error");
    } else {
      state.historyB.push({
        role: "assistant",
        message: requireAssistantResponse(terminalState.responseB, "B"),
      });
    }
    persistBattlegroundState();
    renderOutputs();
  }

  function requireAssistantResponse(responseText, side) {
    if (typeof responseText !== "string") {
      throw new Error(`assistant response for side ${side} must be a string`);
    }
    const assistantMessage = responseText.trim();
    if (assistantMessage === "") {
      throw new Error(`assistant response for side ${side} must not be empty`);
    }
    return assistantMessage;
  }

  function buildCompletionStatus(erroredSides) {
    if (!Array.isArray(erroredSides)) {
      throw new Error("errored sides must be an array");
    }
    if (erroredSides.length === 0) {
      return "Turn complete.";
    }
    return `Turn complete with side errors on: ${erroredSides.join(", ")}.`;
  }

  function renderOutputs() {
    modelAOutput.textContent = renderSideLines("A").join("\n");
    modelBOutput.textContent = renderSideLines("B").join("\n");
  }

  function renderSideLines(side) {
    const messages = getMessagesForSide(side);
    return messages.map((entry) => {
      if (entry.role === "user") {
        return `You: ${entry.text}`;
      }
      if (entry.role === "assistant") {
        return `Model ${side}: ${entry.text}`;
      }
      if (entry.role === "thinking") {
        return `Model ${side}: Thinking...`;
      }
      if (entry.role === "error") {
        return `Error: ${entry.text}`;
      }
      throw new Error(`unsupported battleground message role: ${entry.role}`);
    });
  }

  function persistBattlegroundState() {
    const payload = {
      selectedModelA: state.selectedModelA,
      selectedModelB: state.selectedModelB,
      isModelSelectionLocked: state.isModelSelectionLocked,
      historyA: cloneHistory(state.historyA),
      historyB: cloneHistory(state.historyB),
      messagesA: clonePersistableMessages(state.messagesA),
      messagesB: clonePersistableMessages(state.messagesB),
    };
    localStorage.setItem(BATTLEGROUND_STORAGE_KEY, JSON.stringify(payload));
  }

  function clonePersistableMessages(messages) {
    return messages
      .filter((entry) => entry.role !== "thinking")
      .map((entry) => ({ role: entry.role, text: entry.text }));
  }

  function getLastEntry(messages) {
    if (!Array.isArray(messages)) {
      throw new Error("message collection must be an array");
    }
    if (messages.length === 0) {
      return null;
    }
    return messages[messages.length - 1];
  }

  function handleClearBattlegroundChat() {
    logger.info("battleground_chat_reset_started");
    state.selectedModelA = "";
    state.selectedModelB = "";
    state.isModelSelectionLocked = false;
    state.historyA = [];
    state.historyB = [];
    state.messagesA = [];
    state.messagesB = [];
    state.isSubmitting = false;
    modelASelect.value = "";
    modelBSelect.value = "";
    applyModelLockState();
    persistBattlegroundState();
    renderOutputs();
    setBattlegroundStatus("Started a new battleground chat.");
    logger.info("battleground_chat_reset_completed");
  }

  function setBattlegroundStatus(message) {
    if (typeof message !== "string") {
      throw new Error("battleground status message must be a string");
    }
    battlegroundStatus.textContent = message;
  }

  function ensureSelectHasPlaceholder(selectElement) {
    if (selectElement.options.length === 0) {
      throw new Error(`select '${selectElement.id}' must include at least one option`);
    }
    const firstOption = selectElement.options[0];
    if (firstOption.value !== "") {
      throw new Error(`select '${selectElement.id}' placeholder value must be empty`);
    }
    if (firstOption.text.trim().length === 0) {
      throw new Error(`select '${selectElement.id}' placeholder label must not be empty`);
    }
  }
})();
