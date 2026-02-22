(() => {
  "use strict";

  if (typeof window.RagCommon === "undefined") {
    throw new Error("RagCommon is required before battleground.js");
  }

  const {
    requireElement,
    requireString,
    requireErrorMessage,
    renderMarkdown,
    removeCitationArtifacts,
  } = window.RagCommon;
  const logger = console;

  const navChat = requireElement("nav-chat");
  const navBattleground = requireElement("nav-battleground");
  const chatSection = requireElement("chat-section");
  const battlegroundSection = requireElement("battleground-section");
  const battlegroundForm = requireElement("battleground-form");
  const battlegroundMessage = requireElement("battleground-message");
  const battlegroundSubmit = requireElement("battleground-submit");
  const modelASelect = requireElement("model-a-select");
  const modelBSelect = requireElement("model-b-select");
  const battlegroundStatus = requireElement("battleground-status");
  const modelATitle = requireElement("battleground-model-a-title");
  const modelBTitle = requireElement("battleground-model-b-title");
  const modelAOutput = requireElement("battleground-model-a-output");
  const modelBOutput = requireElement("battleground-model-b-output");

  const battlegroundHistory = [];

  initializeTabNavigation();
  void initializeBattleground().catch(handleInitializationFailure);

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
      navChat.className =
        "rounded-xl border border-zinc-900 bg-zinc-900 px-4 py-2 text-sm font-semibold text-white transition";
      navBattleground.setAttribute("aria-selected", "false");
      navBattleground.className =
        "rounded-xl border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:border-red-500 hover:text-red-600";
      chatSection.classList.remove("hidden");
      battlegroundSection.classList.add("hidden");
      return;
    }
    if (tabId === "battleground") {
      navChat.setAttribute("aria-selected", "false");
      navChat.className =
        "rounded-xl border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 transition hover:border-red-500 hover:text-red-600";
      navBattleground.setAttribute("aria-selected", "true");
      navBattleground.className =
        "rounded-xl border border-zinc-900 bg-zinc-900 px-4 py-2 text-sm font-semibold text-white transition";
      chatSection.classList.add("hidden");
      battlegroundSection.classList.remove("hidden");
      return;
    }
    throw new Error(`unsupported tab id: ${tabId}`);
  }

  async function initializeBattleground() {
    ensureSelectHasPlaceholder(modelASelect);
    ensureSelectHasPlaceholder(modelBSelect);
    resetModelTitles();
    clearOutputs();
    battlegroundForm.addEventListener("submit", handleCompareSubmit);
    await loadModelOptions();
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
    setBattlegroundStatus("Models loaded. Ask a question to start comparison.");
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

  async function handleCompareSubmit(event) {
    event.preventDefault();
    battlegroundSubmit.disabled = true;
    try {
      const message = readRequiredMessage();
      const modelA = readRequiredModel(modelASelect, "Choose a model for Model A.");
      const modelB = readRequiredModel(modelBSelect, "Choose a model for Model B.");
      if (modelA === modelB) {
        throw new Error("Model A and Model B must be different.");
      }
      setModelTitles(modelA, modelB);
      const historyPayload = buildRequestHistoryPayload();
      logger.info(
        "battleground_compare_request_started model_a=%s model_b=%s history_turns=%s",
        modelA,
        modelB,
        historyPayload.length
      );
      setBattlegroundStatus("Comparing models...");
      const sideState = createSideState();
      renderThinkingState(sideState);
      const result = await streamComparison(
        {
          message,
          history: historyPayload,
          model_a: modelA,
          model_b: modelB,
        },
        sideState
      );
      appendConversationTurns(message, modelA, modelB, sideState);
      setBattlegroundStatus(buildCompletionStatus(result.erroredSides));
      battlegroundMessage.value = "";
      logger.info(
        "battleground_compare_request_completed history_turns=%s",
        historyPayload.length + 2
      );
    } catch (error) {
      const message = requireErrorMessage(error);
      setBattlegroundStatus(message);
      logger.error("battleground_compare_request_failed error=%s", message);
    } finally {
      battlegroundSubmit.disabled = false;
    }
  }

  function clearOutputs() {
    modelAOutput.innerHTML = "";
    modelBOutput.innerHTML = "";
  }

  function resetModelTitles() {
    modelATitle.textContent = "Model A";
    modelBTitle.textContent = "Model B";
  }

  function setModelTitles(modelA, modelB) {
    if (typeof modelA !== "string" || modelA.trim() === "") {
      throw new Error("modelA title value must be a non-empty string");
    }
    if (typeof modelB !== "string" || modelB.trim() === "") {
      throw new Error("modelB title value must be a non-empty string");
    }
    modelATitle.textContent = `Model A · ${modelA}`;
    modelBTitle.textContent = `Model B · ${modelB}`;
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

  function createSideState() {
    return {
      A: { markdownText: "", terminalText: "", thinking: true },
      B: { markdownText: "", terminalText: "", thinking: true },
    };
  }

  function requireSideState(sideState) {
    if (typeof sideState !== "object" || sideState === null || Array.isArray(sideState)) {
      throw new Error("side state must be an object");
    }
    if (!("A" in sideState) || !("B" in sideState)) {
      throw new Error("side state must include A and B records");
    }
  }

  function resolveSideState(sideState, side) {
    requireSideState(sideState);
    if (side === "A" || side === "B") {
      return sideState[side];
    }
    throw new Error(`unsupported battleground side: ${side}`);
  }

  function renderThinkingState(sideState) {
    clearOutputs();
    renderSide("A", sideState);
    renderSide("B", sideState);
  }

  function renderSide(side, sideState) {
    const state = resolveSideState(sideState, side);
    const outputElement = resolveSideOutput(side);
    const combinedText = buildCombinedSideText(state);
    const cleanedText = removeCitationArtifacts(combinedText);
    const markdownHtml = cleanedText.trim() === "" ? "" : renderMarkdown(cleanedText);
    const thinkingHtml = state.thinking
      ? "<p class=\"mt-2 text-sm text-zinc-500 animate-pulse\">Thinking...</p>"
      : "";
    outputElement.innerHTML = `<div class="markdown-body">${markdownHtml}</div>${thinkingHtml}`;
  }

  function buildCombinedSideText(state) {
    if (typeof state !== "object" || state === null || Array.isArray(state)) {
      throw new Error("side render state must be an object");
    }
    if (typeof state.markdownText !== "string" || typeof state.terminalText !== "string") {
      throw new Error("side render state must include markdownText and terminalText strings");
    }
    if (state.terminalText === "") {
      return state.markdownText;
    }
    if (state.markdownText === "") {
      return state.terminalText;
    }
    return `${state.markdownText}\n${state.terminalText}`;
  }

  async function streamComparison(payload, sideState) {
    requireSideState(sideState);
    const terminalState = createTerminalState();
    const response = await fetch("/battleground/compare/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
      buffered = consumeNdjsonBuffer(buffered, false, terminalState, sideState);
    }
    buffered += decoder.decode();
    consumeNdjsonBuffer(buffered, true, terminalState, sideState);
    assertTerminalStateComplete(terminalState);
    return { erroredSides: listErroredSides(terminalState) };
  }

  function consumeNdjsonBuffer(buffered, allowTrailingLine, terminalState, sideState) {
    if (typeof buffered !== "string") {
      throw new Error("ndjson buffer must be a string");
    }
    if (typeof allowTrailingLine !== "boolean") {
      throw new Error("allowTrailingLine must be a boolean");
    }
    requireTerminalState(terminalState);
    requireSideState(sideState);
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
      applyStreamEvent(line, terminalState, sideState);
    });
    return trailing;
  }

  function applyStreamEvent(rawLine, terminalState, sideState) {
    requireTerminalState(terminalState);
    requireSideState(sideState);
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
    const renderState = resolveSideState(sideState, side);
    const hasChunk = Object.prototype.hasOwnProperty.call(event, "chunk");
    const hasDone = Object.prototype.hasOwnProperty.call(event, "done");
    const hasError = Object.prototype.hasOwnProperty.call(event, "error");
    if ([hasChunk, hasDone, hasError].filter((value) => value).length !== 1) {
      throw new Error("battleground event must include exactly one of chunk, done, or error");
    }
    if (hasChunk) {
      renderState.markdownText += requireString(event, "chunk", "battleground chunk event");
      renderState.thinking = false;
      renderSide(side, sideState);
      return;
    }
    if (hasDone) {
      if (event.done !== true) {
        throw new Error("battleground done event must set done=true");
      }
      terminalState[side] = true;
      renderState.thinking = false;
      renderState.terminalText = appendTerminalText(renderState.terminalText, "Done.");
      renderSide(side, sideState);
      return;
    }
    terminalState[side] = true;
    markSideAsErrored(terminalState, side);
    renderState.thinking = false;
    renderState.terminalText = appendTerminalText(
      renderState.terminalText,
      `Error: ${requireString(event, "error", "battleground error event")}`
    );
    renderSide(side, sideState);
  }

  function appendTerminalText(existingText, newLine) {
    if (typeof existingText !== "string") {
      throw new Error("existing terminal text must be a string");
    }
    if (typeof newLine !== "string" || newLine.trim() === "") {
      throw new Error("terminal line must be a non-empty string");
    }
    if (existingText === "") {
      return newLine;
    }
    return `${existingText}\n${newLine}`;
  }

  function buildRequestHistoryPayload() {
    return battlegroundHistory.map((turn, index) => {
      const context = `battleground history turn[${index}]`;
      const role = requireString(turn, "role", context);
      const message = requireString(turn, "message", context);
      if (role !== "user" && role !== "assistant") {
        throw new Error(`${context} role must be user or assistant`);
      }
      return { role, message };
    });
  }

  function appendConversationTurns(message, modelA, modelB, sideState) {
    const assistantMessage = buildAssistantHistoryMessage(modelA, modelB, sideState);
    battlegroundHistory.push({ role: "user", message });
    battlegroundHistory.push({ role: "assistant", message: assistantMessage });
  }

  function buildAssistantHistoryMessage(modelA, modelB, sideState) {
    if (typeof modelA !== "string" || modelA.trim() === "") {
      throw new Error("assistant history modelA must be a non-empty string");
    }
    if (typeof modelB !== "string" || modelB.trim() === "") {
      throw new Error("assistant history modelB must be a non-empty string");
    }
    const sideA = buildHistorySideText(sideState, "A");
    const sideB = buildHistorySideText(sideState, "B");
    return `Model A (${modelA}):\n${sideA}\n\nModel B (${modelB}):\n${sideB}`;
  }

  function buildHistorySideText(sideState, side) {
    const state = resolveSideState(sideState, side);
    const combined = buildCombinedSideText(state);
    const normalized = removeCitationArtifacts(combined).trim();
    if (normalized === "") {
      throw new Error(`battleground side ${side} response text is empty`);
    }
    return normalized;
  }

  function createTerminalState() {
    return { A: false, B: false, errorA: false, errorB: false };
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

  function buildCompletionStatus(erroredSides) {
    if (!Array.isArray(erroredSides)) {
      throw new Error("errored sides must be an array");
    }
    if (erroredSides.length === 0) {
      return "Comparison complete.";
    }
    return `Comparison complete with side errors on: ${erroredSides.join(", ")}.`;
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
      throw new Error(
        `Battleground stream ended before terminal events for side(s): ${incompleteSides.join(", ")}.`
      );
    }
  }

  function resolveSideOutput(side) {
    if (side === "A") {
      return modelAOutput;
    }
    if (side === "B") {
      return modelBOutput;
    }
    throw new Error(`unsupported battleground side: ${side}`);
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
