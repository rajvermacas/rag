(() => {
  "use strict";

  if (typeof window.RagCommon === "undefined") {
    throw new Error("RagCommon is required before battleground.js");
  }

  const { requireElement } = window.RagCommon;

  const navChat = requireElement("nav-chat");
  const navBattleground = requireElement("nav-battleground");
  const chatSection = requireElement("chat-section");
  const battlegroundSection = requireElement("battleground-section");
  const battlegroundForm = requireElement("battleground-form");
  const modelASelect = requireElement("model-a-select");
  const modelBSelect = requireElement("model-b-select");
  const battlegroundStatus = requireElement("battleground-status");
  const modelAOutput = requireElement("battleground-model-a-output");
  const modelBOutput = requireElement("battleground-model-b-output");

  initializeTabNavigation();
  initializeBattlegroundScaffold();

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

  function initializeBattlegroundScaffold() {
    ensureSelectHasPlaceholder(modelASelect);
    ensureSelectHasPlaceholder(modelBSelect);
    battlegroundStatus.textContent = "Battleground scaffold ready. Model loading arrives in a later task.";
    modelAOutput.textContent = "Model A response will appear here.";
    modelBOutput.textContent = "Model B response will appear here.";

    battlegroundForm.addEventListener("submit", (event) => {
      event.preventDefault();
      battlegroundStatus.textContent = "Comparison scaffold submitted. Streaming integration arrives in a later task.";
    });
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
