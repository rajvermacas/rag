import json
import shutil
import subprocess

_BATTLEGROUND_STREAM_HARNESS_TEMPLATE = """
(async () => {
const commonScript = __COMMON_SCRIPT__;
const battlegroundScript = __BATTLEGROUND_SCRIPT__;
const windowListeners = new Map();
const fetchCalls = [];
const readSnapshots = [];

function createOptionElement() {
  let textValue = "";
  return {
    tagName: "option",
    value: "",
    selected: false,
    get text() { return textValue; },
    set text(value) { textValue = String(value); },
    get textContent() { return textValue; },
    set textContent(value) { textValue = String(value); },
  };
}

function createElement(id, tagName = "div") {
  const listeners = new Map();
  const attributes = new Map();
  const classState = new Set();
  let classNameValue = "";
  let innerHtmlValue = "";
  const element = {
    id,
    tagName,
    value: "",
    disabled: false,
    textContent: "",
    children: [],
    scrollTop: 0,
    scrollHeight: 0,
    options: [],
    classList: {
      add: (...classes) => {
        classes.forEach((name) => classState.add(name));
        classNameValue = Array.from(classState).join(" ");
      },
      remove: (...classes) => {
        classes.forEach((name) => classState.delete(name));
        classNameValue = Array.from(classState).join(" ");
      },
      contains: (name) => classState.has(name),
    },
    append: function(child) {
      if (this.tagName === "select") {
        if (!child || child.tagName !== "option") throw new Error("select children must be option elements");
        this.options.push(child);
        if (child.selected) this.value = child.value;
        return;
      }
      this.children.push(child);
      this.scrollHeight = this.children.length;
    },
    querySelector: () => createElement("", "button"),
    addEventListener: function(eventName, handler) {
      if (!listeners.has(eventName)) listeners.set(eventName, []);
      listeners.get(eventName).push(handler);
    },
    setAttribute: function(name, value) {
      attributes.set(name, String(value));
    },
    getAttribute: function(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    click: function() {
      const handlers = listeners.get("click") || [];
      handlers.forEach((handler) => handler({ preventDefault: () => undefined }));
    },
    submit: async function() {
      const handlers = listeners.get("submit") || [];
      for (const handler of handlers) {
        await handler({ preventDefault: () => undefined });
      }
    },
    reset: () => undefined,
  };
  Object.defineProperty(element, "className", {
    get: () => classNameValue,
    set: (value) => {
      const classNames = String(value).split(/\\s+/).filter((name) => name !== "");
      classState.clear();
      classNames.forEach((name) => classState.add(name));
      classNameValue = classNames.join(" ");
    },
  });
  Object.defineProperty(element, "innerHTML", {
    get: () => innerHtmlValue,
    set: (value) => {
      innerHtmlValue = String(value);
      if (element.tagName === "select" && innerHtmlValue === "") {
        element.options = [];
        element.value = "";
      }
      if (element.tagName !== "select" && innerHtmlValue === "") {
        element.children = [];
      }
    },
  });
  return element;
}

function createSelectWithPlaceholder(id, placeholderLabel) {
  const select = createElement(id, "select");
  const placeholder = createOptionElement();
  placeholder.value = "";
  placeholder.text = placeholderLabel;
  placeholder.selected = true;
  select.append(placeholder);
  select.value = "";
  return select;
}

const elements = {
  "nav-chat": createElement("nav-chat", "button"),
  "nav-battleground": createElement("nav-battleground", "button"),
  "chat-section": createElement("chat-section", "section"),
  "battleground-section": createElement("battleground-section", "section"),
  "battleground-form": createElement("battleground-form", "form"),
  "battleground-submit": createElement("battleground-submit", "button"),
  "model-a-select": createSelectWithPlaceholder("model-a-select", "Select model A"),
  "model-b-select": createSelectWithPlaceholder("model-b-select", "Select model B"),
  "battleground-message": createElement("battleground-message", "textarea"),
  "battleground-status": createElement("battleground-status", "p"),
  "battleground-model-a-title": createElement("battleground-model-a-title", "h4"),
  "battleground-model-b-title": createElement("battleground-model-b-title", "h4"),
  "battleground-model-a-output": createElement("battleground-model-a-output", "div"),
  "battleground-model-b-output": createElement("battleground-model-b-output", "div"),
};
elements["chat-section"].className = "";
elements["battleground-section"].className = "hidden";
elements["nav-chat"].setAttribute("aria-selected", "true");
elements["nav-battleground"].setAttribute("aria-selected", "false");

const encoder = new TextEncoder();
const streamChunks = [
  encoder.encode("{\\"side\\":\\"A\\",\\"chunk\\":\\"A says **hi**\\"}\\n"),
  encoder.encode("{\\"side\\":\\"B\\",\\"chunk\\":\\"B says *hi*\\"}\\n{\\"side\\":\\"A\\",\\"done\\":true}\\n"),
  encoder.encode("{\\"side\\":\\"B\\",\\"error\\":\\"B failed\\"}\\n"),
];

globalThis.window = globalThis;
globalThis.console = { info: () => undefined, error: () => undefined, log: () => undefined };
globalThis.document = {
  getElementById: (id) => (id in elements ? elements[id] : null),
  createElement: (tag) => {
    if (tag === "option") return createOptionElement();
    return createElement("", tag);
  },
};
globalThis.marked = {
  setOptions: () => undefined,
  parse: (value) => String(value).replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>").replace(/\\*(.*?)\\*/g, "<em>$1</em>").replace(/\\n/g, "<br/>"),
};
globalThis.DOMPurify = { sanitize: (value) => value };
globalThis.addEventListener = (eventName, handler) => {
  if (!windowListeners.has(eventName)) windowListeners.set(eventName, []);
  windowListeners.get(eventName).push(handler);
};
globalThis.fetch = async (url, options = {}) => {
  const method = typeof options.method === "string" ? options.method : "GET";
  const normalizedMethod = method.toUpperCase();
  fetchCalls.push({
    url,
    method: normalizedMethod,
    body: "body" in options ? JSON.parse(options.body) : null,
  });
  if (url === "/models/battleground") {
    return {
      ok: true,
      json: async () => ({ models: [{ backend_id: "lab_vllm", provider: "openai_compatible", model: "openai/gpt-4o-mini", label: "lab_vllm (openai_compatible) · openai/gpt-4o-mini" }, { backend_id: "lab_vllm", provider: "openai_compatible", model: "anthropic/claude-3.5-sonnet", label: "lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet" }] }),
    };
  }
  if (url === "/battleground/compare/stream") {
    let chunkIndex = 0;
    return {
      ok: true,
      body: {
        getReader: () => ({
          read: async () => {
            readSnapshots.push({
              modelA: elements["battleground-model-a-output"].textContent,
              modelB: elements["battleground-model-b-output"].textContent,
              modelAHtml: elements["battleground-model-a-output"].innerHTML,
              modelBHtml: elements["battleground-model-b-output"].innerHTML,
              status: elements["battleground-status"].textContent,
            });
            if (chunkIndex >= streamChunks.length) {
              return { value: undefined, done: true };
            }
            const value = streamChunks[chunkIndex];
            chunkIndex += 1;
            return { value, done: false };
          },
        }),
      },
      json: async () => {
        throw new Error("stream response should not call json()");
      },
    };
  }
  throw new Error(`unexpected fetch url: ${url}`);
};

eval(commonScript);
eval(battlegroundScript);
for (const handler of windowListeners.get("load") || []) {
  await handler();
}
for (let i = 0; i < 10 && elements["model-a-select"].options.length < 3; i += 1) {
  await Promise.resolve();
}

elements["nav-battleground"].click();
const afterBattlegroundTab = {
  chatHidden: elements["chat-section"].classList.contains("hidden"),
  battlegroundHidden: elements["battleground-section"].classList.contains("hidden"),
  chatSelected: elements["nav-chat"].getAttribute("aria-selected"),
  battlegroundSelected: elements["nav-battleground"].getAttribute("aria-selected"),
};
elements["nav-chat"].click();
const afterChatTab = {
  chatHidden: elements["chat-section"].classList.contains("hidden"),
  battlegroundHidden: elements["battleground-section"].classList.contains("hidden"),
  chatSelected: elements["nav-chat"].getAttribute("aria-selected"),
  battlegroundSelected: elements["nav-battleground"].getAttribute("aria-selected"),
};

elements["model-a-select"].value = "lab_vllm||openai/gpt-4o-mini";
elements["model-b-select"].value = "lab_vllm||anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "Which answer is better?";
await elements["battleground-form"].submit();
const firstRequestBody = fetchCalls[1].body;
elements["battleground-message"].value = "Can you follow up with examples?";
await elements["battleground-form"].submit();
const secondRequestBody = fetchCalls[2].body;

process.stdout.write(JSON.stringify({
  modelAOptions: elements["model-a-select"].options.map((option) => option.value),
  modelBOptions: elements["model-b-select"].options.map((option) => option.value),
  fetchCalls,
  firstRequestBody,
  secondRequestBody,
  readSnapshots,
  modelATitle: elements["battleground-model-a-title"].textContent,
  modelBTitle: elements["battleground-model-b-title"].textContent,
  modelAOutput: elements["battleground-model-a-output"].textContent,
  modelBOutput: elements["battleground-model-b-output"].textContent,
  modelAHtml: elements["battleground-model-a-output"].innerHTML,
  modelBHtml: elements["battleground-model-b-output"].innerHTML,
  finalStatus: elements["battleground-status"].textContent,
  afterBattlegroundTab,
  afterChatTab,
}));
})().catch((error) => {
  process.stderr.write(String(error));
  process.exit(1);
});
"""

_BATTLEGROUND_VALIDATION_HARNESS_TEMPLATE = """
(async () => {
const commonScript = __COMMON_SCRIPT__;
const battlegroundScript = __BATTLEGROUND_SCRIPT__;
const fetchCalls = [];

function createOptionElement() {
  let textValue = "";
  return {
    tagName: "option",
    value: "",
    selected: false,
    get text() { return textValue; },
    set text(value) { textValue = String(value); },
    get textContent() { return textValue; },
    set textContent(value) { textValue = String(value); },
  };
}

function createElement(id, tagName = "div") {
  const listeners = new Map();
  const attributes = new Map();
  let classNameValue = "";
  let innerHtmlValue = "";
  const element = {
    id,
    tagName,
    value: "",
    textContent: "",
    options: [],
    classList: {
      add: (...names) => { classNameValue = `${classNameValue} ${names.join(" ")}`.trim(); },
      remove: () => undefined,
      contains: (name) => classNameValue.split(/\\s+/).includes(name),
    },
    append: function(child) {
      if (this.tagName === "select") {
        this.options.push(child);
        if (child.selected) this.value = child.value;
      }
    },
    addEventListener: function(eventName, handler) {
      if (!listeners.has(eventName)) listeners.set(eventName, []);
      listeners.get(eventName).push(handler);
    },
    setAttribute: function(name, value) { attributes.set(name, String(value)); },
    getAttribute: function(name) { return attributes.has(name) ? attributes.get(name) : null; },
    click: function() {
      const handlers = listeners.get("click") || [];
      handlers.forEach((handler) => handler({ preventDefault: () => undefined }));
    },
    submit: async function() {
      const handlers = listeners.get("submit") || [];
      for (const handler of handlers) {
        await handler({ preventDefault: () => undefined });
      }
    },
    reset: () => undefined,
  };
  Object.defineProperty(element, "className", {
    get: () => classNameValue,
    set: (value) => { classNameValue = String(value); },
  });
  Object.defineProperty(element, "innerHTML", {
    get: () => innerHtmlValue,
    set: (value) => {
      innerHtmlValue = String(value);
      if (element.tagName === "select" && innerHtmlValue === "") {
        element.options = [];
        element.value = "";
      }
    },
  });
  return element;
}

function createSelectWithPlaceholder(id, placeholderLabel) {
  const select = createElement(id, "select");
  const option = createOptionElement();
  option.value = "";
  option.text = placeholderLabel;
  option.selected = true;
  select.append(option);
  select.value = "";
  return select;
}

const elements = {
  "nav-chat": createElement("nav-chat", "button"),
  "nav-battleground": createElement("nav-battleground", "button"),
  "chat-section": createElement("chat-section", "section"),
  "battleground-section": createElement("battleground-section", "section"),
  "battleground-form": createElement("battleground-form", "form"),
  "battleground-submit": createElement("battleground-submit", "button"),
  "model-a-select": createSelectWithPlaceholder("model-a-select", "Select model A"),
  "model-b-select": createSelectWithPlaceholder("model-b-select", "Select model B"),
  "battleground-message": createElement("battleground-message", "textarea"),
  "battleground-status": createElement("battleground-status", "p"),
  "battleground-model-a-title": createElement("battleground-model-a-title", "h4"),
  "battleground-model-b-title": createElement("battleground-model-b-title", "h4"),
  "battleground-model-a-output": createElement("battleground-model-a-output", "div"),
  "battleground-model-b-output": createElement("battleground-model-b-output", "div"),
};
elements["chat-section"].className = "";
elements["battleground-section"].className = "hidden";
elements["nav-chat"].setAttribute("aria-selected", "true");
elements["nav-battleground"].setAttribute("aria-selected", "false");

globalThis.window = globalThis;
globalThis.console = { info: () => undefined, error: () => undefined, log: () => undefined };
globalThis.document = {
  getElementById: (id) => (id in elements ? elements[id] : null),
  createElement: (tag) => {
    if (tag === "option") return createOptionElement();
    return createElement("", tag);
  },
};
globalThis.marked = { setOptions: () => undefined, parse: (value) => value };
globalThis.DOMPurify = { sanitize: (value) => value };
globalThis.addEventListener = () => undefined;
globalThis.fetch = async (url, options = {}) => {
  const method = typeof options.method === "string" ? options.method : "GET";
  fetchCalls.push({ url, method: method.toUpperCase() });
  if (url === "/models/battleground") {
    return {
      ok: true,
      json: async () => ({ models: [{ backend_id: "lab_vllm", provider: "openai_compatible", model: "openai/gpt-4o-mini", label: "lab_vllm (openai_compatible) · openai/gpt-4o-mini" }, { backend_id: "lab_vllm", provider: "openai_compatible", model: "anthropic/claude-3.5-sonnet", label: "lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet" }] }),
    };
  }
  if (url === "/battleground/compare/stream") {
    throw new Error("validation should prevent stream request");
  }
  throw new Error(`unexpected fetch url: ${url}`);
};

eval(commonScript);
eval(battlegroundScript);
for (let i = 0; i < 10 && elements["model-a-select"].options.length < 3; i += 1) {
  await Promise.resolve();
}

const statuses = [];
elements["model-a-select"].value = "lab_vllm||openai/gpt-4o-mini";
elements["model-b-select"].value = "lab_vllm||anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "   ";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

elements["model-a-select"].value = "";
elements["model-b-select"].value = "lab_vllm||anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "valid question";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

elements["model-a-select"].value = "lab_vllm||openai/gpt-4o-mini";
elements["model-b-select"].value = "";
elements["battleground-message"].value = "valid question";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

elements["model-a-select"].value = "lab_vllm||openai/gpt-4o-mini";
elements["model-b-select"].value = "lab_vllm||openai/gpt-4o-mini";
elements["battleground-message"].value = "valid question";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

process.stdout.write(JSON.stringify({
  statuses,
  modelAOptions: elements["model-a-select"].options.map((option) => option.value),
  modelBOptions: elements["model-b-select"].options.map((option) => option.value),
  postCallCount: fetchCalls.filter((call) => call.url === "/battleground/compare/stream").length,
  fetchCalls,
}));
})().catch((error) => {
  process.stderr.write(String(error));
  process.exit(1);
});
"""

_BATTLEGROUND_BOOTSTRAP_FAILURE_HARNESS_TEMPLATE = """
(async () => {
const commonScript = __COMMON_SCRIPT__;
const battlegroundScript = __BATTLEGROUND_SCRIPT__;
let unhandledRejection = null;
const errorLogs = [];

process.on("unhandledRejection", (error) => {
  unhandledRejection = error instanceof Error ? error.message : String(error);
});

function createOptionElement() {
  let textValue = "";
  return {
    tagName: "option",
    value: "",
    selected: false,
    get text() { return textValue; },
    set text(value) { textValue = String(value); },
    get textContent() { return textValue; },
    set textContent(value) { textValue = String(value); },
  };
}

function createElement(id, tagName = "div") {
  const listeners = new Map();
  const attributes = new Map();
  let classNameValue = "";
  let innerHtmlValue = "";
  const element = {
    id,
    tagName,
    value: "",
    textContent: "",
    options: [],
    classList: {
      add: (...names) => { classNameValue = `${classNameValue} ${names.join(" ")}`.trim(); },
      remove: () => undefined,
      contains: (name) => classNameValue.split(/\\s+/).includes(name),
    },
    append: function(child) {
      if (this.tagName === "select") {
        this.options.push(child);
        if (child.selected) this.value = child.value;
      }
    },
    addEventListener: function(eventName, handler) {
      if (!listeners.has(eventName)) listeners.set(eventName, []);
      listeners.get(eventName).push(handler);
    },
    setAttribute: function(name, value) { attributes.set(name, String(value)); },
    getAttribute: function(name) { return attributes.has(name) ? attributes.get(name) : null; },
  };
  Object.defineProperty(element, "className", {
    get: () => classNameValue,
    set: (value) => { classNameValue = String(value); },
  });
  Object.defineProperty(element, "innerHTML", {
    get: () => innerHtmlValue,
    set: (value) => {
      innerHtmlValue = String(value);
      if (element.tagName === "select" && innerHtmlValue === "") {
        element.options = [];
        element.value = "";
      }
    },
  });
  return element;
}

function createSelectWithPlaceholder(id, placeholderLabel) {
  const select = createElement(id, "select");
  const option = createOptionElement();
  option.value = "";
  option.text = placeholderLabel;
  option.selected = true;
  select.append(option);
  select.value = "";
  return select;
}

const elements = {
  "nav-chat": createElement("nav-chat", "button"),
  "nav-battleground": createElement("nav-battleground", "button"),
  "chat-section": createElement("chat-section", "section"),
  "battleground-section": createElement("battleground-section", "section"),
  "battleground-form": createElement("battleground-form", "form"),
  "battleground-submit": createElement("battleground-submit", "button"),
  "model-a-select": createSelectWithPlaceholder("model-a-select", "Select model A"),
  "model-b-select": createSelectWithPlaceholder("model-b-select", "Select model B"),
  "battleground-message": createElement("battleground-message", "textarea"),
  "battleground-status": createElement("battleground-status", "p"),
  "battleground-model-a-title": createElement("battleground-model-a-title", "h4"),
  "battleground-model-b-title": createElement("battleground-model-b-title", "h4"),
  "battleground-model-a-output": createElement("battleground-model-a-output", "div"),
  "battleground-model-b-output": createElement("battleground-model-b-output", "div"),
};

globalThis.window = globalThis;
globalThis.console = {
  info: () => undefined,
  log: () => undefined,
  error: (...args) => {
    errorLogs.push(args.map((value) => String(value)).join(" "));
  },
};
globalThis.document = {
  getElementById: (id) => (id in elements ? elements[id] : null),
  createElement: (tag) => {
    if (tag === "option") return createOptionElement();
    return createElement("", tag);
  },
};
globalThis.marked = { setOptions: () => undefined, parse: (value) => value };
globalThis.DOMPurify = { sanitize: (value) => value };
globalThis.addEventListener = () => undefined;
globalThis.fetch = async (url) => {
  if (url === "/models/battleground") {
    return {
      ok: false,
      json: async () => ({ detail: "Battleground model list failed." }),
    };
  }
  throw new Error(`unexpected fetch url: ${url}`);
};

eval(commonScript);
eval(battlegroundScript);
for (let i = 0; i < 10; i += 1) {
  await Promise.resolve();
}

process.stdout.write(JSON.stringify({
  status: elements["battleground-status"].textContent,
  errorLogs,
  unhandledRejection,
}));
})().catch((error) => {
  process.stderr.write(String(error));
  process.exit(1);
});
"""

_BATTLEGROUND_TRUNCATED_STREAM_HARNESS_TEMPLATE = """
(async () => {
const commonScript = __COMMON_SCRIPT__;
const battlegroundScript = __BATTLEGROUND_SCRIPT__;
const encoder = new TextEncoder();
const streamChunks = [
  encoder.encode("{\\"side\\":\\"A\\",\\"chunk\\":\\"A partial\\"}\\n"),
  encoder.encode("{\\"side\\":\\"B\\",\\"chunk\\":\\"B partial\\"}\\n"),
];

function createOptionElement() {
  let textValue = "";
  return {
    tagName: "option",
    value: "",
    selected: false,
    get text() { return textValue; },
    set text(value) { textValue = String(value); },
    get textContent() { return textValue; },
    set textContent(value) { textValue = String(value); },
  };
}

function createElement(id, tagName = "div") {
  const listeners = new Map();
  const attributes = new Map();
  let classNameValue = "";
  let innerHtmlValue = "";
  const element = {
    id,
    tagName,
    value: "",
    textContent: "",
    options: [],
    classList: {
      add: (...names) => { classNameValue = `${classNameValue} ${names.join(" ")}`.trim(); },
      remove: () => undefined,
      contains: (name) => classNameValue.split(/\\s+/).includes(name),
    },
    append: function(child) {
      if (this.tagName === "select") {
        this.options.push(child);
        if (child.selected) this.value = child.value;
      }
    },
    addEventListener: function(eventName, handler) {
      if (!listeners.has(eventName)) listeners.set(eventName, []);
      listeners.get(eventName).push(handler);
    },
    setAttribute: function(name, value) { attributes.set(name, String(value)); },
    getAttribute: function(name) { return attributes.has(name) ? attributes.get(name) : null; },
    submit: async function() {
      const handlers = listeners.get("submit") || [];
      for (const handler of handlers) {
        await handler({ preventDefault: () => undefined });
      }
    },
  };
  Object.defineProperty(element, "className", {
    get: () => classNameValue,
    set: (value) => { classNameValue = String(value); },
  });
  Object.defineProperty(element, "innerHTML", {
    get: () => innerHtmlValue,
    set: (value) => {
      innerHtmlValue = String(value);
      if (element.tagName === "select" && innerHtmlValue === "") {
        element.options = [];
        element.value = "";
      }
    },
  });
  return element;
}

function createSelectWithPlaceholder(id, placeholderLabel) {
  const select = createElement(id, "select");
  const option = createOptionElement();
  option.value = "";
  option.text = placeholderLabel;
  option.selected = true;
  select.append(option);
  select.value = "";
  return select;
}

const elements = {
  "nav-chat": createElement("nav-chat", "button"),
  "nav-battleground": createElement("nav-battleground", "button"),
  "chat-section": createElement("chat-section", "section"),
  "battleground-section": createElement("battleground-section", "section"),
  "battleground-form": createElement("battleground-form", "form"),
  "battleground-submit": createElement("battleground-submit", "button"),
  "model-a-select": createSelectWithPlaceholder("model-a-select", "Select model A"),
  "model-b-select": createSelectWithPlaceholder("model-b-select", "Select model B"),
  "battleground-message": createElement("battleground-message", "textarea"),
  "battleground-status": createElement("battleground-status", "p"),
  "battleground-model-a-title": createElement("battleground-model-a-title", "h4"),
  "battleground-model-b-title": createElement("battleground-model-b-title", "h4"),
  "battleground-model-a-output": createElement("battleground-model-a-output", "div"),
  "battleground-model-b-output": createElement("battleground-model-b-output", "div"),
};

globalThis.window = globalThis;
globalThis.console = { info: () => undefined, error: () => undefined, log: () => undefined };
globalThis.document = {
  getElementById: (id) => (id in elements ? elements[id] : null),
  createElement: (tag) => {
    if (tag === "option") return createOptionElement();
    return createElement("", tag);
  },
};
globalThis.marked = { setOptions: () => undefined, parse: (value) => value };
globalThis.DOMPurify = { sanitize: (value) => value };
globalThis.addEventListener = () => undefined;
globalThis.fetch = async (url, options = {}) => {
  if (url === "/models/battleground") {
    return {
      ok: true,
      json: async () => ({ models: [{ backend_id: "lab_vllm", provider: "openai_compatible", model: "openai/gpt-4o-mini", label: "lab_vllm (openai_compatible) · openai/gpt-4o-mini" }, { backend_id: "lab_vllm", provider: "openai_compatible", model: "anthropic/claude-3.5-sonnet", label: "lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet" }] }),
    };
  }
  if (url === "/battleground/compare/stream") {
    if (typeof options.body !== "string") {
      throw new Error("battleground stream body must be provided");
    }
    let index = 0;
    return {
      ok: true,
      body: {
        getReader: () => ({
          read: async () => {
            if (index >= streamChunks.length) return { value: undefined, done: true };
            const value = streamChunks[index];
            index += 1;
            return { value, done: false };
          },
        }),
      },
      json: async () => {
        throw new Error("stream response should not call json()");
      },
    };
  }
  throw new Error(`unexpected fetch url: ${url}`);
};

eval(commonScript);
eval(battlegroundScript);
for (let i = 0; i < 10 && elements["model-a-select"].options.length < 3; i += 1) {
  await Promise.resolve();
}

elements["model-a-select"].value = "lab_vllm||openai/gpt-4o-mini";
elements["model-b-select"].value = "lab_vllm||anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "Which answer is better?";
await elements["battleground-form"].submit();

process.stdout.write(JSON.stringify({
  finalStatus: elements["battleground-status"].textContent,
  modelAOutput: elements["battleground-model-a-output"].textContent,
  modelBOutput: elements["battleground-model-b-output"].textContent,
  modelAHtml: elements["battleground-model-a-output"].innerHTML,
  modelBHtml: elements["battleground-model-b-output"].innerHTML,
}));
})().catch((error) => {
  process.stderr.write(String(error));
  process.exit(1);
});
"""


def _run_node_harness(harness: str, context: str) -> str:
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError(f"node executable is required for {context}")

    completed = subprocess.run(
        [node_path, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        error_output = completed.stderr.strip()
        raise RuntimeError(f"node harness failed for {context}: {error_output}")
    return completed.stdout


def _run_battleground_harness(
    common_script: str,
    battleground_script: str,
    harness_template: str,
    context: str,
) -> dict[str, object]:
    harness = harness_template
    harness = harness.replace("__COMMON_SCRIPT__", json.dumps(common_script))
    harness = harness.replace("__BATTLEGROUND_SCRIPT__", json.dumps(battleground_script))
    payload = json.loads(_run_node_harness(harness, context))
    if not isinstance(payload, dict):
        raise RuntimeError(f"node harness output for {context} must be an object")
    return payload
