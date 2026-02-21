import json
import shutil
import subprocess

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.chat import ChatResult
from app.services.ingest import IngestResult


class FakeIngestService:
    async def ingest_upload(self, upload) -> IngestResult:
        return IngestResult(doc_id="doc-123", chunks_indexed=3)


class FakeChatService:
    async def answer_question(self, question: str, history) -> ChatResult:
        return ChatResult(
            answer="ok",
            citations=[],
            grounded=False,
            retrieved_count=0,
        )

    async def stream_answer_question(self, question: str, history):
        yield "ok"


class FakeDocumentService:
    def list_documents(self):
        return [{"doc_id": "doc-123", "filename": "a.txt", "chunks_indexed": 3}]

    def delete_document(self, doc_id: str):
        raise AssertionError("Document service should not be called in index test")


_CHAT_SESSION_HARNESS_TEMPLATE = """
(async () => {
const commonScript = __COMMON_SCRIPT__;
const chatScript = __CHAT_SCRIPT__;
const defaultGreeting = __DEFAULT_GREETING__;
const localStorageRecords = new Map();
const windowListeners = new Map();

function createElement(id, tagName = "div") {
  const listeners = new Map();
  return {
    id,
    tagName,
    value: "",
    disabled: false,
    className: "",
    textContent: "",
    innerHTML: "",
    children: [],
    scrollTop: 0,
    scrollHeight: 0,
    classList: { add: () => undefined, remove: () => undefined },
    append: function(child) { this.children.push(child); this.scrollHeight = this.children.length; },
    querySelector: () => createElement("", "button"),
    addEventListener: function(eventName, handler) {
      if (!listeners.has(eventName)) listeners.set(eventName, []);
      listeners.get(eventName).push(handler);
    },
    click: function() {
      const handlers = listeners.get("click") || [];
      handlers.forEach((handler) => handler({ preventDefault: () => undefined }));
    },
    reset: () => undefined,
  };
}

const ids = [
  "upload-form","upload-button","upload-loader","upload-button-label","upload-status",
  "refresh-documents","documents-list","documents-status","chat-form","chat-button",
  "chat-window","chat-history-select","clear-chat"
];
const elements = Object.fromEntries(ids.map((id) => [id, createElement(id)]));
globalThis.window = globalThis;
globalThis.console = { info: () => undefined, error: () => undefined, log: () => undefined };
globalThis.document = {
  getElementById: (id) => (id in elements ? elements[id] : null),
  createElement: (tag) => createElement("", tag),
};
globalThis.localStorage = {
  getItem: (key) => (localStorageRecords.has(key) ? localStorageRecords.get(key) : null),
  setItem: (key, value) => localStorageRecords.set(key, value),
};
globalThis.crypto = { randomUUID: (() => { let n = 0; return () => `uuid-${++n}`; })() };
globalThis.fetch = async (url) => {
  if (url === "/documents") return { ok: true, json: async () => ({ documents: [] }) };
  throw new Error(`unexpected fetch url: ${url}`);
};
globalThis.confirm = () => true;
globalThis.marked = { setOptions: () => undefined, parse: (value) => value };
globalThis.DOMPurify = { sanitize: (value) => value };
globalThis.addEventListener = (eventName, handler) => {
  if (!windowListeners.has(eventName)) windowListeners.set(eventName, []);
  windowListeners.get(eventName).push(handler);
};
eval(commonScript);
eval(chatScript);
for (const handler of windowListeners.get("load") || []) {
  await handler();
}

const initialPayload = JSON.parse(localStorageRecords.get("rag-chat-sessions"));
const initialSession = initialPayload.sessions[0];
const initialMessage = initialSession.messages[0];
elements["clear-chat"].click();
const afterClickPayload = JSON.parse(localStorageRecords.get("rag-chat-sessions"));

process.stdout.write(JSON.stringify({
  initialSessionCount: initialPayload.sessions.length,
  initialHistoryLength: initialSession.history.length,
  initialGreetingText: initialMessage.text,
  isInitialGreetingAssistant: initialMessage.role === "assistant",
  greetingMatchesDefault: initialMessage.text === defaultGreeting,
  afterClickSessionCount: afterClickPayload.sessions.length
}));
})().catch((error) => {
  process.stderr.write(String(error));
  process.exit(1);
});
"""

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
  "battleground-model-a-output": createElement("battleground-model-a-output", "div"),
  "battleground-model-b-output": createElement("battleground-model-b-output", "div"),
};
elements["chat-section"].className = "";
elements["battleground-section"].className = "hidden";
elements["nav-chat"].setAttribute("aria-selected", "true");
elements["nav-battleground"].setAttribute("aria-selected", "false");

const encoder = new TextEncoder();
const streamChunks = [
  encoder.encode("{\\"side\\":\\"A\\",\\"chunk\\":\\"A says hi\\"}\\n"),
  encoder.encode("{\\"side\\":\\"B\\",\\"chunk\\":\\"B says hi\\"}\\n{\\"side\\":\\"A\\",\\"done\\":true}\\n"),
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
globalThis.marked = { setOptions: () => undefined, parse: (value) => value };
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
      json: async () => ({ models: ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"] }),
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

elements["model-a-select"].value = "openai/gpt-4o-mini";
elements["model-b-select"].value = "anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "Which answer is better?";
await elements["battleground-form"].submit();

process.stdout.write(JSON.stringify({
  modelAOptions: elements["model-a-select"].options.map((option) => option.value),
  modelBOptions: elements["model-b-select"].options.map((option) => option.value),
  fetchCalls,
  readSnapshots,
  modelAOutput: elements["battleground-model-a-output"].textContent,
  modelBOutput: elements["battleground-model-b-output"].textContent,
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
      json: async () => ({ models: ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"] }),
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
elements["model-a-select"].value = "openai/gpt-4o-mini";
elements["model-b-select"].value = "anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "   ";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

elements["model-a-select"].value = "";
elements["model-b-select"].value = "anthropic/claude-3.5-sonnet";
elements["battleground-message"].value = "valid question";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

elements["model-a-select"].value = "openai/gpt-4o-mini";
elements["model-b-select"].value = "";
elements["battleground-message"].value = "valid question";
await elements["battleground-form"].submit();
statuses.push(elements["battleground-status"].textContent);

elements["model-a-select"].value = "openai/gpt-4o-mini";
elements["model-b-select"].value = "openai/gpt-4o-mini";
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


def _run_remove_citation_artifacts(script_source: str, input_text: str) -> str:
    harness = f"""
const commonScript = {json.dumps(script_source)};
const inputText = {json.dumps(input_text)};
globalThis.window = globalThis;
globalThis.marked = {{
  setOptions: () => undefined,
  parse: (value) => value,
}};
globalThis.DOMPurify = {{
  sanitize: (value) => value,
}};
eval(commonScript);
const result = window.RagCommon.removeCitationArtifacts(inputText);
if (typeof result !== "string") {{
  throw new Error("removeCitationArtifacts must return a string");
}}
process.stdout.write(JSON.stringify({{ result }}));
"""
    payload = json.loads(_run_node_harness(harness, "ui common.js behavior test"))
    if "result" not in payload:
        raise RuntimeError("node harness output missing 'result'")
    result = payload["result"]
    if not isinstance(result, str):
        raise RuntimeError("node harness output 'result' must be a string")
    return result


def _run_chat_session_harness(common_script: str, chat_script: str) -> dict[str, object]:
    harness = _CHAT_SESSION_HARNESS_TEMPLATE
    harness = harness.replace("__COMMON_SCRIPT__", json.dumps(common_script))
    harness = harness.replace("__CHAT_SCRIPT__", json.dumps(chat_script))
    harness = harness.replace("__DEFAULT_GREETING__", json.dumps("Hello! How can I assist you today?"))
    payload = json.loads(_run_node_harness(harness, "ui chat.js behavior test"))
    if not isinstance(payload, dict):
        raise RuntimeError("node harness output for chat.js must be an object")
    return payload


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


def _build_index_page_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_client=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    return TestClient(create_app())


def test_index_page_has_chat_and_battleground_scaffolds(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    response = client.get("/")
    html = response.text

    assert response.status_code == 200
    assert 'id="upload-form"' in html
    assert 'id="chat-form"' in html
    assert 'id="nav-chat"' in html
    assert 'id="nav-battleground"' in html
    assert 'id="battleground-form"' in html
    assert 'id="model-a-select"' in html
    assert 'id="model-b-select"' in html
    assert 'id="documents-list"' in html
    assert 'id="refresh-documents"' in html
    assert 'id="chat-history-select"' in html
    assert 'id="clear-chat"' in html
    assert "New Chat" in html
    assert '<script src="/static/js/common.js"></script>' in html
    assert '<script src="/static/js/chat.js"></script>' in html
    assert '<script src="/static/js/battleground.js"></script>' in html
    assert "let conversationHistory = [];" not in html
    assert "Palette: Red, Gray, Black, White" not in html
    assert 'id="documents-panel"' not in html
    assert 'id="nav-documents"' not in html


def test_common_script_removes_citation_artifacts_without_collapsing_newlines(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    response = client.get("/static/js/common.js")
    assert response.status_code == 200
    cleaned = _run_remove_citation_artifacts(
        response.text,
        "Line one    with   spaces\n[source #1]Line two\t\twith tabs",
    )
    assert cleaned == "Line one with spaces\nLine two with tabs"


def test_chat_script_keeps_single_session_when_new_chat_clicked_from_pristine_greeting(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    chat_response = client.get("/static/js/chat.js")
    assert common_response.status_code == 200
    assert chat_response.status_code == 200

    payload = _run_chat_session_harness(common_response.text, chat_response.text)
    assert payload["initialSessionCount"] == 1
    assert payload["initialHistoryLength"] == 0
    assert payload["isInitialGreetingAssistant"] is True
    assert payload["initialGreetingText"] == "Hello! How can I assist you today?"
    assert payload["greetingMatchesDefault"] is True
    assert payload["afterClickSessionCount"] == 1


def test_battleground_script_loads_models_streams_side_outputs_and_preserves_tabs(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    battleground_response = client.get("/static/js/battleground.js")
    assert common_response.status_code == 200
    assert battleground_response.status_code == 200

    payload = _run_battleground_harness(
        common_response.text,
        battleground_response.text,
        _BATTLEGROUND_STREAM_HARNESS_TEMPLATE,
        "ui battleground.js stream behavior test",
    )

    assert payload["modelAOptions"] == [
        "",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["modelBOptions"] == [
        "",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["fetchCalls"][0] == {"url": "/models/battleground", "method": "GET", "body": None}
    assert payload["fetchCalls"][1] == {
        "url": "/battleground/compare/stream",
        "method": "POST",
        "body": {
            "message": "Which answer is better?",
            "history": [],
            "model_a": "openai/gpt-4o-mini",
            "model_b": "anthropic/claude-3.5-sonnet",
        },
    }
    read_snapshots = payload["readSnapshots"]
    assert read_snapshots[1]["modelA"] == "A says hi"
    assert read_snapshots[1]["modelB"] == ""
    assert read_snapshots[2]["modelA"] == "A says hi\nDone."
    assert read_snapshots[2]["modelB"] == "B says hi"
    assert payload["modelAOutput"] == "A says hi\nDone."
    assert payload["modelBOutput"] == "B says hi\nError: B failed"
    assert payload["finalStatus"] == "Comparison complete."
    assert payload["afterBattlegroundTab"] == {
        "chatHidden": True,
        "battlegroundHidden": False,
        "chatSelected": "false",
        "battlegroundSelected": "true",
    }
    assert payload["afterChatTab"] == {
        "chatHidden": False,
        "battlegroundHidden": True,
        "chatSelected": "true",
        "battlegroundSelected": "false",
    }


def test_battleground_script_fails_fast_on_invalid_client_inputs(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    battleground_response = client.get("/static/js/battleground.js")
    assert common_response.status_code == 200
    assert battleground_response.status_code == 200

    payload = _run_battleground_harness(
        common_response.text,
        battleground_response.text,
        _BATTLEGROUND_VALIDATION_HARNESS_TEMPLATE,
        "ui battleground.js validation behavior test",
    )

    assert payload["modelAOptions"] == [
        "",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["modelBOptions"] == [
        "",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["statuses"] == [
        "Enter a prompt before starting comparison.",
        "Choose a model for Model A.",
        "Choose a model for Model B.",
        "Model A and Model B must be different.",
    ]
    assert payload["postCallCount"] == 0
