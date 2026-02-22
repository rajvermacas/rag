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
    async def answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ) -> ChatResult:
        return ChatResult(
            answer="ok",
            citations=[],
            grounded=False,
            retrieved_count=0,
        )

    async def stream_answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ):
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
const fetchCalls = [];
const encoder = new TextEncoder();
let pendingMessage = "";
let releaseStreamCompletion = null;
const streamCompletion = new Promise((resolve) => {
  releaseStreamCompletion = resolve;
});

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
    submit: async function() {
      const handlers = listeners.get("submit") || [];
      for (const handler of handlers) {
        await handler({ preventDefault: () => undefined });
      }
    },
    reset: () => undefined,
  };
}

const ids = [
  "upload-form","upload-button","upload-loader","upload-button-label","upload-status",
  "refresh-documents","documents-list","documents-status","chat-form","chat-button",
  "chat-window","chat-model-select","chat-history-select","clear-chat"
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
globalThis.FormData = function() {
  return {
    get: (name) => {
      if (name === "message") return pendingMessage;
      return null;
    },
  };
};
globalThis.fetch = async (url, options = {}) => {
  const method = typeof options.method === "string" ? options.method : "GET";
  fetchCalls.push({
    url,
    method: method.toUpperCase(),
    body: "body" in options ? JSON.parse(options.body) : null,
  });
  if (url === "/models/chat") {
    return {
      ok: true,
      json: async () => ({
        models: [
          {
            backend_id: "lab_vllm",
            provider: "openai_compatible",
            model: "openai/gpt-4o-mini",
            label: "lab_vllm (openai_compatible) · openai/gpt-4o-mini",
          },
          {
            backend_id: "lab_vllm",
            provider: "openai_compatible",
            model: "anthropic/claude-3.5-sonnet",
            label: "lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet",
          },
        ],
      }),
    };
  }
  if (url === "/documents") return { ok: true, json: async () => ({ documents: [] }) };
  if (url === "/chat/stream") {
    let readCount = 0;
    return {
      ok: true,
      body: {
        getReader: () => ({
          read: async () => {
            if (readCount === 0) {
              readCount += 1;
              return { value: encoder.encode(__STREAM_CHUNK__), done: false };
            }
            if (readCount === 1) {
              readCount += 1;
              await streamCompletion;
              return { value: undefined, done: true };
            }
            throw new Error("unexpected chat stream read invocation");
          },
        }),
      },
      json: async () => ({ detail: "unused" }),
    };
  }
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
const chatModelOptionValues = elements["chat-model-select"].children.map((option) => option.value);
const chatModelOptionLabels = elements["chat-model-select"].children.map((option) => option.textContent);
elements["chat-model-select"].value = "lab_vllm||openai/gpt-4o-mini";
pendingMessage = "What is revenue?";
const submitPromise = elements["chat-form"].submit();
await Promise.resolve();
const chatButtonDisabledWhileStreamCompleting = elements["chat-button"].disabled;
releaseStreamCompletion();
await submitPromise;
const streamRequest = fetchCalls.find((call) => call.url === "/chat/stream");
const postSubmitPayload = JSON.parse(localStorageRecords.get("rag-chat-sessions"));
const postSubmitSession = postSubmitPayload.sessions[0];
const lastMessage = postSubmitSession.messages[postSubmitSession.messages.length - 1];

process.stdout.write(JSON.stringify({
  initialSessionCount: initialPayload.sessions.length,
  initialHistoryLength: initialSession.history.length,
  initialGreetingText: initialMessage.text,
  isInitialGreetingAssistant: initialMessage.role === "assistant",
  greetingMatchesDefault: initialMessage.text === defaultGreeting,
  afterClickSessionCount: afterClickPayload.sessions.length,
  chatModelOptionValues,
  chatModelOptionLabels,
  streamRequestBody: streamRequest ? streamRequest.body : null,
  chatButtonDisabledWhileStreamCompleting,
  lastAssistantRole: lastMessage.role,
  lastAssistantMessageText: lastMessage.text
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


def _run_chat_session_harness(
    common_script: str,
    chat_script: str,
    stream_chunk: str = "Revenue is 20.",
) -> dict[str, object]:
    harness = _CHAT_SESSION_HARNESS_TEMPLATE
    harness = harness.replace("__COMMON_SCRIPT__", json.dumps(common_script))
    harness = harness.replace("__CHAT_SCRIPT__", json.dumps(chat_script))
    harness = harness.replace("__DEFAULT_GREETING__", json.dumps("Hello! How can I assist you today?"))
    harness = harness.replace("__STREAM_CHUNK__", json.dumps(stream_chunk))
    payload = json.loads(_run_node_harness(harness, "ui chat.js behavior test"))
    if not isinstance(payload, dict):
        raise RuntimeError("node harness output for chat.js must be an object")
    return payload


def _build_index_page_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_provider_router=object(),
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
    assert 'id="battleground-model-a-title"' in html
    assert 'id="battleground-model-b-title"' in html
    assert 'id="battleground-comparison-panel"' in html
    assert "Ask a question for both models..." in html
    assert "Comparison Output" not in html
    assert "hidden grid gap-6 rounded-2xl border border-zinc-200 bg-white p-4 shadow-xl shadow-zinc-300/30 md:p-6" in html
    assert 'id="battleground-model-selection"' in html
    assert 'class="mt-4 flex flex-col gap-3 sm:flex-row"' in html
    assert "xl:grid-cols-2" in html
    assert 'id="documents-list"' in html
    assert 'id="refresh-documents"' in html
    assert 'id="chat-history-select"' in html
    assert 'id="chat-model-select"' in html
    assert 'id="clear-chat"' in html
    assert 'id="chat-form" class="mt-4 flex flex-col gap-3 sm:flex-row" autocomplete="off"' in html
    assert 'name="message"' in html
    assert 'autocomplete="off"' in html
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
    assert payload["chatModelOptionValues"] == [
        "",
        "lab_vllm||openai/gpt-4o-mini",
        "lab_vllm||anthropic/claude-3.5-sonnet",
    ]
    assert payload["chatModelOptionLabels"] == [
        "Select model",
        "lab_vllm (openai_compatible) · openai/gpt-4o-mini",
        "lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet",
    ]
    assert payload["streamRequestBody"] == {
        "message": "What is revenue?",
        "history": [{"role": "user", "message": "What is revenue?"}],
        "backend_id": "lab_vllm",
        "model": "openai/gpt-4o-mini",
    }
    assert payload["chatButtonDisabledWhileStreamCompleting"] is False


def test_chat_script_replaces_citation_only_stream_with_visible_assistant_message(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    chat_response = client.get("/static/js/chat.js")
    assert common_response.status_code == 200
    assert chat_response.status_code == 200

    payload = _run_chat_session_harness(
        common_response.text,
        chat_response.text,
        stream_chunk="[a.txt#0]",
    )

    assert payload["lastAssistantRole"] == "assistant"
    assert payload["lastAssistantMessageText"] == (
        "I could not generate a response from the current context. "
        "Please rephrase your question."
    )


def test_chat_script_replaces_empty_response_sentinel_with_visible_assistant_message(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    chat_response = client.get("/static/js/chat.js")
    assert common_response.status_code == 200
    assert chat_response.status_code == 200

    payload = _run_chat_session_harness(
        common_response.text,
        chat_response.text,
        stream_chunk="Empty Response",
    )

    assert payload["lastAssistantRole"] == "assistant"
    assert payload["lastAssistantMessageText"] == (
        "I could not generate a response from the current context. "
        "Please rephrase your question."
    )


def test_compiled_css_contains_chat_alignment_utilities(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    response = client.get("/static/css/output.css")
    assert response.status_code == 200
    css = response.text

    assert ".justify-end{" in css
    assert ".justify-start{" in css
