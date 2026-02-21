(() => {
  "use strict";

  if (typeof marked === "undefined") {
    throw new Error("marked is required for markdown rendering");
  }
  if (typeof DOMPurify === "undefined") {
    throw new Error("DOMPurify is required for markdown rendering");
  }

  const CITATION_ARTIFACT_PATTERN = /\[[^\]\n]*(?:#chunk_id\s*=\s*\d+|#\d+)[^\]\n]*\]/gi;

  marked.setOptions({
    breaks: true,
    gfm: true,
  });

  function requireElement(id) {
    const element = document.getElementById(id);
    if (element === null) {
      throw new Error(`missing required element: ${id}`);
    }
    return element;
  }

  function requireString(record, key, context) {
    if (!(key in record)) {
      throw new Error(`missing '${key}' in ${context}`);
    }
    const value = record[key];
    if (typeof value !== "string") {
      throw new Error(`'${key}' must be a string in ${context}`);
    }
    return value;
  }

  function requireNumber(record, key, context) {
    if (!(key in record)) {
      throw new Error(`missing '${key}' in ${context}`);
    }
    const value = record[key];
    if (typeof value !== "number") {
      throw new Error(`'${key}' must be a number in ${context}`);
    }
    return value;
  }

  function requireErrorMessage(error) {
    if (error instanceof Error) {
      return error.message;
    }
    throw new Error("chat error must be an Error instance");
  }

  function renderMarkdown(markdownText) {
    if (typeof markdownText !== "string") {
      throw new Error("markdown input must be a string");
    }
    const html = marked.parse(markdownText);
    return DOMPurify.sanitize(html);
  }

  function removeCitationArtifacts(text) {
    if (typeof text !== "string") {
      throw new Error("citation cleanup input must be a string");
    }
    return text.replace(CITATION_ARTIFACT_PATTERN, "").replace(/[ \t]{2,}/g, " ").trimStart();
  }

  function escapeHtml(value) {
    if (typeof value !== "string") {
      throw new Error("escapeHtml input must be a string");
    }
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll("\"", "&quot;")
      .replaceAll("'", "&#39;");
  }

  window.RagCommon = Object.freeze({
    requireElement,
    requireString,
    requireNumber,
    requireErrorMessage,
    renderMarkdown,
    removeCitationArtifacts,
    escapeHtml,
  });
})();
