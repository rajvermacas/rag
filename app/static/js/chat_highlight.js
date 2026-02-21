function applyDocumentHighlights(markdownContainer, citations) {
  if (!(markdownContainer instanceof Element)) {
    throw new Error("markdown container must be an Element");
  }
  if (!Array.isArray(citations)) {
    throw new Error("assistant citations must be an array");
  }
  citations.forEach((citation, index) => {
    validateCitationForHighlight(citation, index);
    highlightCitationInContainer(markdownContainer, citation, index);
  });
}

function validateCitationForHighlight(citation, index) {
  const context = `assistant_message.citations[${index}]`;
  if (typeof citation !== "object" || citation === null) {
    throw new Error(`citation must be an object in ${context}`);
  }
  if (!("text" in citation) || typeof citation.text !== "string") {
    throw new Error(`citation text must be a string in ${context}`);
  }
  if (citation.text.trim() === "") {
    throw new Error(`citation text must not be empty in ${context}`);
  }
  if (!("filename" in citation) || typeof citation.filename !== "string") {
    throw new Error(`citation filename must be a string in ${context}`);
  }
}

function highlightCitationInContainer(container, citation, citationIndex) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (true) {
    const textNode = walker.nextNode();
    if (textNode === null) {
      break;
    }
    textNodes.push(textNode);
  }

  textNodes.forEach((textNode) => {
    const parentElement = textNode.parentElement;
    if (parentElement === null) {
      throw new Error("text node parent element is required");
    }
    if (parentElement.closest("code, pre, .source-highlight") !== null) {
      return;
    }
    const sourceText = textNode.textContent;
    if (sourceText === null) {
      throw new Error("text node content must not be null");
    }
    const highlightedNodes = buildHighlightedNodes(sourceText, citation, citationIndex);
    if (highlightedNodes === null) {
      return;
    }
    if (textNode.parentNode === null) {
      throw new Error("text node parent must not be null");
    }
    const fragment = document.createDocumentFragment();
    highlightedNodes.forEach((node) => fragment.append(node));
    textNode.parentNode.replaceChild(fragment, textNode);
  });
}

function buildHighlightedNodes(text, citation, citationIndex) {
  const highlightedText = citation.text;
  let cursor = 0;
  let matchIndex = text.indexOf(highlightedText);
  if (matchIndex === -1) {
    return null;
  }

  const nodes = [];
  while (matchIndex !== -1) {
    if (matchIndex > cursor) {
      nodes.push(document.createTextNode(text.slice(cursor, matchIndex)));
    }
    const highlightedSpan = document.createElement("span");
    highlightedSpan.className = "source-highlight rounded bg-emerald-100 px-1 font-semibold text-emerald-800 ring-1 ring-emerald-300/70";
    highlightedSpan.title = `Source document: ${citation.filename}`;
    highlightedSpan.setAttribute("data-source-document", citation.filename);
    highlightedSpan.setAttribute("data-citation-index", String(citationIndex));
    highlightedSpan.textContent = highlightedText;
    nodes.push(highlightedSpan);
    cursor = matchIndex + highlightedText.length;
    matchIndex = text.indexOf(highlightedText, cursor);
  }
  if (cursor < text.length) {
    nodes.push(document.createTextNode(text.slice(cursor)));
  }
  return nodes;
}
