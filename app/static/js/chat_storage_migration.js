function migrateLegacyChatState(parsedState) {
  if (typeof parsedState !== "object" || parsedState === null) {
    throw new Error("persisted chat state must be an object");
  }
  if (!("sessions" in parsedState) || !Array.isArray(parsedState.sessions)) {
    throw new Error("persisted chat state missing sessions array");
  }

  let migrated = false;
  const migratedSessions = parsedState.sessions.map((sessionRecord, sessionIndex) => {
    if (typeof sessionRecord !== "object" || sessionRecord === null) {
      throw new Error(`session[${sessionIndex}] must be an object`);
    }
    if (!("messages" in sessionRecord) || !Array.isArray(sessionRecord.messages)) {
      throw new Error(`session[${sessionIndex}] missing messages array`);
    }

    const migratedMessages = sessionRecord.messages.map((entry, messageIndex) => {
      if (typeof entry !== "object" || entry === null) {
        throw new Error(`session[${sessionIndex}].messages[${messageIndex}] must be an object`);
      }
      if (!("role" in entry) || typeof entry.role !== "string") {
        throw new Error(`session[${sessionIndex}].messages[${messageIndex}] missing role`);
      }
      if (!("text" in entry) || typeof entry.text !== "string") {
        throw new Error(`session[${sessionIndex}].messages[${messageIndex}] missing text`);
      }
      if ("citations" in entry) {
        return entry;
      }
      migrated = true;
      return { ...entry, citations: [] };
    });

    return { ...sessionRecord, messages: migratedMessages };
  });

  if (!migrated) {
    return parsedState;
  }

  console.info("chat_storage_migration_applied added_missing_citations=true");
  return { ...parsedState, sessions: migratedSessions };
}
