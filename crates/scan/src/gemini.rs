use std::fs;

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use serde::Deserialize;

/// Scanner for Gemini session JSON logs.
#[derive(Debug, Default)]
pub struct GeminiScanner;

#[derive(Debug, Deserialize)]
struct GeminiSession {
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "projectHash")]
    project_hash: Option<String>,
    messages: Vec<GeminiMessage>,
}

#[derive(Debug, Deserialize)]
struct GeminiMessage {
    #[serde(rename = "type")]
    msg_type: String, // "user", "gemini"
    content: Vec<ContentPart>,
    timestamp: String,
    #[serde(default)]
    tokens: Option<TokenInfo>,
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenInfo {
    thoughts: Option<u32>,
}

impl Scanner for GeminiScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::Gemini
    }

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        let roots = match cfg.expanded_paths() {
            Ok(v) => v,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let project = cfg.project.clone();

        let iter = roots.into_iter().flat_map(move |root| {
            let project = project.clone();
            walkdir::WalkDir::new(&root)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .filter(|e| {
                    e.file_name().to_string_lossy().starts_with("session-")
                        && e.path().extension().is_some_and(|x| x == "json")
                })
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    let source_id = path.file_name().unwrap().to_string_lossy().into_owned();
                    Ok(SourceItem {
                        source_id,
                        path: Some(path),
                        project: project.clone(),
                        bytes: None,
                        ignore: Vec::new(),
                    })
                })
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let path = item
            .path
            .as_ref()
            .ok_or_else(|| Error::Parse("gemini: path missing".into()))?;
        let content = fs::read_to_string(path)?;
        let session: GeminiSession = serde_json::from_str(&content)
            .map_err(|e| Error::Parse(format!("gemini json: {e}")))?;

        let mut chunks = Vec::new();
        let mut user_msg: Option<&GeminiMessage> = None;

        for msg in &session.messages {
            if msg.msg_type == "user" {
                user_msg = Some(msg);
            } else if msg.msg_type == "gemini" {
                if let Some(user) = user_msg {
                    let user_text = extract_text(user);
                    let gemini_text = extract_text(msg);

                    let combined_text =
                        format!("### User\n{}\n\n### Gemini\n{}", user_text, gemini_text);
                    let ts = DateTime::parse_from_rfc3339(&user.timestamp)
                        .or_else(|_| DateTime::parse_from_rfc3339(&msg.timestamp))
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc));

                    let chunk_index = chunks.len() as u32;
                    let chunk_id = Chunk::make_id(Source::Gemini, &item.source_id, chunk_index);

                    let mut extra = serde_json::json!({
                        "session_id": session.session_id,
                        "project_hash": session.project_hash,
                    });

                    if let Some(tokens) = &msg.tokens {
                        if let Some(thoughts) = tokens.thoughts {
                            extra
                                .as_object_mut()
                                .unwrap()
                                .insert("thought_tokens".into(), thoughts.into());
                        }
                    }

                    chunks.push(Chunk {
                        chunk_id,
                        source: Source::Gemini,
                        project: item.project.clone(),
                        source_id: item.source_id.clone(),
                        chunk_index,
                        ts,
                        role: Some("exchange".into()),
                        text: combined_text.clone(),
                        sha256: Chunk::content_hash(&combined_text),
                        links: Links::default(),
                        extra,
                    });

                    user_msg = None;
                }
            }
        }

        Ok(chunks)
    }
}

fn extract_text(msg: &GeminiMessage) -> String {
    msg.content
        .iter()
        .filter_map(|p| p.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n")
}
