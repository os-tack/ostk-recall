use ostk_recall_core::{ContextRole, RecallHit, SynthesizedPage};
use std::collections::HashMap;

pub struct Synthesizer;

impl Synthesizer {
    /// Synthesizes multiple retrieval hits into a single, high-signal Virtual Memory Page.
    pub fn collapse(candidates: Vec<RecallHit>) -> Vec<SynthesizedPage> {
        if candidates.is_empty() {
            return vec![];
        }

        // Identity Resolution: Group hits by symbol_name (if present) or source_id
        let mut groups: HashMap<String, Vec<RecallHit>> = HashMap::new();
        for hit in candidates {
            let key = hit
                .extra
                .get("symbol_name")
                .and_then(|v| v.as_str())
                .unwrap_or(&hit.source_id)
                .to_string();
            groups.entry(key).or_default().push(hit);
        }

        let mut pages = Vec::new();
        for (group_key, mut group) in groups {
            // Sort by score descending within each group to pick the best Primary
            group.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut head = None;
            let mut lineage = Vec::new();
            let mut evidence = Vec::new();

            for mut hit in group {
                let role = Self::calculate_role(&hit);
                hit.role = Some(role);

                match role {
                    ContextRole::Primary if head.is_none() => {
                        head = Some(hit);
                    }
                    ContextRole::Primary | ContextRole::Evolution => {
                        lineage.push(hit);
                    }
                    ContextRole::Usage => {
                        evidence.push(hit);
                    }
                }
            }

            // Fallback: if no Primary was found, pick the best hit as head
            if head.is_none() {
                if !lineage.is_empty() {
                    head = Some(lineage.remove(0));
                } else if !evidence.is_empty() {
                    head = Some(evidence.remove(0));
                }
            }

            if let Some(h) = head {
                let is_symbol = h.extra.get("symbol_name").is_some();
                let title = if is_symbol {
                    format!("Symbol: {group_key}")
                } else {
                    format!("Page: {group_key}")
                };

                let total_lineage = lineage.len();
                let total_evidence = evidence.len();

                // Lazy Loading: Truncate lineage and evidence to keep page size manageable.
                // Downstream can fetch more if total > len.
                if lineage.len() > 3 {
                    lineage.truncate(3);
                }
                if evidence.len() > 3 {
                    evidence.truncate(3);
                }

                let summary = format!(
                    "Synthesized memory for {group_key} with {total_lineage} evolution and {total_evidence} evidence points."
                );

                pages.push(SynthesizedPage {
                    title,
                    head: h,
                    lineage,
                    evidence,
                    total_lineage,
                    total_evidence,
                    summary,
                });
            }
        }

        // Re-sort pages by the head score to maintain global ranking relevance
        pages.sort_by(|a, b| {
            b.head
                .score
                .partial_cmp(&a.head.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pages
    }

    fn calculate_role(hit: &RecallHit) -> ContextRole {
        if hit.stale {
            ContextRole::Evolution
        } else if hit.source == "transcript" || hit.source == "probe" {
            ContextRole::Usage
        } else {
            ContextRole::Primary
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::{ContextRole, Links};
    use serde_json::json;

    fn fake_hit(
        chunk_id: &str,
        source_id: &str,
        score: f32,
        source: &str,
        stale: bool,
    ) -> RecallHit {
        RecallHit {
            chunk_id: chunk_id.to_string(),
            project: None,
            source: source.to_string(),
            source_id: source_id.to_string(),
            ts: None,
            snippet: "snippet".to_string(),
            score,
            links: Links::default(),
            extra: serde_json::Value::Null,
            stale,
            role: None,
            base_score: None,
            thread_score: None,
            embedding_score: None,
            thread_weight: None,
            embedding_weight: None,
            attention_score: None,
            attention_weight: None,
        }
    }

    #[test]
    fn collapse_groups_by_source_id() {
        let hits = vec![
            fake_hit("c1", "src/main.rs", 0.9, "code", false),
            fake_hit("c2", "src/main.rs", 0.8, "code", true),
            fake_hit("t1", "transcript1", 0.7, "transcript", false),
        ];

        let pages = Synthesizer::collapse(hits);
        assert_eq!(pages.len(), 2);

        let main_page = pages
            .iter()
            .find(|p| p.head.source_id == "src/main.rs")
            .unwrap();
        assert_eq!(main_page.head.chunk_id, "c1");
        assert_eq!(main_page.lineage.len(), 1);
        assert_eq!(main_page.total_lineage, 1);
        assert_eq!(main_page.lineage[0].chunk_id, "c2");
        assert_eq!(main_page.head.role, Some(ContextRole::Primary));
        assert_eq!(main_page.lineage[0].role, Some(ContextRole::Evolution));

        let t_page = pages
            .iter()
            .find(|p| p.head.source_id == "transcript1")
            .unwrap();
        assert_eq!(t_page.head.role, Some(ContextRole::Usage));
    }

    #[test]
    fn collapse_groups_by_symbol_name() {
        let mut h1 = fake_hit("c1", "src/main.rs", 0.9, "code", false);
        h1.extra = json!({"symbol_name": "main"});

        let mut h2 = fake_hit("c2", "src/old.rs", 0.8, "code", false);
        h2.extra = json!({"symbol_name": "main"});

        let hits = vec![h1, h2];
        let pages = Synthesizer::collapse(hits);

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].title, "Symbol: main");
        assert_eq!(pages[0].head.chunk_id, "c1");
        assert_eq!(pages[0].lineage.len(), 1);
        assert_eq!(pages[0].total_lineage, 1);
    }

    #[test]
    fn collapse_truncates_for_lazy_loading() {
        let mut hits = vec![fake_hit("p", "S", 1.0, "code", false)];
        for i in 0..10 {
            hits.push(fake_hit(&format!("l{i}"), "S", 0.5, "code", true));
        }

        let pages = Synthesizer::collapse(hits);
        assert_eq!(pages[0].total_lineage, 10);
        assert_eq!(pages[0].lineage.len(), 3);
    }
}
