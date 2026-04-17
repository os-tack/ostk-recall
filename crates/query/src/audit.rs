//! `recall_audit` — SELECT-only `DuckDB` passthrough over `audit_events`.

use ostk_recall_store::EventsDb;

use crate::error::{QueryError, Result};
use crate::types::AuditResult;

pub fn recall_audit(events: &EventsDb, sql: &str) -> Result<AuditResult> {
    gate(sql)?;
    let (columns, rows) = events.execute_select(sql)?;
    Ok(AuditResult { columns, rows })
}

/// Reject anything that isn't a single SELECT statement. Cheap guard; the
/// underlying `DuckDB` connection is our ultimate defense-in-depth.
fn gate(sql: &str) -> Result<()> {
    let trimmed = sql.trim_start();
    let head: String = trimmed
        .chars()
        .take(8)
        .collect::<String>()
        .to_ascii_uppercase();
    if !(head.starts_with("SELECT ")
        || head.starts_with("SELECT\t")
        || head.starts_with("SELECT\n")
        || head.starts_with("WITH ")
        || head.starts_with("WITH\t")
        || head.starts_with("WITH\n"))
    {
        return Err(QueryError::Forbidden(
            "only SELECT / WITH statements are allowed".into(),
        ));
    }
    // Disallow stacked statements. A semicolon followed by *any* non-whitespace
    // text is a hard no. Allows trailing `;` though.
    let mut stacked = false;
    let mut after_semi = false;
    for c in sql.chars() {
        if after_semi && !c.is_whitespace() {
            stacked = true;
            break;
        }
        if c == ';' {
            after_semi = true;
        }
    }
    if stacked {
        return Err(QueryError::Forbidden(
            "stacked statements are not allowed".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_allowed() {
        assert!(gate("SELECT 1").is_ok());
        assert!(gate("  select * from audit_events").is_ok());
        assert!(gate("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
    }

    #[test]
    fn non_select_rejected() {
        assert!(gate("INSERT INTO audit_events VALUES (1)").is_err());
        assert!(gate("DELETE FROM audit_events").is_err());
        assert!(gate("DROP TABLE audit_events").is_err());
        assert!(gate("").is_err());
    }

    #[test]
    fn stacked_rejected() {
        assert!(gate("SELECT 1; SELECT 2").is_err());
        assert!(gate("SELECT 1; DROP TABLE audit_events").is_err());
    }

    #[test]
    fn trailing_semicolon_ok() {
        assert!(gate("SELECT 1;").is_ok());
        assert!(gate("SELECT 1;  \n").is_ok());
    }
}
