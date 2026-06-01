//! Bounded exponential-backoff retry helper.

use std::time::Duration;

/// Retry `op` up to `max_attempts` times, sleeping with exponential backoff
/// (base 50ms, doubling, capped at 5s) between attempts. Returns the first
/// `Ok`, or the last `Err` once attempts are exhausted.
pub fn retry_with_backoff<T, E, F>(max_attempts: u32, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut attempt = 0;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) => {
                attempt += 1;
                if attempt >= max_attempts {
                    return Err(e);
                }
                let backoff = Duration::from_millis(50u64 << attempt.min(7));
                std::thread::sleep(backoff.min(Duration::from_secs(5)));
            }
        }
    }
}
