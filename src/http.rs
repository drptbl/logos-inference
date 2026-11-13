//! What both HTTP servers share.

use std::io::Read;

/// The largest request body read. logos-agents caps an input at 8 MiB.
pub const MAX_BODY: u64 = 16 << 20;

/// A request's body, at most `MAX_BODY`, or the status and reason to refuse it with.
pub fn read_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, (u16, &'static str)> {
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut body)
        .map_err(|_| (400, "the request couldn't be read"))?;
    if body.len() as u64 > MAX_BODY {
        return Err((413, "the request is too big"));
    }
    Ok(body)
}
