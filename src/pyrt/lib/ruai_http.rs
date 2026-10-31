//! `ruai.http`: an HTTP(S) client for py0 programs (`from ruai.http import request`).
//!
//! `request(method, url, headers, body, timeout)` returns a `Response` whose
//! status is any HTTP status (4xx/5xx are not errors); its body is read line
//! by line (`readline`, for server-sent events) or at once (`read`). Failing to
//! connect raises OSError. CPython stand-in: `py0/python/ruai/http.py` (urllib).

use crate::pyrt::err::{PyErr, OS_ERROR};
use indexmap::IndexMap;
use std::io::{BufRead, BufReader, Read};
use std::time::Duration;

type Body = std::rc::Rc<std::cell::RefCell<BufReader<Box<dyn Read + Send>>>>;

/// An HTTP response; the body is read as it arrives. Copies share the body
/// (like names of one Python object).
#[derive(Clone)]
pub struct Response {
    status: i64,
    body: Body,
}

impl std::fmt::Debug for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Response({})", self.status)
    }
}

impl PartialEq for Response {
    fn eq(&self, o: &Response) -> bool {
        std::rc::Rc::ptr_eq(&self.body, &o.body)
    }
}

impl Response {
    /// `r.status`
    pub fn status(&self) -> i64 {
        self.status
    }
    /// The next line of the body with its `\n`; `""` at the end.
    pub fn readline(&mut self) -> Result<String, PyErr> {
        let mut buf = Vec::new();
        self.body.borrow_mut().read_until(b'\n', &mut buf).map_err(io_err)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
    /// The rest of the body.
    pub fn read(&mut self) -> Result<String, PyErr> {
        let mut buf = Vec::new();
        self.body.borrow_mut().read_to_end(&mut buf).map_err(io_err)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

fn io_err(e: std::io::Error) -> PyErr {
    PyErr::new(OS_ERROR, e.to_string())
}

/// `request(method, url, headers, body, timeout)`: `body` is sent when not empty.
pub fn request(method: &str, url: &str, headers: &IndexMap<String, String>, body: &str, timeout: f64) -> Result<Response, PyErr> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs_f64(timeout.max(0.001))))
        .build()
        .into();
    let mut b = ureq::http::Request::builder().method(method).uri(url);
    for (k, v) in headers {
        b = b.header(k.as_str(), v.as_str());
    }
    let req = b.body(body.as_bytes().to_vec()).map_err(|e| PyErr::new(crate::pyrt::err::VALUE_ERROR, e.to_string()))?;
    let resp = agent.run(req).map_err(|e| PyErr::new(OS_ERROR, format!("<urlopen error {}>", e)))?;
    let status = resp.status().as_u16() as i64;
    let reader: Box<dyn Read + Send> = Box::new(resp.into_body().into_reader());
    Ok(Response { status, body: std::rc::Rc::new(std::cell::RefCell::new(BufReader::new(reader))) })
}
