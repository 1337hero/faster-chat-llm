//! Parser for the model's `<tool_call>` output format.
//!
//! Format:
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=ARG>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Incremental-safe: handles tags split across Event::Text chunks.

use serde_json::{Map, Value, json};

/// A parsed tool call.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value, // object with parameter values
}

/// Incrementally parses tool calls from streamed text.
pub struct ToolCallParser {
    buf: String,
    next_id: u32,
    /// Text that is safe to emit as content (before any tool_call start).
    pending_text: String,
    /// Completed tool calls ready to emit.
    pending_calls: Vec<ToolCall>,
    /// Are we inside a <tool_call> block?
    in_tool_call: bool,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            next_id: 0,
            pending_text: String::new(),
            pending_calls: Vec::new(),
            in_tool_call: false,
        }
    }

    /// Push new text from the stream. Returns (safe_text, completed_calls).
    /// `last` indicates this is the final chunk (flushes any held-back text).
    pub fn push(&mut self, text: &str, last: bool) -> (String, Vec<ToolCall>) {
        self.buf.push_str(text);
        self.parse_buffer(last);

        let text_out = std::mem::take(&mut self.pending_text);
        let calls_out = std::mem::take(&mut self.pending_calls);
        (text_out, calls_out)
    }

    fn parse_buffer(&mut self, last: bool) {
        loop {
            if !self.in_tool_call {
                // Look for <tool_call> start
                if let Some(pos) = self.buf.find("<tool_call>") {
                    // Everything before is safe text
                    self.pending_text.push_str(&self.buf[..pos]);
                    self.buf.drain(..pos + "<tool_call>".len());
                    self.in_tool_call = true;
                    continue;
                }
                // Hold back a tail that could be the start of <tool_call>
                let hold = if last {
                    0
                } else {
                    "<tool_call>".len().saturating_sub(1).min(self.buf.len())
                };
                let safe = self.buf.len().saturating_sub(hold);
                // Ensure we don't split a char
                let mut safe = safe;
                while safe > 0 && !self.buf.is_char_boundary(safe) {
                    safe -= 1;
                }
                if safe > 0 {
                    self.pending_text.push_str(&self.buf[..safe]);
                    self.buf.drain(..safe);
                }
                break;
            } else {
                // Inside a tool_call, look for </tool_call>
                if let Some(end_pos) = self.buf.find("</tool_call>") {
                    let call_text = self.buf[..end_pos].to_string();
                    if let Some(call) = self.parse_tool_call(&call_text) {
                        self.pending_calls.push(call);
                    }
                    self.buf.drain(..end_pos + "</tool_call>".len());
                    self.in_tool_call = false;
                    continue;
                }
                // Not complete yet; if last, discard malformed content
                if last {
                    self.buf.clear();
                    self.in_tool_call = false;
                }
                break;
            }
        }
    }

    fn parse_tool_call(&mut self, text: &str) -> Option<ToolCall> {
        // Extract function name from <function=NAME>
        let func_start = text.find("<function=")?;
        let name_start = func_start + "<function=".len();
        let name_end = text[name_start..].find('>')? + name_start;
        let name = text[name_start..name_end].trim().to_string();

        if name.is_empty() {
            return None;
        }

        // Extract parameters
        let mut arguments = Map::new();
        let func_content_start = name_end + 1;
        let func_end = text[func_content_start..].find("</function>")? + func_content_start;
        let func_body = &text[func_content_start..func_end];

        // Parse each <parameter=NAME>VALUE</parameter>
        let mut rest = func_body;
        while let Some(param_start) = rest.find("<parameter=") {
            let after_tag = param_start + "<parameter=".len();
            let Some(name_end_rel) = rest[after_tag..].find('>') else {
                break;
            };
            let param_name = rest[after_tag..after_tag + name_end_rel].trim();

            let value_start = after_tag + name_end_rel + 1;
            let Some(end_tag) = rest[value_start..].find("</parameter>") else {
                break;
            };
            let value_text = rest[value_start..value_start + end_tag].trim();

            // Try parsing as JSON; if that fails, it's a string
            let value: Value = serde_json::from_str(value_text)
                .unwrap_or_else(|_| Value::String(value_text.to_string()));

            arguments.insert(param_name.to_string(), value);
            rest = &rest[value_start + end_tag + "</parameter>".len()..];
        }

        let id = format!("call_{}", self.next_id);
        self.next_id += 1;

        Some(ToolCall {
            id,
            name,
            arguments: Value::Object(arguments),
        })
    }

    // ponytail: in_progress() removed - not currently needed, add back when streaming requires it
}

/// Format tool calls for OpenAI Chat Completions response.
pub fn tool_calls_to_openai(calls: &[ToolCall]) -> Value {
    let arr: Vec<Value> = calls
        .iter()
        .map(|c| {
            json!({
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.arguments.to_string()
                }
            })
        })
        .collect();
    Value::Array(arr)
}

/// Format tool calls for Anthropic Messages response (tool_use blocks).
pub fn tool_calls_to_anthropic(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .map(|c| {
            json!({
                "type": "tool_use",
                "id": c.id,
                "name": c.name,
                "input": c.arguments
            })
        })
        .collect()
}

// ponytail: tool_calls_to_responses removed - Responses API formats inline, add back if needed

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_call() {
        let mut p = ToolCallParser::new();
        let input = r#"Sure! <tool_call>
<function=get_weather>
<parameter=location>
"San Francisco"
</parameter>
<parameter=unit>
"celsius"
</parameter>
</function>
</tool_call>"#;

        let (text, calls) = p.push(input, true);
        assert_eq!(text, "Sure! ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "San Francisco");
        assert_eq!(calls[0].arguments["unit"], "celsius");
    }

    #[test]
    fn parse_multiple_calls() {
        let mut p = ToolCallParser::new();
        let input = r#"<tool_call>
<function=foo>
<parameter=a>1</parameter>
</function>
</tool_call>middle<tool_call>
<function=bar>
<parameter=b>"x"</parameter>
</function>
</tool_call>end"#;

        let (text, calls) = p.push(input, true);
        assert_eq!(text, "middleend");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "foo");
        assert_eq!(calls[0].arguments["a"], 1);
        assert_eq!(calls[1].name, "bar");
        assert_eq!(calls[1].arguments["b"], "x");
    }

    #[test]
    fn incremental_across_chunks() {
        let mut p = ToolCallParser::new();

        // Split right in the middle of <tool_call>
        // We hold back up to 10 chars to detect partial tags, so "hello <tool_" (12 chars)
        // emits only "he" (12 - 10 = 2).
        let (t1, c1) = p.push("hello <tool_", false);
        assert_eq!(t1, "he");
        assert!(c1.is_empty());

        let (t2, c2) = p.push(
            "call>\n<function=test>\n</function>\n</tool_call>done",
            true,
        );
        // "llo <tool_call>..." completes the tag, so "llo " goes before it and "done" after.
        assert_eq!(t2, "llo done");
        assert_eq!(c2.len(), 1);
        assert_eq!(c2[0].name, "test");
    }

    #[test]
    fn no_tool_calls() {
        let mut p = ToolCallParser::new();
        let (text, calls) = p.push("Just regular text with no calls", true);
        assert_eq!(text, "Just regular text with no calls");
        assert!(calls.is_empty());
    }

    #[test]
    fn json_number_parameter() {
        let mut p = ToolCallParser::new();
        let input = r#"<tool_call>
<function=calculate>
<parameter=x>42</parameter>
<parameter=y>1.25</parameter>
</function>
</tool_call>"#;

        let (_, calls) = p.push(input, true);
        assert_eq!(calls[0].arguments["x"], 42);
        assert!((calls[0].arguments["y"].as_f64().unwrap() - 1.25).abs() < 0.001);
    }

    #[test]
    fn malformed_incomplete() {
        let mut p = ToolCallParser::new();
        // Incomplete tool call at end, no closing tag
        let (text, calls) = p.push("text<tool_call><function=bad>", true);
        assert_eq!(text, "text");
        assert!(calls.is_empty()); // Malformed is discarded
    }
}
