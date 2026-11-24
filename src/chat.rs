//! Chat prompt rendering with the GGUF's own Jinja template.

use minijinja::{Environment, Error, ErrorKind, context};
use serde_json::Value;

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, String> {
        let mut env = Environment::new();
        // Hugging Face renders chat templates with these two on; the templates rely on it.
        let syntax = minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|e| e.to_string())?;
        env.set_syntax(syntax);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", source)
            .map_err(|e| format!("chat template: {e}"))?;
        Ok(Self { env })
    }

    /// Render `messages` with a generation prompt. `kwargs` are extra template variables
    /// (e.g. `enable_thinking`); `tools` is passed through untouched.
    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        kwargs: &serde_json::Map<String, Value>,
    ) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let messages = &with_object_arguments(messages);
        use minijinja::value::Serde;
        let base = context! { messages => Serde(messages), tools => Serde(tools), add_generation_prompt => true };
        let extra = minijinja::Value::from(Serde(kwargs));
        tmpl.render(context! { ..extra, ..base })
            .map_err(|e| format!("chat template: {e:#}"))
    }
}

/// OpenAI clients send a tool call's `arguments` as a JSON string; the template iterates them
/// as a mapping. Parse the strings so both forms render the same.
fn with_object_arguments(messages: &Value) -> Value {
    let mut messages = messages.clone();
    for call in messages
        .as_array_mut()
        .into_iter()
        .flatten()
        .filter_map(|m| m.get_mut("tool_calls"))
        .filter_map(Value::as_array_mut)
        .flatten()
    {
        let target = if call.get("function").is_some() {
            &mut call["function"]
        } else {
            call
        };
        if let Some(parsed) = target
            .get("arguments")
            .and_then(Value::as_str)
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            && parsed.is_object()
        {
            target["arguments"] = parsed;
        }
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn string_arguments_become_objects() {
        let msgs = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "g", "arguments": {"n": 1}}}]},
        ]);
        let out = with_object_arguments(&msgs);
        assert_eq!(
            out[1]["tool_calls"][0]["function"]["arguments"],
            json!({"city": "Paris"})
        );
        assert_eq!(
            out[2]["tool_calls"][0]["function"]["arguments"],
            json!({"n": 1})
        );
    }
}
