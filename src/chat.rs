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
        let syntax = minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().map_err(|e| e.to_string())?;
        env.set_syntax(syntax);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", source).map_err(|e| format!("chat template: {e}"))?;
        Ok(Self { env })
    }

    /// Render `messages` with a generation prompt. `kwargs` are extra template variables
    /// (e.g. `enable_thinking`); `tools` is passed through untouched.
    pub fn render(&self, messages: &Value, tools: Option<&Value>, kwargs: &serde_json::Map<String, Value>) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        use minijinja::value::Serde;
        let base = context! { messages => Serde(messages), tools => Serde(tools), add_generation_prompt => true };
        let extra = minijinja::Value::from(Serde(kwargs));
        tmpl.render(context! { ..extra, ..base }).map_err(|e| format!("chat template: {e:#}"))
    }
}
