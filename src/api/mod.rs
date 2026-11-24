//! API adapters: OpenAI Chat/Completions/Responses, Anthropic Messages, llama-native.
//! Each module translates its wire format to/from the internal `GenRequest` / event stream.

pub mod anthropic;
pub mod internal;
pub mod llama_native;
pub mod openai_chat;
pub mod openai_completions;
pub mod openai_responses;
pub mod toolcall;
