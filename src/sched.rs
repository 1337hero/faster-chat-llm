//! The driver: owns the engine, multiplexes requests onto sequence slots, runs the
//! draft → verify → accept loop, and streams text back. All GPU work happens on this thread.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use crate::engine::Engine;
use crate::hip::BMAX;
use crate::sampler::{Rng, SampleParams, sample};
use crate::tokenizer::Tokenizer;

pub struct Job {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SampleParams,
    pub stop: Vec<String>,
    pub tx: Sender<Event>,
}

pub enum Event {
    Text(String),
    Done(Done),
}

pub struct Done {
    pub finish_reason: &'static str,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub drafted: usize,
    pub accepted: usize,
}

struct Active {
    job: Job,
    /// Prompt tokens already in the engine.
    fed: usize,
    cached_tokens: usize,
    /// Sampled but not yet run through the trunk.
    last: Option<u32>,
    n_gen: usize,
    rng: Rng,
    /// Bytes decoded but not yet sent (incomplete UTF-8, or held back for stop matching).
    pending: Vec<u8>,
    text: String,
    sent: usize,
    t0: Instant,
    t_first: Option<Instant>,
    drafted: usize,
    accepted: usize,
}

#[derive(Default)]
struct Slot {
    /// Tokens whose effects are in the engine state for this slot.
    cached: Vec<u32>,
    active: Option<Active>,
}

pub struct Config {
    pub draft_n: usize,
    pub ctx: usize,
}

pub struct Driver {
    eng: Engine,
    tok: Tokenizer,
    slots: Vec<Slot>,
    cfg: Config,
    stop_ids: Vec<u32>,
    prefill_turn: bool,
}

impl Driver {
    pub fn new(eng: Engine, tok: Tokenizer, n_slots: usize, cfg: Config) -> Self {
        let stop_ids = vec![tok.eos, tok.bos];
        Self { eng, tok, slots: (0..n_slots).map(|_| Slot::default()).collect(), cfg, stop_ids, prefill_turn: false }
    }

    pub fn run(mut self, rx: Receiver<Job>) {
        let mut queue: Vec<Job> = Vec::new();
        loop {
            if queue.is_empty() && self.slots.iter().all(|s| s.active.is_none()) {
                match rx.recv() {
                    Ok(j) => queue.push(j),
                    Err(_) => return,
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(j) => queue.push(j),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }
            while !queue.is_empty() && self.slots.iter().any(|s| s.active.is_none()) {
                let job = queue.remove(0);
                self.admit(job);
            }
            self.tick();
        }
    }

    /// Place a job on an idle slot, reusing the slot's state when it is a prefix of the prompt.
    fn admit(&mut self, job: Job) {
        let used: usize = self.slots.iter().map(|s| s.cached.len()).sum();
        let reusable = |s: &Slot| s.cached.len() < job.prompt.len() && job.prompt.starts_with(&s.cached);
        let idle = || self.slots.iter().enumerate().filter(|(_, s)| s.active.is_none());
        // Longest reusable prefix first; otherwise evict the idle slot holding the least.
        let pick = idle().filter(|(_, s)| reusable(s)).max_by_key(|(_, s)| s.cached.len()).or_else(|| idle().min_by_key(|(_, s)| s.cached.len()));
        let (i, slot) = pick.expect("admit called without an idle slot");
        let keep = if reusable(slot) { slot.cached.len() } else { 0 };
        if job.prompt.is_empty() || used - slot.cached.len() + job.prompt.len() + 1 > self.cfg.ctx {
            let _ = job.tx.send(Event::Done(Done { finish_reason: "length", prompt_tokens: job.prompt.len(), cached_tokens: 0, completion_tokens: 0, prefill_s: 0.0, decode_s: 0.0, drafted: 0, accepted: 0 }));
            return;
        }
        if keep == 0 {
            self.eng.reset(i);
            self.slots[i].cached.clear();
        }
        let rng = Rng::new(job.sampling.seed);
        self.slots[i].active = Some(Active { job, fed: keep, cached_tokens: keep, last: None, n_gen: 0, rng, pending: Vec::new(), text: String::new(), sent: 0, t0: Instant::now(), t_first: None, drafted: 0, accepted: 0 });
    }

    /// One engine step: a prefill chunk for one slot, or a decode step for all decoding slots.
    fn tick(&mut self) {
        let prefilling: Vec<usize> = (0..self.slots.len()).filter(|&i| self.slots[i].active.as_ref().is_some_and(|a| a.last.is_none())).collect();
        let decoding: Vec<usize> = (0..self.slots.len()).filter(|&i| self.slots[i].active.as_ref().is_some_and(|a| a.last.is_some())).collect();
        // Alternate so a long prefill cannot starve the other slot's decode.
        let do_prefill = !prefilling.is_empty() && (decoding.is_empty() || self.prefill_turn);
        self.prefill_turn = !self.prefill_turn;
        if do_prefill {
            self.prefill(prefilling[0]);
        } else if !decoding.is_empty() {
            self.decode(&decoding);
        }
    }

    fn prefill(&mut self, i: usize) {
        let a = self.slots[i].active.as_ref().unwrap();
        let chunk: Vec<u32> = a.job.prompt[a.fed..(a.fed + BMAX).min(a.job.prompt.len())].to_vec();
        let out = self.eng.step(&[(i, &chunk, true)]);
        let slot = &mut self.slots[i];
        slot.cached.extend(&chunk);
        let a = slot.active.as_mut().unwrap();
        a.fed += chunk.len();
        if a.fed == a.job.prompt.len() {
            let row = chunk.len() - 1;
            let tok = if a.job.sampling.greedy() { out[row].token } else { sample(&self.eng.logits(row), &a.job.sampling, &mut a.rng) };
            a.t_first = Some(Instant::now());
            self.emit(i, &[tok], 0);
        }
    }

    fn decode(&mut self, slots: &[usize]) {
        let lasts: Vec<(usize, u32)> = slots.iter().map(|&i| (i, self.slots[i].active.as_ref().unwrap().last.unwrap())).collect();
        let n_draft = self.cfg.draft_n.min(BMAX / slots.len() - 1);
        let drafts = if n_draft > 0 { self.eng.draft(&lasts, n_draft) } else { vec![Vec::new(); slots.len()] };
        let rows: Vec<Vec<u32>> = lasts.iter().zip(&drafts).map(|(&(_, last), d)| std::iter::once(last).chain(d.iter().copied()).collect()).collect();
        let groups: Vec<(usize, &[u32], bool)> = slots.iter().zip(&rows).map(|(&i, r)| (i, r.as_slice(), false)).collect();
        let out = self.eng.step(&groups);

        let mut row0 = 0;
        for ((&i, r), d) in slots.iter().zip(&rows).zip(&drafts) {
            let a = self.slots[i].active.as_mut().unwrap();
            // Row j's target token is the model's own choice given the accepted prefix; a draft
            // survives only if it equals that choice, so the output never depends on the drafter.
            let mut toks = Vec::new();
            for j in 0..r.len() {
                let t = if a.job.sampling.greedy() { out[row0 + j].token } else { sample(&self.eng.logits(row0 + j), &a.job.sampling, &mut a.rng) };
                toks.push(t);
                if j >= d.len() || d[j] != t {
                    break;
                }
            }
            a.drafted += d.len();
            let n = self.emit(i, &toks, r.len());
            if let Some(a) = self.slots[i].active.as_mut() {
                a.accepted += n - 1;
            }
            row0 += r.len();
        }
    }

    /// Take `toks` (targets of consecutive rows of the last step) as output for slot `i`: accept
    /// their rows, stream the text, finish on a stop condition. `step_rows` is 0 after a commit
    /// step. Returns how many tokens were consumed.
    fn emit(&mut self, i: usize, toks: &[u32], step_rows: usize) -> usize {
        let slot = &mut self.slots[i];
        let a = slot.active.as_mut().unwrap();
        let prev_last = a.last;
        let mut finish = None;
        let mut used = 0;
        for &t in toks {
            used += 1;
            if self.stop_ids.contains(&t) {
                finish = Some("stop");
                break;
            }
            a.n_gen += 1;
            if self.tok.token_types.get(t as usize).copied() != Some(3) {
                a.pending.extend_from_slice(self.tok.decode_id(t));
            }
            a.last = Some(t);
            if a.n_gen >= a.job.max_tokens || slot.cached.len() + used + 1 >= self.cfg.ctx {
                finish = Some("length");
                break;
            }
        }
        // The rows whose targets we consumed are now part of the sequence: their inputs were
        // the previous last token followed by the tokens just taken (all but the newest).
        if step_rows > 0 {
            self.eng.accept(i, used);
            slot.cached.push(prev_last.unwrap());
            slot.cached.extend(&toks[..used - 1]);
        }
        // Move complete UTF-8 into the text buffer.
        let valid = match std::str::from_utf8(&a.pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to() + e.error_len().unwrap_or(0),
        };
        a.text.push_str(&String::from_utf8_lossy(&a.pending[..valid]));
        a.pending.drain(..valid);

        let mut ok = true;
        // Unsent text always covers the last `hold` bytes, so a stop string can only start there.
        let hold = a.job.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        if let Some(pos) = a.job.stop.iter().filter_map(|s| a.text[a.sent..].find(s.as_str())).min() {
            a.text.truncate(a.sent + pos);
            finish = Some("stop");
        }
        let mut upto = if finish.is_some() { a.text.len() } else { a.text.len().saturating_sub(hold).max(a.sent) };
        while !a.text.is_char_boundary(upto) {
            upto -= 1;
        }
        if upto > a.sent {
            ok = a.job.tx.send(Event::Text(a.text[a.sent..upto].to_string())).is_ok();
            a.sent = upto;
        }
        if finish.is_some() || !ok {
            let a = slot.active.take().unwrap();
            let t_first = a.t_first.unwrap_or_else(Instant::now);
            let _ = a.job.tx.send(Event::Done(Done {
                finish_reason: finish.unwrap_or("abort"),
                prompt_tokens: a.job.prompt.len(),
                cached_tokens: a.cached_tokens,
                completion_tokens: a.n_gen,
                prefill_s: (t_first - a.t0).as_secs_f64(),
                decode_s: t_first.elapsed().as_secs_f64(),
                drafted: a.drafted,
                accepted: a.accepted,
            }));
        }
        used
    }
}
