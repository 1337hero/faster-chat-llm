//! Host-side token sampling: top-k → top-p → min-p → temperature, then a seeded draw.
//! Temperature 0 is greedy and never reaches this module (the argmax runs on the card).

#[derive(Clone, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

impl SampleParams {
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

/// SplitMix64: tiny, seedable, good enough for sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn sample(logits: &[f32], p: &SampleParams, rng: &mut Rng) -> u32 {
    let k = if p.top_k == 0 { logits.len() } else { p.top_k.min(logits.len()) };
    let mut cand: Vec<(u32, f32)> = logits.iter().copied().enumerate().map(|(i, v)| (i as u32, v)).collect();
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(k);
    }
    cand.sort_by(|a, b| b.1.total_cmp(&a.1));

    let softmax = |c: &[(u32, f32)], t: f32| -> Vec<f64> {
        let max = c[0].1 as f64;
        let w: Vec<f64> = c.iter().map(|&(_, v)| ((v as f64 - max) / t as f64).exp()).collect();
        let s: f64 = w.iter().sum();
        w.into_iter().map(|x| x / s).collect()
    };
    let probs = softmax(&cand, 1.0);
    let mut keep = cand.len();
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, pr) in probs.iter().enumerate() {
            cum += pr;
            if cum >= p.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
    }
    if p.min_p > 0.0 {
        let floor = probs[0] * p.min_p as f64;
        keep = keep.min(probs.iter().take_while(|&&pr| pr >= floor).count().max(1));
    }
    cand.truncate(keep);

    let probs = softmax(&cand, p.temperature);
    let r = rng.next_f64();
    let mut cum = 0.0;
    for (c, pr) in cand.iter().zip(&probs) {
        cum += pr;
        if r < cum {
            return c.0;
        }
    }
    cand[cand.len() - 1].0
}
