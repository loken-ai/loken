//! The routing a measurement replays when its input stands in for a real stream: experts drawn
//! per token and layer from the heat the file or a profile gives each expert, and the guess a
//! layer's read-ahead makes of them.

/// A deterministic generator: the same seed gives the same routes on any machine, so two
/// records of the same controls drew the same experts.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 bits, from a splitmix step.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A number below `n`; `n` is at least 1.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n
    }
}

/// The heat of every expert from a prior that only orders them, hottest first: the weight of
/// rank `r` falls as `1 / (r + 1)`, the shape a routing count takes over the experts of a
/// layer. An expert the prior leaves out weighs as the rank after the last named.
pub fn heat_from_order(order: &[usize], n_experts: usize) -> Vec<u64> {
    // Integer weights so that the heat is exact and serialisable: the hottest expert weighs as
    // many units as there are experts, rank `r` that count over `r + 1`.
    let scale = n_experts as u64;
    let unranked = scale / (order.len() as u64 + 1);
    let mut heat = vec![unranked.max(1); n_experts];
    for (r, &e) in order.iter().enumerate() {
        if let Some(h) = heat.get_mut(e) {
            *h = (scale / (r as u64 + 1)).max(1);
        }
    }
    heat
}

/// `k` distinct experts drawn by `heat`, each draw weighted by the heat of the experts not yet
/// drawn. A heat of all zeros draws uniformly.
pub fn draw(heat: &[u64], k: usize, rng: &mut Lcg) -> Vec<usize> {
    let n = heat.len();
    let k = k.min(n);
    let uniform = heat.iter().all(|&h| h == 0);
    let mut left: Vec<usize> = (0..n).collect();
    let mut weight: Vec<u64> = if uniform { vec![1; n] } else { heat.to_vec() };
    let mut total: u64 = weight.iter().sum();
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        let mut pick = (rng.next_f64() * total as f64) as u64;
        let mut at = left.len() - 1;
        for (i, &e) in left.iter().enumerate() {
            let w = weight[e];
            if pick < w {
                at = i;
                break;
            }
            pick -= w;
        }
        let e = left.swap_remove(at);
        total -= weight[e];
        weight[e] = 0;
        out.push(e);
    }
    out
}

/// The guess a read-ahead makes of `route`: `hit` of its experts kept, rounded to the nearest
/// count, the rest replaced by experts the route does not name, drawn by `heat`.
pub fn guess(route: &[usize], heat: &[u64], hit: f64, rng: &mut Lcg) -> Vec<usize> {
    let k = route.len();
    let keep = ((hit.clamp(0.0, 1.0) * k as f64).round() as usize).min(k);
    let mut out: Vec<usize> = route[..keep].to_vec();
    // Draw from the experts off the route: their heat as it is, the route's set to zero.
    let mut off_route = heat.to_vec();
    for &e in route {
        if let Some(h) = off_route.get_mut(e) {
            *h = 0;
        }
    }
    let all_zero = off_route.iter().all(|&h| h == 0);
    if all_zero && heat.len() > k {
        // Every off-route expert is cold: draw them uniformly, the route excluded.
        for (e, h) in off_route.iter_mut().enumerate() {
            *h = u64::from(!route.contains(&e));
        }
    }
    for e in draw(&off_route, k - keep, rng) {
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

/// The routes and guesses of `tokens` decode steps through layers whose expert heat is given:
/// per layer, per token, `k` experts and the guess of them.
pub struct Script {
    pub routes: Vec<Vec<Vec<usize>>>,
    pub guesses: Vec<Vec<Vec<usize>>>,
}

pub fn script(heat: &[Vec<u64>], k: usize, tokens: usize, hit: f64, seed: u64) -> Script {
    let mut rng = Lcg::new(seed);
    let mut routes = Vec::with_capacity(heat.len());
    let mut guesses = Vec::with_capacity(heat.len());
    for layer in heat {
        let mut r = Vec::with_capacity(tokens);
        let mut g = Vec::with_capacity(tokens);
        for _ in 0..tokens {
            let route = draw(layer, k, &mut rng);
            g.push(guess(&route, layer, hit, &mut rng));
            r.push(route);
        }
        routes.push(r);
        guesses.push(g);
    }
    Script { routes, guesses }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distinct(v: &[usize]) -> bool {
        let s: std::collections::HashSet<usize> = v.iter().copied().collect();
        s.len() == v.len()
    }

    /// A draw names `k` distinct experts below the count, the same for the same seed.
    #[test]
    fn a_draw_is_distinct_within_range_and_repeatable() {
        let heat = heat_from_order(&[3, 1, 4], 8);
        let a = draw(&heat, 6, &mut Lcg::new(7));
        let b = draw(&heat, 6, &mut Lcg::new(7));
        assert_eq!(a, b);
        assert_eq!(a.len(), 6);
        assert!(distinct(&a));
        assert!(a.iter().all(|&e| e < 8));
        assert_eq!(draw(&heat, 12, &mut Lcg::new(1)).len(), 8);
    }

    /// The hottest expert of an ordered prior is drawn far more often than a cold one.
    #[test]
    fn the_heat_is_followed() {
        let heat = heat_from_order(&(0..64).collect::<Vec<_>>(), 64);
        assert_eq!(heat[0], 64);
        assert_eq!(heat[1], 32);
        assert_eq!(heat[63], 1);
        let mut rng = Lcg::new(3);
        let mut hot = 0;
        let mut cold = 0;
        for _ in 0..500 {
            let r = draw(&heat, 6, &mut rng);
            hot += usize::from(r.contains(&0));
            cold += usize::from(r.contains(&63));
        }
        assert!(hot > 4 * cold, "hot {hot}, cold {cold}");
        assert!(distinct(&draw(&[0; 10], 4, &mut rng)));
    }

    /// A guess keeps the asked share of the route and fills the rest off the route.
    #[test]
    fn a_guess_keeps_the_asked_share_of_the_route() {
        let heat = heat_from_order(&[], 32);
        let mut rng = Lcg::new(11);
        let route = draw(&heat, 6, &mut rng);
        let g = guess(&route, &heat, 1.0, &mut rng);
        assert_eq!(g, route);
        let g = guess(&route, &heat, 0.5, &mut rng);
        assert_eq!(g.len(), 6);
        assert!(distinct(&g));
        assert_eq!(g.iter().filter(|e| route.contains(e)).count(), 3);
        let g = guess(&route, &heat, 0.0, &mut rng);
        assert!(g.iter().all(|e| !route.contains(e)));
    }

    /// A script has one route and one guess per layer and token.
    #[test]
    fn a_script_covers_every_layer_and_token() {
        let heat = vec![heat_from_order(&[1, 2], 16); 3];
        let s = script(&heat, 4, 5, 0.75, 9);
        assert_eq!(s.routes.len(), 3);
        assert_eq!(s.guesses.len(), 3);
        for (r, g) in s.routes.iter().zip(&s.guesses) {
            assert_eq!(r.len(), 5);
            assert_eq!(g.len(), 5);
            for (route, guess) in r.iter().zip(g) {
                assert_eq!(route.len(), 4);
                assert_eq!(guess.len(), 4);
                assert_eq!(guess.iter().filter(|e| route.contains(e)).count(), 3);
            }
        }
    }
}
