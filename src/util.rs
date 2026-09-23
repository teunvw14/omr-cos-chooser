use statrs::distribution::{ContinuousCDF, Normal, Uniform};
use rand::distr::Distribution;
use ndarray::{Array1, Array2, ArrayView2, s};


pub fn bs_price_call(K: f64, S_0: f64, r: f64, sigma: f64, tau: f64) -> f64 {
    let d1 = (f64::ln(S_0 / K) + (r + sigma * sigma / 2.0) * tau) / (sigma * f64::sqrt(tau));
    let d2 = d1 - sigma * f64::sqrt(tau);
    let norm = Normal::new(0.0, 1.0).unwrap();
    S_0 * norm.cdf(d1) - K * f64::exp(-r * tau) * norm.cdf(d2)
}

pub fn bs_price_put(K: f64, S_0: f64, r: f64, sigma: f64, tau: f64) -> f64 {
    let d1 = (f64::ln(S_0 / K) + (r + sigma * sigma / 2.0) * tau) / (sigma * f64::sqrt(tau));
    let d2 = d1 - sigma * f64::sqrt(tau);
    let norm = Normal::new(0.0, 1.0).unwrap();
    K * f64::exp(-r * tau) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
}

/// Price a derivative with a payoff. `Paths` is assumed to represent each path
/// as a row.
pub fn mc_price<Fun>(
    payoff: Fun,
    paths: ArrayView2<f64>,
    r: f64,
    tau: f64,
) -> f64
where Fun: Fn(f64) -> f64,
{
    let final_values = paths.slice(s![.., -1]);
    let num_paths = final_values.len() as f64;
    let payoffs = final_values.map(|x| payoff(*x));
    let discount = f64::exp(-r * tau);
    discount * (payoffs / num_paths).sum()
}

pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n);

    let step = (end - start) / (n-1) as f64;
    for n in 0..n {
        result.push(start + n as f64 * step);
    }

    result
}

/// Generate paths following a Geometric Brownian Motion (GBM), i.e.
/// $dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$, where $W$ is a Wiener process.
/// 
/// ## Parameters
/// - `S_0`: Starting value for all paths
/// - `mu`: the process drift
/// - `sigma`: the process "standard deviation"
/// - `tau`: running time
/// - `n_paths`: number of paths to simulate
/// 
/// ## Return Value
/// Returns a 2d matrix, where each row is a path. 
pub fn gbm_generate_paths(S_0: f64, mu: f64, sigma: f64, tau: f64, n_paths: usize, n_steps: usize) -> Array2<f64> {
    let mut X: Array2<f64> = Array2::zeros((n_paths, n_steps+1));
    X.slice_mut(s![.., 0]).fill(S_0.ln());

    // n_steps - 1 because we count t_0 as the first "step"
    let dt = tau/n_steps as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    for p in 0..n_paths {
        for i in 0..n_steps {
            X[(p, i+1)] = X[(p, i)] + (mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * norm.sample(&mut rng);
        }
    }

    X.exp()
}

pub fn merton_generate_paths(S_0: f64, mu: f64, sigma: f64, jump_rate: f64, jump_mean: f64, jump_std: f64, tau: f64, n_paths: usize, n_steps: usize) -> Array2<f64> {
    let mut X: Array2<f64> = Array2::zeros((n_paths, n_steps));
    X.slice_mut(s![.., 0]).fill(S_0.ln());

    // n_steps - 1 because we count t_0 as the first "step"
    let dt = tau/(n_steps-1) as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();
    let norm_jump = Normal::new(jump_mean, jump_std).unwrap();
    let unif = Uniform::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    for p in 0..n_paths {
        for i in 1..n_steps {
            let jumps = if unif.sample(&mut rng) < dt * jump_rate {
                1.0
            } else { 0.0 };
            X[(p, i)] = X[(p, i-1)] + mu * dt + sigma * dt.sqrt() * norm.sample(&mut rng) + jumps * norm_jump.sample(&mut rng);
        }
    }

    X.exp()
}

/// Calculates the vertical y bounds given some margin around the min and max
pub fn chart_bounds_y<'a, I>(ys: I, margin: f64) -> (f64, f64) 
where I: IntoIterator<Item = &'a f64>
{
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    for elem in ys {
        if *elem > max {
            max = *elem;
        }
        if *elem < min {
            min = *elem;
        }
    }
    
    let delta = max - min;
    let margin = margin * delta;
    let min_adjusted = min - margin;
    let max_adjusted = max + margin;
    
    (min_adjusted, max_adjusted)
}
