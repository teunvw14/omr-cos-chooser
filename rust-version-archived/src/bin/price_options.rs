use std::cmp::max;
use std::{f64::consts::PI, time::Instant};
use std::error::Error;

use ndarray::{Array2, s};
use num_complex::Complex;
use integrate::newton_cotes::{newton_rule};
use rayon::prelude::*;use plotters::prelude::*;

// let colors = vec![
//     "#1D6B34", 
//     "#01403A", 
//     "#03A688",
//     "#02735E", 
//     "#03A678",
// ];


use cos_chooser::util::{bs_price_call, chart_bounds_y, gbm_generate_paths, linspace, mc_price, merton_generate_paths};

const i: Complex<f64> = Complex::new(0.0, 1.0);


/// Calculate the price of a derivative with `payoff_fn` as payoff at final
/// time.
///
/// # Parameters
/// - payoff_fn: the payoff function for the derivative, as a function of the underlying
/// - char_fun: the characteristic function for the distribution of the underlying at final time
/// - r: risk-free rate
/// - tau: time until payoff
/// - n: number of terms in the fourier approximation
/// - a: left integration bound
/// - b: right integration bound
/// - h: integration precision
fn cos_price_derivative<Fun, FunComplex>(
    payoff_fn: Fun,
    char_fun: FunComplex,
    r: f64,
    tau: f64,
    n: u64,
    a: f64,
    b: f64,
    h: f64,
) -> f64
where
    Fun: Fn(f64) -> f64 + Sync,
    FunComplex: Fn(f64) -> Complex<f64> + Sync,
{
    let mut result = 0.0;
    let n_intervals = f64::ceil((b - a) / h) as u64;

    // First term multiplied by 1/2, so calculate "manually"
    let C_0 = (char_fun(0.0)).re;
    let H_0 = (2.0 / (b - a)) * newton_rule(&payoff_fn, a, b, n_intervals);

    result += 0.5 * C_0 * H_0;
    // println!("first term: {result} | C_0: {C_0} H_0: {H_0}");

    // The somewhat unnatural looking loop is used to allow for parallel
    // processing with Rayon (using `into_par_iter`).
    result += (1..n).into_par_iter().map(|k|{
        let C_k =
            (char_fun(k as f64 * PI / (b - a)) * Complex::exp(-i * k as f64 * a * PI / (b - a))).re;
        let integrand = |s| payoff_fn(s) * f64::cos(k as f64 * PI * (s - a) / (b - a));
        let H_k = (2.0 / (b - a)) * newton_rule(integrand, a, b, n_intervals);
        C_k * H_k
    }).reduce(|| 0.0, |a, b| a + b);

    let discount = f64::exp(-r * tau);
    
    discount * result
}

fn main() -> Result<(), Box<dyn Error>> {
    // We calculate the price of a European Call, with the underlying S(t)
    // following a GBM.

    let K = 100.0;
    let S_0 = 100.0;
    let r = 0.02;
    let sigma = 0.2;
    let tau = 1.0;
    let jump_rate = 1.0;
    let jump_mean = 0.0;
    let jump_std = 0.1;

    // We now use the COS method
    // We use X := ln(S(t)/K). 
    
    fn call_payoff(k: f64) -> impl Fn(f64) -> f64 {
        move |s: f64| (s - k).max(0.0)
    }

    fn call_payoff_ln(k: f64) -> impl Fn(f64) -> f64 {
        move |x: f64| (f64::exp(x) - k).max(0.0)
    }

    // Payoff for the variable transform X(t) = ln(S(t)/K), so S(t) = K * e^X
    fn call_payoff_ln_s_k(k: f64) -> impl Fn(f64) -> f64 {
        move |x: f64| (k * f64::exp(x) - k).max(0.0)
    }

    fn straddle_payoff_ln(k: f64) -> impl Fn(f64) -> f64 {
        move |x: f64| (f64::exp(x) - k).abs()
    }

    let mu_gbm = r - 0.5*sigma*sigma;
    let sigma = sigma;
    let gbm_char_fun = |u| Complex::exp(i * u * mu_gbm * tau - 0.5*sigma*sigma * u*u * tau);
    let mu_merton = r - 0.5*sigma*sigma - jump_rate * (f64::exp(0.5*jump_std*jump_std + jump_mean) - 1.0);
    let merton_char_fun = |u| Complex::exp(i * u * mu_merton * tau - 0.5*sigma*sigma * u*u * tau) * Complex::exp(jump_rate*tau*(Complex::exp(i * jump_mean * u - 0.5 * jump_std*jump_std * u*u) - 1.0));

    // Note that [a, b] is the area over which you integrate of ln(S(t))
    let n = 256;
    let a_exp: f64 = 0.1;
    let a = a_exp.ln();
    let b_exp: f64 = 1_000.0;
    let b = b_exp.ln();
    let h = 0.1;
    
    // analytic
    let analytical_value = bs_price_call(K, S_0, r, sigma, tau);
    println!("Analytical option value: {analytical_value:.4}");
    
    // cos method (GBM)
    let start = Instant::now();
    let price = cos_price_derivative(call_payoff_ln_s_k(K), gbm_char_fun, r, tau, n, a, b, h);
    let duration = start.elapsed();
    println!("COS method Call option value (GBM): {price:.4} (multi core {:?}s)", duration);
    
    // Monte Carlo pricing (GBM)
    let n_paths = 10_000_000;
    let n_steps = 25;
    let start = Instant::now();
    let paths = gbm_generate_paths(S_0, r, sigma, tau, n_paths, n_steps);
    let gbm_std = paths.slice(s![.., -1]).std(0.0);
    let price = mc_price(call_payoff(K), paths.view(), r, tau);
    let duration = start.elapsed();
    println!("MC method Call option value (GBM): {price:.4} (multi core {:?}s)", duration);

    // TODO: this should be a test
    // let paths_final_values = paths.slice(s![.., -1]);
    // let paths_mean = paths_final_values.ln().mean().unwrap();
    // let paths_var = paths_final_values.ln().var(0.0);
    // let paths_mean_ex = S_0.ln() + (r - 0.5 * sigma * sigma) * tau;
    // let paths_var_ex = sigma * sigma;
    
    // println!("Paths mean: {paths_mean} (expected {paths_mean_ex}), paths var: {paths_var} (expected {paths_var_ex})");

    // Monte Carlo pricing (Merton)
    let n_paths = 1_000_000;
    let n_steps = 250; // There seems to be some sort of discretization error in step size!
    let start = Instant::now();
    let adjusted_mu = r - 0.5*sigma*sigma - jump_rate * (f64::exp(0.5 * jump_std*jump_std + jump_mean) - 1.0);
    let mut paths = merton_generate_paths(S_0, adjusted_mu, sigma, jump_rate, jump_mean, jump_std, tau, n_paths, n_steps);
    let merton_std = paths.slice(s![.., -1]).std(0.0);
    let merton_mean = paths.slice(s![.., -1]).mean().unwrap();
    let price_mc = mc_price(call_payoff(K), paths.view(), r, tau);
    let duration = start.elapsed();
    println!("MC method Call option value (Merton): {price_mc:.4} (multi core {:?}s) ", duration);

    // Monte Carlo Pricing (GBM-Moment-Matched-Merton)
    paths = (gbm_std/merton_std) * (paths - merton_mean) + merton_mean;
    let price = mc_price(call_payoff(K), paths.view(), r, tau);
    println!("MC method Call option value (Moment-Matched-Merton): {price:.4}");

    // cos method (Merton)
    let start = Instant::now();
    let price = cos_price_derivative(call_payoff_ln_s_k(K), merton_char_fun, r, tau, n, a, b, h);
    let duration = start.elapsed();
    println!("COS method Call option value (Merton): {price:.4} (multi core {:?}s)", duration);
    let ratio = 100.0 * f64::abs(1.0 - price_mc / price);
    println!("MC vs COS diff (Merton): {ratio:.2}%");

    /////////////////
    // Plots
    /////////////////

    let steps = 100;
    let (x_min, x_max) = (50.0, 150.0);
    let xs = linspace(x_min, x_max, steps);
    let ys: Vec<f64> = xs.iter().map(|x| cos_price_derivative(call_payoff_ln(*x), gbm_char_fun, r, tau, n, a, b, h)).collect();
    let ys_theo: Vec<f64> = xs.iter().map(|x| bs_price_call(*x, S_0, r, sigma, tau)).collect();

    let errs: Vec<f64> = ys.iter()
        .zip(&ys_theo)
        .map(|(y, y_theo)| (y-y_theo).abs())
        .collect();

    // Plot COS method values vs theoretical values
    let (y_min, y_max) = chart_bounds_y(&ys, 0.1);
    let h = 600;
    let w = 800;
    let margin = (0.05 * w as f64) as i32;
    let path = "img/cos-vs-theo.svg".to_string();
    let root = SVGBackend::new(&path, (w, h)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(margin)
        // Set the caption of the chart
        .caption("COS method value vs theoretical", ("sans-serif", 40).into_font())
        // Set the size of the label region
        .x_label_area_size(20)
        .y_label_area_size(40)
        // Finally attach a coordinate on the drawing area and make a chart context
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("K (strike)")
        .y_desc("Option Value")
        .draw()?;

    let prices_cos = xs.iter().zip(&ys).map(|(a, b)| (a.clone(), b.clone()));
    let prices_theo = xs.iter().zip(&ys).map(|(a, b)| (a.clone(), b.clone()));
    chart.draw_series(DashedLineSeries::new(prices_cos, 10, 2, RED.stroke_width(5)))?;
    chart.draw_series(DashedLineSeries::new(prices_theo, 11, 3, BLUE.stroke_width(5)))?;
    root.present()?;

    // Plot Error
    let (y_min, y_max) = chart_bounds_y(&errs, 0.1);
    let h = 600;
    let w = 800;
    let margin = (0.05 * w as f64) as i32;
    let path = "img/cos-errs.svg".to_string();
    let root = SVGBackend::new(&path, (w, h)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(margin)
        // Set the caption of the chart
        .caption("COS method errors", ("sans-serif", 40).into_font())
        // Set the size of the label region
        .x_label_area_size(20)
        .y_label_area_size(40)
        // Finally attach a coordinate on the drawing area and make a chart context
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("K (strike)")
        .y_desc("Absolute Error")
        .draw()?;

    let errs_series = xs.iter().zip(&errs).map(|(a, b)| (a.clone(), b.clone()));
    chart.draw_series(LineSeries::new(errs_series, RED.stroke_width(3)))?;
    root.present()?;
    
    Ok(())
}
