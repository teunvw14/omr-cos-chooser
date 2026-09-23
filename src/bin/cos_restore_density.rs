use std::{error::Error, f64::consts::PI};

use num_complex::Complex;

use cos_chooser::util::{linspace};

const i: Complex<f64> = Complex::new(0.0, 1.0);

/// Restore the distribution of a random variable X from it's characteristic
/// function.
/// # Parameters
/// - char_fun: the characteristic function for the distribution of the underlying at final time
/// - n: number of terms in the fourier approximation
/// - a: left integration bound
/// - b: right integration bound
/// - h: integration precision
fn cos_restore_distr<FunComplex>(    
    char_fun: FunComplex,
    n: u64,
    a: f64,
    b: f64,
    y: f64,
) -> f64
where FunComplex: Fn(f64) -> Complex<f64>,
{
    let mut result = 0.0;

    // First term multiplied by 1/2, so calculate "manually"
    let F_0 = (2.0 / (b - a)) * (char_fun(0.0)).re;
    result += 0.5 * F_0;

    for k in 1..n {
        let F_k = (2.0 / (b - a)) * (char_fun(k as f64 * PI / (b-a)) * Complex::exp(-i * k as f64 * a * PI / (b-a))).re;
        let freq_contribution = F_k * f64::cos(k as f64 * PI * (y-a)/(b-a));
        result += freq_contribution;
    }

    result
}

fn main() -> Result<(), Box<dyn Error>> {
// Let's try to reconstruct the Normal pdf from the characteristic function.

    let norm_char_fun = |u: f64| Complex::new(-u*u/2.0, 0.0).exp();
    let n = 256;
    let a = -10.0;
    let b = 10.0;

    let y = 0.0;
    // let pdf_norm = cos_restore_distr(norm_char_fun, n, a, b, h, y);
    // println!("f_N({y}) = {pdf_norm}");
    let steps = 100;
    let xs = linspace(-3.0, 3.0, steps);
    let mut ys = Vec::with_capacity(steps);
    for x in &xs {
        ys.push(cos_restore_distr(norm_char_fun, n, a, b, *x));
    }
    let xs_len = xs.len();
    let ys_len = ys.len();

    // Create heatmap chart
    // chart!(xs, ys)?
    //     .mark_area()?
    //     .configure_area(|a| a.with_color("blue").with_opacity(0.3))
    //     .encode((alt::x("xs"), alt::y("ys")))?
    //     .with_x_label("x")
    //     .with_y_label("f_N(x)")
    //     .save("out.png")?;

    Ok(())
}
