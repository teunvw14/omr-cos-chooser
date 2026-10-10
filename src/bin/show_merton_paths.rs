use std::error::Error;

use ndarray::{Array2, s};
use plotters::prelude::*;

use cos_chooser::util::{chart_bounds_y, gbm_generate_paths, linspace, merton_generate_paths};


fn main() -> Result<(), Box<dyn Error>>{
    let S_0 = 100.0;
    let r = 0.02;
    let sigma = 0.2;
    let tau = 1.0;

    let jump_rate = 1.0;
    let jump_mean = 0.0;
    let jump_std = 0.2;
    // Drift adjusted so that the mean of the distribution is equal to that of a jump-less distribution
    let adjusted_mu = r - 0.5*sigma*sigma - jump_rate * (f64::exp(0.5 * jump_std*jump_std + jump_mean) - 1.0);
    
    let n_paths = 100_000;
    let n_steps = 100;
    let ts = linspace(0.0, tau, n_steps);

    let paths_gbm: Array2<f64> = gbm_generate_paths(S_0, r, sigma, tau, n_paths, n_steps);
    let mut paths: Array2<f64> = merton_generate_paths(S_0, adjusted_mu, sigma, jump_rate, jump_mean, jump_std, tau, n_paths, n_steps);
    
    let gbm_mean = paths_gbm.slice(s![.., -1]).mean().unwrap();
    let merton_mean = paths.slice(s![.., -1]).mean().unwrap();
    let gbm_std = paths_gbm.slice(s![.., -1]).std(0.0);
    let merton_std = paths.slice(s![.., -1]).std(0.0);
    let mean_diff = f64::abs(gbm_mean - merton_mean);
    let std_diff = f64::abs(gbm_std - merton_std);
    
    println!("GBM mean: {gbm_mean}, Merton mean: {merton_mean}, diff: {mean_diff}");
    println!("GBM var: {gbm_std}, Merton var: {merton_std}, diff: {std_diff}");

    // Plot

    // let (y_min, y_max) = chart_bounds_y(paths.view(), 0.1);
    // let h = 2000;
    // let w = 3000;
    // let margin = (0.05 * w as f64) as i32;
    // let path = "img/merton.svg".to_string();
    // let root = SVGBackend::new(&path, (w, h)).into_drawing_area();
    // root.fill(&WHITE)?;
    // let mut chart = ChartBuilder::on(&root)
    //     .margin(margin)
    //     // Set the caption of the chart
    //     .caption("Merton Jump Process Plot", ("sans-serif", 40).into_font())
    //     // Set the size of the label region
    //     .x_label_area_size(20)
    //     .y_label_area_size(40)
    //     // Finally attach a coordinate on the drawing area and make a chart context
    //     .build_cartesian_2d(0f64..tau, y_min..y_max)?;

    // chart.configure_mesh().draw()?;

    // for (p, path) in paths.rows().into_iter().enumerate() {
    //     // let path_as_vec: Vec<f64> = path.iter().map(|x| *x).collect();
    //     let series = ts.iter().zip(path.iter()).map(|(a, b)| (a.clone(), b.clone()));
    //     chart.draw_series(LineSeries::new(series, Palette99::pick(p).stroke_width(2)))?;
    // }

    // root.present()?;
    // opener::open(&path)?;

    Ok(())
}
