use std::error::Error;

use ndarray::Array2;
use plotters::prelude::*;

use cos_chooser::util::{linspace, gbm_generate_paths, chart_bounds_y};


fn main() -> Result<(), Box<dyn Error>>{
    let S_0 = 100.0;
    let r = 0.02;
    let sigma = 0.2;
    let tau = 1.0;
    
    let n_paths = 1_000;
    let n_steps = 100;
    let ts = linspace(0.0, tau, n_steps);

    let paths: Array2<f64> = gbm_generate_paths(S_0, r, sigma, tau, n_paths, n_steps);

    let (y_min, y_max) = chart_bounds_y(paths.view(), 0.1);
    let h = 2000;
    let w = 3000;
    let margin = (0.05 * w as f64) as i32;
    let path = "img/mc.svg".to_string();
    let root = SVGBackend::new(&path, (w, h)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(margin)
        // Set the caption of the chart
        .caption("Monte Carlo Plot", ("sans-serif", 40).into_font())
        // Set the size of the label region
        .x_label_area_size(20)
        .y_label_area_size(40)
        // Finally attach a coordinate on the drawing area and make a chart context
        .build_cartesian_2d(0f64..tau, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    for (p, path) in paths.rows().into_iter().enumerate() {
        // let path_as_vec: Vec<f64> = path.iter().map(|x| *x).collect();
        let series = ts.iter().zip(path.iter()).map(|(a, b)| (a.clone(), b.clone()));
        chart.draw_series(LineSeries::new(series, Palette99::pick(p).stroke_width(2)))?;
    }

    root.present()?;
    opener::open(&path)?;

    Ok(())
}
