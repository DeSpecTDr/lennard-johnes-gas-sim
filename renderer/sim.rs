use core::f32;
use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use egui_plot::{PlotPoint, PlotPoints};
use itertools::izip;
use rayon::prelude::*;

const CUTOFF: f32 = 4.;

struct PhysParams {
    sigma: f32,
    epsilon: f32,
    mass: f32,
    k_bolz: f32,
}

// Water
// const MODEL: PhysParams = PhysParams {
//     sigma: 2.725e-10,
//     epsilon: 4.9115e-21,
//     mass: 30.103e-27,
//     k_bolz: 1.380649e-23,
// };

const AVOGADRO: f32 = 6.0221408e23;

// Argon
const MODEL: PhysParams = PhysParams {
    sigma: 3.405e-10,
    epsilon: 1.6537e-21,
    mass: 39.948 * 0.001 / AVOGADRO,
    k_bolz: 1.380649e-23,
};

const N_BARS: usize = 20;
const BARS_S: f64 = 20.0;

// fn ui_system(
//     sim: Res<Sim>,
//     mut contexts: EguiContexts,
//     diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
//     mut stats: ResMut<Stats>,
//     state: Res<SimState>,
//     time: Res<Time<Real>>,
// ) {
//     // let dt = time.delta_secs();
//     // let fps = if dt == 0.0 { 0.0 } else { 1.0 / time.delta_secs() };

//     let fps = diagnostics
//         .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
//         .and_then(|f| f.average())
//         .unwrap_or(0.);

//     fn make_rolling(
//         v: &VecDeque<Stat>,
//         f: impl Fn(&Stat) -> f32,
//         div0: bool,
//     ) -> Vec<PlotPoint> {
//         // let n = 2;
//         // let mut s: f64 = v.iter().take(n).map(|x| f(x) as f64).sum();
//         // let s0 = s;
//         // let t = v.get(n).or_else(|| v.back()).map(|x| x.t).unwrap_or(0.0);
//         // std::iter::once(PlotPoint::new(
//         //     t,
//         //     if div0 { s / s0 } else { s / n as f64 },
//         // ))
//         // .chain(v.iter().zip(v.iter().skip(n)).map(|(la, ne)| {
//         //     let last = f(la) as f64;
//         //     let next = f(ne) as f64;
//         //     s = s - last + next;
//         //     PlotPoint::new(ne.t, if div0 { s / s0 } else { s / n as f64 })
//         // }))
//         // .collect();

//         // let t = v.get(n).or_else(|| v.back()).map(|x| x.t).unwrap_or(0.0);
//         let s = f(v.front().unwrap());
//         v.iter()
//             .map(|x| {
//                 let y = f(x);
//                 PlotPoint::new(x.t, if div0 { y / s - 1.0 } else { y })
//             })
//             .collect()
//     };

//     fn make_plot(
//         ui: &mut egui::Ui,
//         name: &str,
//         points: Vec<PlotPoint>,
//         near: f64,
//     ) {
//         egui_plot::Plot::new(name)
//             .allow_zoom(false)
//             .allow_drag(false)
//             .allow_scroll(false)
//             // .legend(Legend::default())
//             .include_y(near - 0.01)
//             .include_y(near + 0.01)
//             .height(400.0)
//             // .include_y(65.0)
//             .show_grid(true)
//             .show(ui, |plot_ui| {
//                 plot_ui.line(
//
// egui_plot::Line::new(PlotPoints::Owned(points)).name(name),
// );             });
//     }

//     fn plot_line(
//         plot_ui: &mut egui_plot::PlotUi,
//         name: &str,
//         points: Vec<PlotPoint>,
//     ) {
//         plot_ui
//
// .line(egui_plot::Line::new(PlotPoints::Owned(points)).name(name));     }

//     // let last = stats.sim_stat
//     let ctx = contexts.ctx_mut();
//     egui::Window::new("Stats").show(ctx, |ui| {
//         let diff_coef =
//             (state.x2s.iter().map(|x| x.length_squared()).sum::<f32>()
//                 / sim.n as f32
//                 / stats.sim_stat.back().map(|x| x.t).unwrap_or(0.0))
//                 / 6.0;

//         let v0 = stats.vels0.get(10);

//         let diff_coef2 = if let Some(v0) = v0 {
//             stats
//                 .vels0
//                 .par_iter()
//                 .skip(11)
//                 .rev()
//                 .map(|x| {
//                     x.iter().zip(v0).map(|(x1, x2)| x1.dot(*x2)).sum::<f32>()
//                 })
//                 .sum::<f32>()
//                 * sim.dt
//                 / 3.0
//                 / sim.n as f32
//         } else {
//             0.0
//         };

//         // Mean free path: {probeg_dist}
//         // Particle radius sqrt(1/(nλ*pi)): {}
//         // (sim.size.powi(3) / probeg_dist / sim.n as f32 /
//         // f32::consts::PI).sqrt()
//         let time = stats.sim_stat.back().map(|x| x.t).unwrap_or(0.0);
//         let sqdisp =
//         state.x2s.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
//         let sqdispvel = sqdisp / time;

//         #[rustfmt::skip]
// ui.label(format!(
// "{fps:.1} fps (limit: {RENDER_FPS})
// Iteration: {}
// Time: {}
// Mean Sq Disp: {}
// Vel: {}
// Time factor: {:.2} ({})
// Particles: {}
// Substeps: {}
// Iterations/s: {}
// Diffusion coef: {diff_coef}
// Diffusion coef 2: {diff_coef2}
// ",
// stats.i,
// time,
// sqdisp,
// sqdispvel,
// sim.target_dt * fps as f32,
// if sim.pause { "paused" } else { "running" },
// sim.n,
// sim.substeps,
// (sim.substeps as f32 * fps as f32 / 100.0).round() * 100.0,
// ));

//         let v = &stats.sim_stat;
//         if sim.enable_stats && !v.is_empty() {
//             let last = v.back().unwrap();
//             let temperature = last.k_e * 2.0 / 3.0 / sim.n as f32;
//             let density = sim.n as f32 * sim.size.powi(-3);
//             let pressure = temperature * density + last.pressure;

//             let diff_coef3: f32 = 10f32.powf(
//                 0.05 + 0.07 * pressure - (1.04 + 0.1 * pressure) /
// temperature,             );
//             #[rustfmt::skip]
// ui.label(format!(
// "Density: {density}
// Pressure: {pressure}
// Potential Pressure: {}
// Kinetic: {}
// Avg Vel: {}
// Potential: {}
// Temperature: {:.2}
// Diffusion should be: {diff_coef3}",
//     last.pressure,
//     last.k_e,
//     (last.k_e * 2.0).sqrt(),
//     last.p_e,
//     temperature
// ));

//             let time_coef_real =
//                 MODEL.sigma / (MODEL.epsilon / MODEL.mass).sqrt();

//             #[rustfmt::skip]
// ui.label(format!(
// "
// Time coef: {time_coef_real}
// Time: {} s
// T: {} K
// Vel: {} m/s
// Diffusion: {:e} m^2/s
// Diffusion should be: {:e} m^2/s
// Density: {} kg/m^3
// Pressure: {:e} Pa
// ",
// time * time_coef_real,
// temperature * MODEL.epsilon / MODEL.k_bolz,
// (2. * last.k_e * MODEL.epsilon / MODEL.mass / sim.n as f32).sqrt(),
// diff_coef * MODEL.sigma.powi(2) / time_coef_real,
// diff_coef3 * MODEL.sigma.powi(2) / time_coef_real,
// sim.n as f32 * MODEL.mass * (sim.size * MODEL.sigma).powi(-3),
// pressure * MODEL.epsilon * MODEL.sigma.powi(-3)
// ));

//             // PV = mu RT
//             // sigma_real = 123
//             // mass
//             // EPSILON

//             // make_plot(ui, "t_e", make_rolling(v, |x| x.p_e + x.k_e, true),
//             // 0.0); make_plot(ui, "imp", make_rolling(v, |x| x.imp,
//             // false), 0.0);

//             egui_plot::Plot::new("plot2")
//             .allow_zoom(false)
//             .allow_drag(false)
//             .allow_scroll(false)
//             .legend(egui_plot::Legend::default())
//             .height(400.0)
//             .x_axis_label("simulation time (s)")
//             // .include_y(65.0)
//             .show_grid(true)
//             .show(ui, |plot_ui| {

//             });

//             egui_plot::Plot::new("plot")
//                 .allow_zoom(false)
//                 .allow_drag(false)
//                 .allow_scroll(false)
//                 .legend(egui_plot::Legend::default())
//                 .include_y(-0.01)
//                 .include_y(0.01)
//                 .height(400.0)
//                 .x_axis_label("simulation time (s)")
//                 // .include_y(65.0)
//                 .show_grid(true)
//                 .show(ui, |plot_ui| {
//                     plot_line(
//                         plot_ui,
//                         "total_energy/total_energy0",
//                         make_rolling(v, |x| x.p_e + x.k_e, true),
//                     );
//                     plot_line(
//                         plot_ui,
//                         "impulse (sigma/s)",
//                         make_rolling(v, |x| x.imp, false),
//                     );
//                 });
//             egui_plot::Plot::new("bar").height(400.0).show(ui, |plot_ui| {
//                 let mut bars = [0.0; N_BARS];
//                 stats.vels.iter().for_each(|x| {
//                     x.iter().zip(bars.iter_mut()).for_each(|(v, s)| {
//                         *s += v;
//                     })
//                 });

//                 let bars = bars
//                     .iter()
//                     .enumerate()
//                     .map(|(i, x)| {
//                         egui_plot::Bar::new(
//                             (0.5 + i as f64) * BARS_S,
//                             *x as f64 / stats.vels.len() as f64,
//                         )
//                         .width(BARS_S)
//                     })
//                     .collect();

//                 plot_ui.bar_chart(egui_plot::BarChart::new(bars));

//                 // plot_ui.bar_chart(egui_plot::BarChart::new(make_bar_chart(
//                 //     &state.vs,
//                 // )));
//             });
//         }
//     });
// }

// fn ui_system(
//     sim: Res<Sim>,
//     mut contexts: EguiContexts,
//     diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
//     mut stats: ResMut<Stats>,
//     state: Res<SimState>,
//     time: Res<Time<Real>>,
// ) {
//     // let dt = time.delta_secs();
//     // let fps = if dt == 0.0 { 0.0 } else { 1.0 / time.delta_secs() };

//     let fps = diagnostics
//         .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
//         .and_then(|f| f.average())
//         .unwrap_or(0.);

//     fn make_rolling(
//         v: &VecDeque<Stat>,
//         f: impl Fn(&Stat) -> f32,
//         div0: bool,
//     ) -> Vec<PlotPoint> {
//         // let n = 2;
//         // let mut s: f64 = v.iter().take(n).map(|x| f(x) as f64).sum();
//         // let s0 = s;
//         // let t = v.get(n).or_else(|| v.back()).map(|x| x.t).unwrap_or(0.0);
//         // std::iter::once(PlotPoint::new(
//         //     t,
//         //     if div0 { s / s0 } else { s / n as f64 },
//         // ))
//         // .chain(v.iter().zip(v.iter().skip(n)).map(|(la, ne)| {
//         //     let last = f(la) as f64;
//         //     let next = f(ne) as f64;
//         //     s = s - last + next;
//         //     PlotPoint::new(ne.t, if div0 { s / s0 } else { s / n as f64 })
//         // }))
//         // .collect();

//         // let t = v.get(n).or_else(|| v.back()).map(|x| x.t).unwrap_or(0.0);
//         let s = f(v.front().unwrap());
//         v.iter()
//             .map(|x| {
//                 let y = f(x);
//                 PlotPoint::new(x.t, if div0 { y / s - 1.0 } else { y })
//             })
//             .collect()
//     };

//     fn make_plot(
//         ui: &mut egui::Ui,
//         name: &str,
//         points: Vec<PlotPoint>,
//         near: f64,
//     ) {
//         egui_plot::Plot::new(name)
//             .allow_zoom(false)
//             .allow_drag(false)
//             .allow_scroll(false)
//             // .legend(Legend::default())
//             .include_y(near - 0.01)
//             .include_y(near + 0.01)
//             .height(400.0)
//             // .include_y(65.0)
//             .show_grid(true)
//             .show(ui, |plot_ui| {
//                 plot_ui.line(

// egui_plot::Line::new(PlotPoints::Owned(points)).name(name),
// );             });
//     }

//     fn plot_line(
//         plot_ui: &mut egui_plot::PlotUi,
//         name: &str,
//         points: Vec<PlotPoint>,
//     ) {
//         plot_ui

// .line(egui_plot::Line::new(PlotPoints::Owned(points)).name(name));     }

//     // let last = stats.sim_stat
//     let ctx = contexts.ctx_mut();
//     egui::Window::new("Stats").show(ctx, |ui| {

//             // PV = mu RT
//             // sigma_real = 123
//             // mass
//             // EPSILON

//             // make_plot(ui, "t_e", make_rolling(v, |x| x.p_e + x.k_e, true),
//             // 0.0); make_plot(ui, "imp", make_rolling(v, |x| x.imp,
//             // false), 0.0);

//             egui_plot::Plot::new("plot2")
//             .allow_zoom(false)
//             .allow_drag(false)
//             .allow_scroll(false)
//             .legend(egui_plot::Legend::default())
//             .height(400.0)
//             .x_axis_label("simulation time (s)")
//             // .include_y(65.0)
//             .show_grid(true)
//             .show(ui, |plot_ui| {

//             });

//             egui_plot::Plot::new("plot")
//                 .allow_zoom(false)
//                 .allow_drag(false)
//                 .allow_scroll(false)
//                 .legend(egui_plot::Legend::default())
//                 .include_y(-0.01)
//                 .include_y(0.01)
//                 .height(400.0)
//                 .x_axis_label("simulation time (s)")
//                 // .include_y(65.0)
//                 .show_grid(true)
//                 .show(ui, |plot_ui| {
//                     plot_line(
//                         plot_ui,
//                         "total_energy/total_energy0",
//                         make_rolling(v, |x| x.p_e + x.k_e, true),
//                     );
//                     plot_line(
//                         plot_ui,
//                         "impulse (sigma/s)",
//                         make_rolling(v, |x| x.imp, false),
//                     );
//                 });
//             egui_plot::Plot::new("bar").height(400.0).show(ui, |plot_ui| {
//                 let mut bars = [0.0; N_BARS];
//                 stats.vels.iter().for_each(|x| {
//                     x.iter().zip(bars.iter_mut()).for_each(|(v, s)| {
//                         *s += v;
//                     })
//                 });

//                 let bars = bars
//                     .iter()
//                     .enumerate()
//                     .map(|(i, x)| {
//                         egui_plot::Bar::new(
//                             (0.5 + i as f64) * BARS_S,
//                             *x as f64 / stats.vels.len() as f64,
//                         )
//                         .width(BARS_S)
//                     })
//                     .collect();

//                 plot_ui.bar_chart(egui_plot::BarChart::new(bars));

//                 // plot_ui.bar_chart(egui_plot::BarChart::new(make_bar_chart(
//                 //     &state.vs,
//                 // )));
//             });
        
//     });
// }

#[derive(Resource)]
pub struct SimController {
    pub params: Sim,
    pub state: SimState,
    pub stats: Stats,
}

#[derive(Resource, Default)]
pub struct Stats {
    pub i: u32,
    pub real_time: VecDeque<(f32, f32)>,
    pub sim_stat: VecDeque<Stat>,
    // vels: VecDeque<Box<[f32]>>,
    pub vels: VecDeque<Box<[(f32, f32)]>>,
    // vels0: VecDeque<Box<[Vec3]>>,
    pub r2t: Vec<f32>,
    pub v0: Box<[Vec3]>,
    pub v0vt: Vec<f32>,

    pub diff_x: Vec<f32>,
}

impl Stats {
    fn new() -> Self {
        const CAP: usize = 10000;
        Self {
            i: 0,
            real_time: VecDeque::with_capacity(CAP),
            sim_stat: VecDeque::with_capacity(CAP),
            vels: VecDeque::with_capacity(100),
            // vels0: VecDeque::with_capacity(30000),
            r2t: vec![0.0; ITERS as usize],
            v0vt: vec![0.0; ITERS as usize],
            ..default()
        }
    }
}

#[derive(Default, Clone)]
pub struct Stat {
    pub t: f32,
    pub k_e: f32,
    pub p_e: f32,
    pub imp: f32,
    pub pressure: f32,
    pub dx2: f32,
}

impl SimController {
    pub fn new(mut sim: Sim) -> Self {
        sim.update();
        assert_ne!(sim.substeps, 0);
        Self {
            params: sim.clone(),
            state: SimState::new(&sim),
            stats: Stats::new(),
        }
    }

    pub fn simple_sim(&mut self) {
        let sim = &self.params;
        let mut state = &mut self.state;
        let stats = &mut self.stats;

        if sim.substeps > 0 {
            for _ in 1..sim.substeps {
                step(sim, state, stats, false);

                // let vels0 = &mut stats.vels0;
                // if vels0.len() == vels0.capacity() {
                //     vels0.pop_front();
                // }
                // vels0.push_back(state.vs.clone());
            }
            let (p_e, pressure) = step(sim, state, stats, true);
            let v = &mut stats.sim_stat;
            let t = v.back().map(|x| x.t).unwrap_or(0.0) + sim.target_dt;
            // stats.sim_time += sim.target_dt as f64;
            // let t = stats.sim_time;
            let k_e: f32 = (state
                .vs
                .iter()
                .map(|x| x.length_squared() as f64)
                .sum::<f64>()
                / 2.0) as f32;
            let imp: f32 = state.vs.iter().sum::<Vec3>().length();
            if v.len() == v.capacity() {
                v.pop_front();
            }
            let dx2 = state.x2s.iter().map(|x| x.length_squared()).sum::<f32>()
                / sim.n as f32;
            v.push_back(Stat { t, k_e, p_e, imp, pressure, dx2 });

            let vels = &mut stats.vels;
            if vels.len() == vels.capacity() {
                vels.pop_front();
            }

            let mut vs: Vec<f32> =
                state.vs.iter().map(|x| x.x.powi(2)).collect();
            vs.sort_unstable_by(|a, b| a.total_cmp(b));
            // let s = BARS_S as f32;
            let s = vs.iter().sum::<f32>() / sim.n as f32 / 2.0;
            let mut bars: Vec<(f32, f32)> = Vec::new();
            let mut it = vs.iter();
            'a: for i in 0..N_BARS {
                let mut c = 0;
                for v in it.by_ref() {
                    if *v > (i + 1) as f32 * s {
                        break;
                    }
                    c += 1;
                }
                let lnc = if c == 0 { 0.0 } else { (c as f32).ln() };
                bars.push((s * i as f32, lnc));
            }
            vels.push_back(bars.into_boxed_slice());

            stats.i += sim.substeps;
        }
    }


    fn stats(&mut self) {

        let v = self.stats.sim_stat.back().cloned().unwrap_or_default();
        let state = &self.state;
        let t = v.t;
        let n = self.params.n as f32;


        let diff_coef = state.x2s.iter().map(|x| x.length_squared()).sum::<f32>()
            / n
            / t
            / 6.0;

    // let v0 = stats.vels0.get(10);

    // let diff_coef2 = if let Some(v0) = v0 {
    //     stats
    //         .vels0
    //         .par_iter()
    //         .skip(11)
    //         .rev()
    //         .map(|x| {
    //             x.iter().zip(v0).map(|(x1, x2)| x1.dot(*x2)).sum::<f32>()
    //         })
    //         .sum::<f32>()
    //         * sim.dt
    //         / 3.0
    //         / sim.n as f32
    // } else {
    //     0.0
    // };

    // Mean free path: {probeg_dist}
    // Particle radius sqrt(1/(nλ*pi)): {}
    // (sim.size.powi(3) / probeg_dist / sim.n as f32 /
    // f32::consts::PI).sqrt()
    // let time = stats.sim_stat.back().map(|x| x.t).unwrap_or(0.0);
    // let sqdisp =
    // state.x2s.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    // let sqdispvel = sqdisp / time;


        let last = v;
        let temperature = last.k_e * 2.0 / 3.0 / n as f32;
        let density = n * self.params.size.powi(-3);
        let pressure = temperature * density + last.pressure;

        let diff_coef3: f32 = 10f32.powf(
            0.05 + 0.07 * pressure - (1.04 + 0.1 * pressure) /
temperature,             );
//         #[rustfmt::skip]
// ui.label(format!(
// "Density: {density}
// Pressure: {pressure}
// Potential Pressure: {}
// Kinetic: {}
// Avg Vel: {}
// Potential: {}
// Temperature: {:.2}
// Diffusion should be: {diff_coef3}",
// last.pressure,
// last.k_e,
// (last.k_e * 2.0).sqrt(),
// last.p_e,
// temperature
// ));

//         let time_coef_real =
//             MODEL.sigma / (MODEL.epsilon / MODEL.mass).sqrt();

//         #[rustfmt::skip]
// ui.label(format!(
// "
// Time coef: {time_coef_real}
// Time: {} s
// T: {} K
// Vel: {} m/s
// Diffusion: {:e} m^2/s
// Diffusion should be: {:e} m^2/s
// Density: {} kg/m^3
// Pressure: {:e} Pa
// ",
// time * time_coef_real,
// temperature * MODEL.epsilon / MODEL.k_bolz,
// (2. * last.k_e * MODEL.epsilon / MODEL.mass / sim.n as f32).sqrt(),
// diff_coef * MODEL.sigma.powi(2) / time_coef_real,
// diff_coef3 * MODEL.sigma.powi(2) / time_coef_real,
// sim.n as f32 * MODEL.mass * (sim.size * MODEL.sigma).powi(-3),
// pressure * MODEL.epsilon * MODEL.sigma.powi(-3)
// ));
    }
}

#[derive(Clone)]
pub struct Sim {
    pub n_row: u32,
    pub size: f32,
    pub target_dt: f32,
    pub substeps: u32,
    pub temp0: f32,
    pub disable_speed: bool,
    pub ignore_cells: bool,

    pub grid: f32,
    pub n_spacial: u32,
    pub n: u32,
    pub n_grid: u32,
    pub dt: f32,
    pub half_dt: f32,
}

impl Default for Sim {
    fn default() -> Self {
        let mut s = Self {
            n_row: 8,
            size: 16.0,
            target_dt: 1./30., // TODO
            substeps: 10,
            temp0: 2.0,
            disable_speed: false,
            ignore_cells: false,

            grid: 0.0,
            n_spacial: 0,
            n: 0,
            n_grid: 0,
            dt: 0.0,
            half_dt: 0.0,
        };
        s.update();
        s
    }
}

impl Sim {
    fn update(&mut self) {
        if self.n == 0 {
            self.n = self.n_row.pow(3);
        }

        if self.grid != 0.0 {
            let n_grid = self.size / self.grid;
            assert_eq!(
                n_grid.round(),
                n_grid,
                "n of cells must be a whole number"
            );
            self.n_grid = n_grid as u32;
        } else {
            let n_grid = (self.size / CUTOFF).floor();
            self.grid = self.size / n_grid;
            self.n_grid = n_grid as u32;
            // warn!("grid size: {}", self.grid);
        }
        warn!("grid size: {}", self.grid);

        if self.n_spacial == 0 {
            self.n_spacial = self.n_grid.pow(3);
        }
        self.dt = self.target_dt / self.substeps as f32;
        self.half_dt = self.dt / 2.;
    }
}

pub struct SimState {
    pub i: i32,
    pub reruns: u32,
    pub xs: Box<[Vec3]>,
    pub vs: Box<[Vec3]>,
    pub fs: Box<[Vec3]>,
    pub x2s: Box<[Vec3]>,
    pub spacial_lookup: Box<[(u32, u32)]>,
    pub start_indicies: Box<[u32]>,
}

impl SimState {
    fn new(sim: &Sim) -> Self {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let n = sim.n;
        let mut r = StdRng::seed_from_u64(0);
        let mut get_r = || {
            r.sample(rand_distr::Normal::new(0.0, sim.temp0.sqrt()).unwrap())
        };
        Self {
            i: -200,
            reruns: 0,
            xs: (0..n).map(|i| gen_point(i, sim)).collect(),
            vs: {
                let mut vs: Box<[Vec3]> = (0..n)
                    .map(|_| {
                        if sim.disable_speed {
                            Vec3::ZERO
                        } else {
                            Vec3::new(get_r(), get_r(), get_r())
                        }
                    })
                    .collect();

                let imp = vs.iter().sum::<Vec3>() / n as f32;
                vs.iter_mut().for_each(|x| *x -= imp);
                let v2sum: f32 = vs.iter().map(|x| x.length_squared()).sum();
                let v2target = sim.temp0 * 1.5 * sim.n as f32;
                let factor = (v2target / v2sum).sqrt();
                vs.iter_mut().for_each(|x| *x *= factor);

                vs
            },
            fs: vec![Vec3::ZERO; n as usize].into_boxed_slice(),
            x2s: vec![Vec3::ZERO; n as usize].into_boxed_slice(),
            spacial_lookup: vec![(0, 0); n as usize].into_boxed_slice(),
            start_indicies: vec![u32::MAX; sim.n_spacial as usize]
                .into_boxed_slice(),
        }
    }
}

fn gen_point(i: u32, sim: &Sim) -> Vec3 {
    let s = sim.n_row;
    let half = (s - 1) as f32 / 2.;

    let x = Vec3::new(
        (i / s / s) as f32 - half,
        (i / s % s) as f32 - half,
        (i % s) as f32 - half,
    ) / s as f32
        * 2.;
    x * sim.size / 2.0
}

fn get_cell(x: Vec3, sim: &Sim) -> IVec3 {
    (x / sim.grid + 0.5).floor().as_ivec3()
}

fn get_key(x: IVec3, sim: &Sim) -> u32 {
    if sim.ignore_cells {
        return 0;
    }

    let x = x.rem_euclid(IVec3::splat(sim.n_grid as i32)).as_uvec3();

    // (x.x)
    //     .wrapping_mul(15823)
    //     .wrapping_add((x.y).wrapping_mul(9737333))
    //     .wrapping_add((x.z).wrapping_mul(440817757))
    let hash = (x.z * sim.n_grid + x.y) * sim.n_grid + x.x;
    // TODO: assert that % n_spacial is not needed?
    hash % sim.n_spacial
}

const START: i32 = 200;
const ITERS: i32 = 1000;

fn step(sim: &Sim, state: &mut SimState, stats: &mut Stats, calc_stats: bool) -> (f32, f32) {
    let SimState { i, reruns, xs, vs, fs, x2s, spacial_lookup, start_indicies } = state;
    *i += 1;

    if *i >= (START + ITERS) {
        *i = 0;
        *reruns += 1;
    } 
    if *i == START {
        stats.v0 = vs.clone();
        *x2s = vec![Vec3::ZERO; sim.n as usize].into_boxed_slice();
    }
    if *i >= START {
        let r2t = &mut stats.r2t[(*i as i32 - START) as usize];
        *r2t = (*r2t * (*reruns + 1) as f32 + (x2s.iter().map(|x| x.length_squared()).sum::<f32>())) / ((*reruns + 2) as f32);
        let v0vt = &mut stats.v0vt[(*i as i32 - START) as usize];
        *v0vt = (*v0vt * (*reruns + 1) as f32 + (vs.iter().zip(stats.v0.iter()).map(|(v, v0)| v.dot(*v0)).sum::<f32>())) / ((*reruns + 2) as f32);
    }

    // leapfrog1 TODO: do i mirror the points here?
    for (x, x2, v) in izip!(&mut *xs, &mut *x2s, &*vs) {
        let dx = v * sim.half_dt;
        *x += dx;
        *x2 += dx;
        *x -= (*x / sim.size * 2.0).trunc() * sim.size;
        if !x.is_finite() {
            warn!("nan/inf detected");
        }
    }

    for (i, sl, x) in izip!(0.., &mut *spacial_lookup, &*xs) {
        let cell = get_cell(*x, sim);
        let key = get_key(cell, sim);
        *sl = (i, key);
    }

    // spacial_lookup.sort_unstable_by_key(|x| x.1);

    spacial_lookup.par_sort_unstable_by_key(|x| x.1);

    start_indicies.fill(u32::MAX);

    let mut prev = u32::MAX;
    (0..).zip(spacial_lookup.iter()).for_each(|(i, x)| {
        let key = x.1;
        if prev != key {
            start_indicies[key as usize] = i;
        }
        prev = key;
    });

    let (pot, pressure) = fs
        .par_iter_mut()
        .zip(xs.par_iter())
        .enumerate()
        .map(|(i, (f, &x1))| {
            let cell = get_cell(x1, sim);
            let mut f1 = Vec3::ZERO;
            let mut pot1 = 0.0;
            let mut pressure1 = 0.0;
            for off in OFFSETS {
                let c = cell + off;
                let key = get_key(c, sim);
                let start_i = start_indicies[key as usize] as usize;

                let mut f2 = Vec3::ZERO;
                let mut pot2 = 0.0;
                let mut pressure2 = 0.0;
                spacial_lookup
                    .iter()
                    .skip(start_i)
                    .take_while(|x| x.1 == key)
                    .for_each(|x| {
                        let j = x.0 as usize;
                        if i == j {
                            return;
                        }
                        let x2 = xs[j];
                        let mut dr = x1 - x2;
                        dr -= (dr / sim.size).round() * sim.size;
                        let r2 = dr.length_squared();

                        if r2 > CUTOFF * CUTOFF {
                            return;
                        }

                        let r = r2.sqrt();

                        let ff = dr / r * force(r);

                        f2 -= ff;
                        if calc_stats {
                            pot2 += energy(r);
                            pressure2 += ff.dot(dr);
                        }
                    });

                f1 += f2;
                pot1 += pot2;
                pressure1 += pressure2;
            }
            *f = f1;
            (pot1, pressure1)
        })
        .reduce(|| (0., 0.), |a, b| (a.0 + b.0, a.1 + b.1));

    for (x, v, f, x2) in izip!(&mut *xs, &mut *vs, &*fs, &mut *x2s) {
        *v += f * sim.dt;
        let dx = *v * sim.half_dt;
        *x += dx;
        *x2 += dx;
        *x -= (*x / sim.size * 2.0).trunc() * sim.size;
    }

    // added twice per particle
    (pot / 2.0, pressure * sim.size.powi(-3) / 3.0)
}

fn energy(dr: f32) -> f32 {
    4.0 * (dr.powi(-12) - dr.powi(-6))
}
fn force(dr: f32) -> f32 {
    48.0 * (-dr.powi(-13) + 0.5 * dr.powi(-7))
}

const OFFSETS: [IVec3; 27] = [
    IVec3::new(-1, -1, -1),
    IVec3::new(-1, -1, 0),
    IVec3::new(-1, -1, 1),
    IVec3::new(-1, 0, -1),
    IVec3::new(-1, 0, 0),
    IVec3::new(-1, 0, 1),
    IVec3::new(-1, 1, -1),
    IVec3::new(-1, 1, 0),
    IVec3::new(-1, 1, 1),
    IVec3::new(0, -1, -1),
    IVec3::new(0, -1, 0),
    IVec3::new(0, -1, 1),
    IVec3::new(0, 0, -1),
    // IVec3::new(0, 0, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 1, -1),
    IVec3::new(0, 1, 0),
    IVec3::new(0, 1, 1),
    IVec3::new(1, -1, -1),
    IVec3::new(1, -1, 0),
    IVec3::new(1, -1, 1),
    IVec3::new(1, 0, -1),
    IVec3::new(1, 0, 0),
    IVec3::new(1, 0, 1),
    IVec3::new(1, 1, -1),
    IVec3::new(1, 1, 0),
    IVec3::new(1, 1, 1),
    IVec3::new(0, 0, 0), // last to minimise float loss
];
