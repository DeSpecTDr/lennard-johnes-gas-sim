#![allow(unused)]

use core::f32;
use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use egui_plot::{PlotPoint, PlotPoints};
use itertools::izip;
use ops::FloatPow;
use rayon::prelude::*;

fn main() {
    let sim = Sim::default();
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Simulation".into(),
                ..default()
            }),
            ..default()
        }),
        bevy::diagnostic::LogDiagnosticsPlugin::default(),
        bevy::diagnostic::FrameTimeDiagnosticsPlugin,
        bevy_framepace::FramepacePlugin,
        EguiPlugin,
        // bevy_framepace::debug::DiagnosticsPlugin,
    ))
    .add_systems(Startup, setup)
    // .add_systems(Update, reset_sim.run_if(resource_changed::<Sim>))
    .add_systems(Update, (simple_sim, reset_on_key, ui_system));
    if DRAW {
        app.insert_resource(bevy_framepace::FramepaceSettings {
            limiter: bevy_framepace::Limiter::from_framerate(RENDER_FPS),
        })
        .add_systems(Update, (draw, orbit_camera));
    }
    app.run();
}

const DRAW: bool = true;

const RENDER_FPS: f64 = 30.0;

#[derive(Resource)]
struct Stats {
    i: u32,
    real_time: VecDeque<(f32, f32)>,
    sim_stat: VecDeque<Stat>,
    // vels: VecDeque<Box<[f32]>>,
    vels: VecDeque<Box<[f32]>>,
    vels0: VecDeque<Box<[Vec3]>>,
}

#[derive(Default, Clone)]
struct Stat {
    t: f32,
    k_e: f32,
    p_e: f32,
    imp: f32,
    pressure: f32,
    dx2: f32,
}

impl Default for Stats {
    fn default() -> Self {
        const CAP: usize = 10000;
        Self {
            i: 0,
            real_time: VecDeque::with_capacity(CAP),
            sim_stat: VecDeque::with_capacity(CAP),
            vels: VecDeque::with_capacity(100),
            vels0: VecDeque::with_capacity(30000),
        }
    }
}

struct PhysParams {
    sigma: f32,
    epsilon: f32,
    mass: f32,
    k_bolz: f32,
}

// const MODEL: PhysParams = PhysParams {
//     sigma: 2.725e-10,
//     epsilon: 4.9115e-21,
//     mass: 30.103e-27,
//     k_bolz: 1.380649e-23,
// };

const AVOGADRO: f32 = 6.0221408e23;

const MODEL: PhysParams = PhysParams {
    sigma: 3.405e-10,
    epsilon: 1.6537e-21,
    mass: 39.948 * 0.001 / AVOGADRO,
    k_bolz: 1.380649e-23,
};

const N_BARS: usize = 20;
const BARS_S: f64 = 20.0;

fn ui_system(
    sim: Res<Sim>,
    mut contexts: EguiContexts,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    mut stats: ResMut<Stats>,
    state: Res<SimState>,
    time: Res<Time<Real>>,
) {
    // let dt = time.delta_secs();
    // let fps = if dt == 0.0 { 0.0 } else { 1.0 / time.delta_secs() };

    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|f| f.average())
        .unwrap_or(0.);

    fn make_rolling(
        v: &VecDeque<Stat>,
        f: impl Fn(&Stat) -> f32,
        div0: bool,
    ) -> Vec<PlotPoint> {
        // let n = 2;
        // let mut s: f64 = v.iter().take(n).map(|x| f(x) as f64).sum();
        // let s0 = s;
        // let t = v.get(n).or_else(|| v.back()).map(|x| x.t).unwrap_or(0.0);
        // std::iter::once(PlotPoint::new(
        //     t,
        //     if div0 { s / s0 } else { s / n as f64 },
        // ))
        // .chain(v.iter().zip(v.iter().skip(n)).map(|(la, ne)| {
        //     let last = f(la) as f64;
        //     let next = f(ne) as f64;
        //     s = s - last + next;
        //     PlotPoint::new(ne.t, if div0 { s / s0 } else { s / n as f64 })
        // }))
        // .collect();

        // let t = v.get(n).or_else(|| v.back()).map(|x| x.t).unwrap_or(0.0);
        let s = f(v.front().unwrap());
        v.iter()
            .map(|x| {
                let y = f(x);
                PlotPoint::new(x.t, if div0 { y / s - 1.0 } else { y })
            })
            .collect()
    };

    fn make_plot(
        ui: &mut egui::Ui,
        name: &str,
        points: Vec<PlotPoint>,
        near: f64,
    ) {
        egui_plot::Plot::new(name)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            // .legend(Legend::default())
            .include_y(near - 0.01)
            .include_y(near + 0.01)
            .height(400.0)
            // .include_y(65.0)
            .show_grid(true)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    egui_plot::Line::new(PlotPoints::Owned(points)).name(name),
                );
            });
    }

    fn plot_line(
        plot_ui: &mut egui_plot::PlotUi,
        name: &str,
        points: Vec<PlotPoint>,
    ) {
        plot_ui
            .line(egui_plot::Line::new(PlotPoints::Owned(points)).name(name));
    }

    // let last = stats.sim_stat
    let ctx = contexts.ctx_mut();
    egui::Window::new("Stats").show(ctx, |ui| {
        let diff_coef =
            (state.x2s.iter().map(|x| x.length_squared()).sum::<f32>()
                / sim.n as f32
                / stats.sim_stat.back().map(|x| x.t).unwrap_or(0.0))
                / 2.0
                / (if sim.is_3d { 3.0 } else { 2.0 });

        let v0 = stats.vels0.get(10);

        let diff_coef2 = if let Some(v0) = v0 {
            stats
                .vels0
                .par_iter()
                .skip(11)
                .rev()
                .map(|x| {
                    x.iter().zip(v0).map(|(x1, x2)| x1.dot(*x2)).sum::<f32>()
                })
                .sum::<f32>()
                * sim.dt
                / 3.0
                / sim.n as f32
        } else {
            0.0
        };

        // Mean free path: {probeg_dist}
        // Particle radius sqrt(1/(nλ*pi)): {}
        // (sim.size.powi(3) / probeg_dist / sim.n as f32 /
        // f32::consts::PI).sqrt()
        let time = stats.sim_stat.back().map(|x| x.t).unwrap_or(0.0);
        let sqdisp = 
        state.x2s.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let sqdispvel = sqdisp / time;


        #[rustfmt::skip]
ui.label(format!(
"{fps:.1} fps (limit: {RENDER_FPS})
Iteration: {}
Time: {}
Mean Sq Disp: {}
Vel: {}
Time factor: {:.2} ({})
Particles: {}
Substeps: {}
Iterations/s: {}
Diffusion coef: {diff_coef}
Diffusion coef 2: {diff_coef2}
",
stats.i,
time,
sqdisp,
sqdispvel,
sim.target_dt * fps as f32,
if sim.pause { "paused" } else { "running" },
sim.n,
sim.substeps,
(sim.substeps as f32 * fps as f32 / 100.0).round() * 100.0,
));

        let v = &stats.sim_stat;
        if sim.enable_stats && !v.is_empty() {
            let last = v.back().unwrap();
            let temperature = last.k_e * 2.0 / 3.0 / sim.n as f32;
            let density = sim.n as f32 * sim.size.powi(-3);
            let pressure = temperature * density + last.pressure;

            let diff_coef3: f32 = 10f32.powf(
                0.05 + 0.07 * pressure - (1.04 + 0.1 * pressure) / temperature,
            );
            #[rustfmt::skip]
ui.label(format!(
"Density: {density}
Pressure: {pressure}
Potential Pressure: {}
Kinetic: {}
Avg Vel: {}
Potential: {}
Temperature: {:.2}
Diffusion should be: {diff_coef3}",
    last.pressure,
    last.k_e,
    (last.k_e * 2.0).sqrt(),
    last.p_e,
    temperature
));

            let time_coef_real =
                MODEL.sigma / (MODEL.epsilon / MODEL.mass).sqrt();

            #[rustfmt::skip]
ui.label(format!(
"
Time coef: {time_coef_real}
Time: {} s
T: {} K
Vel: {} m/s
Diffusion: {:e} m^2/s
Diffusion should be: {:e} m^2/s
Density: {} kg/m^3
Pressure: {:e} Pa
",
time * time_coef_real,
temperature * MODEL.epsilon / MODEL.k_bolz,
(2. * last.k_e * MODEL.epsilon / MODEL.mass / sim.n as f32).sqrt(),
diff_coef * MODEL.sigma.powi(2) / time_coef_real,
diff_coef3 * MODEL.sigma.powi(2) / time_coef_real,
sim.n as f32 * MODEL.mass * (sim.size * MODEL.sigma).powi(-3),
pressure * MODEL.epsilon * MODEL.sigma.powi(-3)
));

            // PV = mu RT
            // sigma_real = 123
            // mass
            // EPSILON

            // make_plot(ui, "t_e", make_rolling(v, |x| x.p_e + x.k_e, true),
            // 0.0); make_plot(ui, "imp", make_rolling(v, |x| x.imp,
            // false), 0.0);

            egui_plot::Plot::new("plot2")
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .legend(egui_plot::Legend::default())
            .height(400.0)
            .x_axis_label("simulation time (s)")
            // .include_y(65.0)
            .show_grid(true)
            .show(ui, |plot_ui| {

            });

            egui_plot::Plot::new("plot")
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .legend(egui_plot::Legend::default())
                .include_y(-0.01)
                .include_y(0.01)
                .height(400.0)
                .x_axis_label("simulation time (s)")
                // .include_y(65.0)
                .show_grid(true)
                .show(ui, |plot_ui| {
                    plot_line(
                        plot_ui,
                        "total_energy/total_energy0",
                        make_rolling(v, |x| x.p_e + x.k_e, true),
                    );
                    plot_line(
                        plot_ui,
                        "impulse (sigma/s)",
                        make_rolling(v, |x| x.imp, false),
                    );
                });
            egui_plot::Plot::new("bar").height(400.0).show(ui, |plot_ui| {
                let mut bars = [0.0; N_BARS];
                stats.vels.iter().for_each(|x| {
                    x.iter().zip(bars.iter_mut()).for_each(|(v, s)| {
                        *s += v;
                    })
                });

                let bars = bars
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        egui_plot::Bar::new(
                            (0.5 + i as f64) * BARS_S,
                            *x as f64 / stats.vels.len() as f64,
                        )
                        .width(BARS_S)
                    })
                    .collect();

                plot_ui.bar_chart(egui_plot::BarChart::new(bars));

                // plot_ui.bar_chart(egui_plot::BarChart::new(make_bar_chart(
                //     &state.vs,
                // )));
            });
        }
    });
}

#[derive(Resource)]
struct Sim {
    is_3d: bool,
    n_row: u32,
    size: f32,
    grid: f32,
    target_dt: f32,
    substeps: u32,
    draw: bool,
    draw_meshes: bool,
    disable_speed: bool,
    ignore_cells: bool,
    draw_mirrors: bool,
    enable_stats: bool,
    pause: bool,
    temp0: f32,
    // walls: bool
    n_spacial: u32,
    n: u32,
    n_grid: u32,
    dt: f32,
    half_dt: f32,
}

const CUTOFF: f32 = 4.;

impl Default for Sim {
    fn default() -> Self {
        let mut s = Self {
            is_3d: true,
            n_row: 10,
            size: 16.0,
            grid: CUTOFF,
            // target_dt: 1.0 / RENDER_FPS as f32,
            target_dt: 1. / RENDER_FPS as f32,
            substeps: 5,
            draw: DRAW,
            draw_meshes: DRAW,
            disable_speed: false,
            ignore_cells: false,
            draw_mirrors: false,
            enable_stats: true,
            pause: false,
            temp0: 1.0,

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
            self.n = self.n_row.pow(if self.is_3d { 3 } else { 2 });
        }
        let n_grid = self.size / self.grid;
        assert_eq!(n_grid.round(), n_grid, "n of cells must be a whole number");
        self.n_grid = n_grid as u32;
        if self.n_spacial == 0 {
            self.n_spacial = self.n_grid.pow(3); //self.n;
        }
        self.dt = self.target_dt / self.substeps as f32;
        self.half_dt = self.dt / 2.;
    }
}

#[derive(Resource)]
struct SimState {
    xs: Box<[Vec3]>,
    vs: Box<[Vec3]>,
    fs: Box<[Vec3]>,
    x2s: Box<[Vec3]>,
    spacial_lookup: Box<[(u32, u32)]>,
    start_indicies: Box<[u32]>,
}

impl SimState {
    fn new(sim: &Sim) -> Self {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let n = sim.n;
        let mut r = StdRng::seed_from_u64(0);
        // TODO: use normal distr with sigma = sqrt(temperature)
        let mut get_r = || {
            // r.random_range(-1.0..=1.0);
            r.sample(rand_distr::Normal::new(0.0, sim.temp0.sqrt()).unwrap())
        };
        Self {
            xs: (0..n).map(|i| gen_point(i, sim)).collect(),
            vs: {
                let mut vs: Box<[Vec3]> = (0..n)
                    .map(|_| {
                        if sim.disable_speed {
                            Vec3::ZERO
                        } else {
                            Vec3::new(
                                get_r(),
                                get_r(),
                                if sim.is_3d { get_r() } else { 0.0 },
                            )
                        }
                    })
                    .collect();

                let imp = vs.iter().sum::<Vec3>() / n as f32;
                vs.iter_mut().for_each(|x| *x -= imp);

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

    let x = if sim.is_3d {
        Vec3::new(
            (i / s / s) as f32 - half,
            (i / s % s) as f32 - half,
            (i % s) as f32 - half,
        ) / s as f32
            * 2.
    } else {
        Vec3::new((i / s) as f32 - half, (i % s) as f32 - half, 0.0) / s as f32
            * 2.
    };
    x * sim.size / 2.0
}

fn get_cell(x: Vec3, sim: &Sim) -> IVec3 {
    // let off = (sim.n_grid % 2) as f32 * 0.5;
    // (x / sim.grid + off).floor().as_ivec3()
    (x / sim.grid + 0.5).floor().as_ivec3()
}

fn get_hash(x: IVec3, sim: &Sim) -> u32 {
    if sim.ignore_cells {
        return 0;
    }

    let x = x.rem_euclid(IVec3::splat(sim.n_grid as i32)).as_uvec3();

    // (x.x)
    //     .wrapping_mul(15823)
    //     .wrapping_add((x.y).wrapping_mul(9737333))
    //     .wrapping_add((x.z).wrapping_mul(440817757))
    (x.z * sim.n_grid + x.y) * sim.n_grid + x.x
}

fn get_key(x: u32, sim: &Sim) -> u32 {
    // n_keys = n_particles
    x % sim.n_spacial
}

fn simple_sim(
    sim: Res<Sim>,
    mut state: ResMut<SimState>,
    mut stats: ResMut<Stats>,
) {
    // let SimState { xs, vs, fs, spacial_lookup, start_indicies } =
    // state.into_inner(); for (x, v) in xs.iter_mut().zip(vs.iter()) {
    //     *x += v * sim.dt;
    //     *x -= (*x / sim.size * 2.0).trunc() * sim.size;
    // }
    if sim.substeps > 0 && !sim.pause {
        for _ in 1..sim.substeps {
            step(&sim, &mut state, false);

            let vels0 = &mut stats.vels0;
            if vels0.len() == vels0.capacity() {
                vels0.pop_front();
            }
            vels0.push_back(state.vs.clone());
        }
        if sim.enable_stats {
            let (p_e, pressure) = step(&sim, &mut state, true);
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
            let dx2 = state.x2s.iter().map(|x| x.length_squared()).sum::<f32>() / sim.n as f32;
            v.push_back(Stat { t, k_e, p_e, imp, pressure, dx2 });
            // push(&mut stats.sim_time, t);
            // push(&mut stats.k_e, k_e);
            // push(&mut stats.p_e, p_e);
            // push(&mut stats.t_e, k_e + p_e);
            // push(&mut stats.imp, imp);

            let vels = &mut stats.vels;
            if vels.len() == vels.capacity() {
                vels.pop_front();
            }

            let mut vs: Vec<f32> =
                state.vs.iter().map(|x| x.x.squared()).collect();
            vs.sort_unstable_by(|a, b| a.total_cmp(b));
            // let s = BARS_S as f32;
            let s = vs.iter().sum::<f32>() / sim.n as f32 / 2.0;
            let mut bars: Vec<f32> = Vec::new();
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
                bars.push(lnc);
            }
            vels.push_back(bars.into_boxed_slice());
        } else {
            step(&sim, &mut state, false);
        }
        stats.i += sim.substeps;
    }
}

fn step(sim: &Sim, state: &mut SimState, calc_potential: bool) -> (f32, f32) {
    let SimState { xs, vs, fs, x2s, spacial_lookup, start_indicies } = state;
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

    (0..).zip(spacial_lookup.iter_mut().zip(xs.iter())).for_each(
        |(i, (sl, &x))| {
            let cell = get_cell(x, sim);
            let hash = get_hash(cell, sim);
            let key = get_key(hash, sim);
            *sl = (i, key);
        },
    );

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
                let hash = get_hash(c, sim);
                let key = get_key(hash, sim);
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

                        if r2 == 0. {
                            // println!("");
                            // panic!("{i} {j}")
                        }

                        // assert!(r2 != 0.0);

                        let r = r2.sqrt();

                        let ff = dr / r * force(r);

                        f2 -= ff;
                        if calc_potential {
                            pot2 += energy(r);
                        }
                        pressure2 += ff.dot(dr);
                    });

                // *f = cell.as_vec3();
                f1 += f2;
                pot1 += pot2;
                pressure1 += pressure2;
            }
            *f = f1; // + 10. * x1 / x1.length_squared();
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

const EPS: f32 = 1.0;

fn energy(dr: f32) -> f32 {
    // sigma = 1.0
    4.0 * EPS * (dr.powi(-12) - dr.powi(-6))
}
fn force(dr: f32) -> f32 {
    48.0 * EPS * (-dr.powi(-13) + 0.5 * dr.powi(-7))
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

fn reset_on_key(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<Sim>,
) {
    if input.just_pressed(KeyCode::KeyR) {
        // reset_sim(&mut commands);
        sim.update();
        commands.insert_resource(SimState::new(&sim));
        commands.insert_resource(Stats::default());
    }

    if input.just_pressed(KeyCode::KeyD) {
        sim.draw ^= true;
    }

    if input.just_pressed(KeyCode::KeyP) {
        sim.pause ^= true;
    }
}

fn reset_sim(
    // sim: Res<Sim>,
    commands: &mut Commands,
    // mut query: Query<Entity, With<ParticleMarker>>,
    // mut gizmos: Gizmos,
    // input: Res<ButtonInput<KeyCode>>,
) {
    // if input.just_pressed(KeyCode::KeyR) {
    // commands.insert_resource(Stats::default());
    // let sim = Sim::default();
    // commands.insert_resource(SimState::new(&sim));
    // commands.insert_resource(sim);
    // }
    // for p in &query {
    //     commands.entity(p).despawn();
    // }

    // for _ in 0..sim.n {
    //     commands.spawn()
    // }

    // query.
    // println!("{}", sim.n)
}

fn draw(
    mut commands: Commands,
    // sim: Res<Context>,
    sim: Res<Sim>,
    state: Res<SimState>,
    mut query: Query<(Entity, &mut Transform, &ParticleMarker)>,
    mut gizmos: Gizmos,
    particle_mesh: Res<ParticleMesh>,
) {
    if sim.draw {
        if sim.draw_meshes {
            let it = query.iter_mut();
            if it.len() != sim.n as usize {
                for p in it {
                    commands.entity(p.0).despawn();
                }
                for i in 0..sim.n {
                    commands.spawn((
                        Mesh3d(particle_mesh.mesh.clone()),
                        MeshMaterial3d(particle_mesh.mat.clone()),
                        ParticleMarker(i as usize),
                    ));
                }
            } else {
                query.iter_mut().for_each(|(_, mut transform, marker)| {
                    transform.translation = state.xs[marker.0];
                });
            }
        } else {
            for (&x, &f) in state.xs.iter().zip(state.fs.iter()) {
                let c = Color::srgb_from_array(((f + 2.) / 4.).to_array());
                if sim.is_3d {
                    gizmos
                        .sphere(Isometry3d::from_translation(x), 1.0, c)
                        .resolution(10);
                } else {
                    const OFFSETS2: [Vec3; 4] = [
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(-1.0, 0.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                        Vec3::new(0.0, -1.0, 0.0),
                        // Vec3::new(0.0, 0.0, 0.0),
                    ];
                    if sim.draw_mirrors {
                        for off in OFFSETS2 {
                            gizmos
                                .circle_2d(
                                    Isometry2d::from_translation(
                                        (x + off * sim.size).xy(),
                                    ),
                                    1.0,
                                    Color::WHITE,
                                )
                                .resolution(10);
                        }
                    }
                    gizmos
                        .circle_2d(Isometry2d::from_translation(x.xy()), 1.0, c)
                        .resolution(10);

                    gizmos.rect(
                        Isometry3d::IDENTITY,
                        Vec2::splat(sim.size),
                        Color::WHITE,
                    );
                    gizmos.grid_2d(
                        Isometry2d::IDENTITY,
                        UVec2::splat((sim.size / sim.grid).ceil() as u32),
                        Vec2::splat(sim.grid),
                        Color::WHITE,
                    );
                }
            }
        }
    }

    gizmos.cuboid(
        Transform::IDENTITY.with_scale(Vec3::splat(sim.size)),
        Color::WHITE,
    );

    // gizmos.cuboid(
    //     Transform::IDENTITY.with_translation(Vec3::splat(sim.size / 2.)),
    //     Color::WHITE,
    // );

    // gizmos.cuboid(
    //     Transform::IDENTITY.with_translation(-Vec3::splat(sim.size / 2.)),
    //     Color::WHITE,
    // );
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut contexts: EguiContexts,
) {
    let ctx = contexts.ctx_mut();
    ctx.style_mut(|style| {
        style.text_styles.iter_mut().for_each(|(_text_style, font_id)| {
            font_id.size = 14.0;
        });
    });

    // reset_sim(&mut commands);

    commands.insert_resource(Stats::default());
    let sim = Sim::default();
    commands.insert_resource(SimState::new(&sim));

    if DRAW {
        commands.spawn((
            Camera3d::default(),
            // Projection::Orthographic(OrthographicProjection {
            //     scale: 0.05,
            //     ..OrthographicProjection::default_3d()
            // }),
            Transform::from_xyz(0.0, 50.0, 40.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
            MyCamera,
        ));

        let mesh = if sim.is_3d {
            meshes.add(Sphere::new(0.5).mesh().ico(1).unwrap())
        } else {
            meshes.add(Circle::new(0.5))
        };

        let mut m =
            StandardMaterial::from_color(Color::srgba_u8(124, 144, 255, 255));
        // m.unlit = true;
        let material = materials.add(m);

        commands.spawn((
            DirectionalLight { shadows_enabled: true, ..default() },
            Transform::from_xyz(0.0, sim.size * 2.0, sim.size)
                .looking_at(Vec3::ZERO, Vec3::Y),
        ));

        commands.insert_resource(ParticleMesh { mesh, mat: material });
    }
    commands.insert_resource(sim);
}

#[derive(Resource)]
struct ParticleMesh {
    mesh: Handle<Mesh>,
    mat: Handle<StandardMaterial>,
}

#[derive(Component)]
struct ParticleMarker(usize);

#[derive(Component)]
struct MyCamera;

// fn on_drag_rotate(drag: Trigger<Pointer<Drag>>, mut transforms: Query<&mut
// Transform, With<MyCamera>>) {     if let Ok(mut transform) =
// transforms.get_mut(drag.entity()) {         transform.rotate_y(drag.delta.x *
// 0.02);         transform.rotate_x(drag.delta.y * 0.02);
//     }
// }

fn orbit_camera(
    mut camera: Single<&mut Transform, With<MyCamera>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    mouse_scroll: Res<bevy::input::mouse::AccumulatedMouseScroll>,
    sim: Res<Sim>,
) {
    if mouse_buttons.pressed(MouseButton::Left) {
        let delta = mouse_motion.delta * -0.02;
        let (mut yaw, mut pitch, roll) =
            camera.rotation.to_euler(EulerRot::YXZ);
        yaw += delta.x;
        let lim = std::f32::consts::FRAC_PI_2 - 0.01;
        pitch = (pitch + delta.y).clamp(-lim, lim);
        camera.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
        camera.translation = Vec3::ZERO
            - camera.forward()
                * camera.translation.length()
                * (1.0 - mouse_scroll.delta.y * 0.1);
    }
}
