// TODO: hooks for calc

use core::f32;
use std::collections::VecDeque;

use average::{Estimate, Variance};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use egui_plot::{PlotPoint, PlotPoints};
use itertools::izip;
use rayon::prelude::*;

const CUTOFF: f32 = 2.5; // 4

const N_BARS: usize = 20;
const BARS_S: f64 = 20.0;

#[derive(Resource)]
pub struct SimController {
    pub params: Sim,
    pub state: SimState,
}

impl SimController {
    pub fn new(mut sim: Sim) -> Self {
        sim.update();

        println!("rho: {}", sim.n as f32 * sim.size.powi(-3));

        // assert_ne!(sim.substeps, 0);
        Self { params: sim.clone(), state: SimState::new(&sim) }
    }

    pub fn prepare(&mut self) {
        let sim = &self.params;
        let mut state = &mut self.state;

        // TODO: check if maxwell-like

        let mut i = 1;
        let prev_e: f32 = 0.0;
        let (p_e, _) = step(sim, state, true);
        let e0 = state.k_e() + p_e;
        let e = e0;
        println!("i: {} e: {e}", state.i);

        let mut prev_e = 0.;
        let mut avg_e = crate::utils::SlidingMean::new(5);
        loop {
            // let n = i * 100;
            // let mut e = 0.;
            let mut e = crate::utils::Mean::new();
            for _ in 0..500 {
                let (p_e, _) = step(sim, state, true);
                // e += p_e + state.k_e();
                e.add(p_e + state.k_e());
            }
            avg_e.add(e.avg);
            // avg_e.add()
            // e /= n as f32;

            // println!("i: {} e: {} mi: {} ma: {}", state.i, e.avg, e.min,
            // e.max);
            println!("i: {} e: {} ({})", state.i, avg_e.avg, avg_e.calc());
            // let (p_e, _) = step(sim, state, true);
            // let e = e.mean() as f32;
            // let e = p_e + state.k_e();

            if (prev_e - avg_e.avg).abs() / avg_e.avg < 0.0001 {
                break;
            }
            prev_e = avg_e.avg;
            i += 1;
        }
    }

    pub fn step(&mut self, calc_stats: bool) -> (f32, f32) {
        step(&self.params, &mut self.state, calc_stats)
    }

    pub fn get_t(&self) -> f32 {
        self.state.i as f32 * self.params.dt
    }

    pub fn simple_sim(&mut self) {
        // let sim = &self.params;
        // let mut state = &mut self.state;

        // if sim.substeps > 0 {
        //     for _ in 1..sim.substeps {
        //         step(sim, state, false);

                // let vels0 = &mut stats.vels0;
                // if vels0.len() == vels0.capacity() {
                //     vels0.pop_front();
                // }
                // vels0.push_back(state.vs.clone());
            // }
            // let (p_e, pressure) = step(sim, state, true);
            // let v = &mut stats.sim_stat;
            // let t = v.back().map(|x| x.t).unwrap_or(0.0) + sim.target_dt;
            // // stats.sim_time += sim.target_dt as f64;
            // // let t = stats.sim_time;
            // let k_e: f32 = (state
            //     .vs
            //     .iter()
            //     .map(|x| x.length_squared() as f64)
            //     .sum::<f64>()
            //     / 2.0) as f32;
            // let imp: f32 = state.vs.iter().sum::<Vec3>().length();
            // if v.len() == v.capacity() {
            //     v.pop_front();
            // }
            // let dx2 = state.x2s.iter().map(|x|
            // x.length_squared()).sum::<f32>()     / sim.n as f32;
            // v.push_back(Stat { t, k_e, p_e, imp, pressure, dx2 });

            // let vels = &mut stats.vels;
            // if vels.len() == vels.capacity() {
            //     vels.pop_front();
            // }

            // let mut vs: Vec<f32> =
            //     state.vs.iter().map(|x| x.x.powi(2)).collect();
            // vs.sort_unstable_by(|a, b| a.total_cmp(b));
            // // let s = BARS_S as f32;
            // let s = vs.iter().sum::<f32>() / sim.n as f32 / 2.0;
            // let mut bars: Vec<(f32, f32)> = Vec::new();
            // let mut it = vs.iter();
            // 'a: for i in 0..N_BARS {
            //     let mut c = 0;
            //     for v in it.by_ref() {
            //         if *v > (i + 1) as f32 * s {
            //             break;
            //         }
            //         c += 1;
            //     }
            //     let lnc = if c == 0 { 0.0 } else { (c as f32).ln() };
            //     bars.push((s * i as f32, lnc));
            // }
            // vels.push_back(bars.into_boxed_slice());

            // stats.i += sim.substeps;
        // }
    }
}

#[derive(Clone)]
pub struct Sim {
    pub n_row: u32,
    pub size: f32,
    pub dt: f32,
    pub temp0: f32,
    pub disable_speed: bool,
    pub ignore_cells: bool,

    pub grid: f32,
    pub n_spacial: u32,
    pub n: u32,
    pub n_grid: u32,
    pub half_dt: f32,
}

// free params: NVT
impl Default for Sim {
    fn default() -> Self {
        let mut s = Self {
            // n_row: 8,
            n_row: 15,
            size: 15.0,
            dt: 0.005,
            temp0: 1.0,
            disable_speed: false,
            ignore_cells: false,

            grid: 0.0,
            n_spacial: 0,
            n: 0,
            n_grid: 0,
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
        self.half_dt = self.dt / 2.;
    }
}

pub struct SimState {
    pub i: i32,
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
            i: 0,
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
                let v2target = sim.temp0 * 3.0 * sim.n as f32;
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

    pub fn k_e(&self) -> f32 {
        (self.vs.iter().map(|x| x.length_squared() as f64).sum::<f64>() / 2.0)
            as f32
    }

    #[allow(non_snake_case)]
    pub fn T(&self) -> f32 {
        self.k_e() * 2.0 / 3.0 / self.xs.len() as f32
    }

    pub fn p(&self) -> Vec3 {
        self.vs.iter().sum::<Vec3>()
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

fn step(sim: &Sim, state: &mut SimState, calc_stats: bool) -> (f32, f32) {
    let SimState { i, xs, vs, fs, x2s, spacial_lookup, start_indicies } = state;
    *i += 1;

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

    spacial_lookup.sort_unstable_by_key(|x| x.1);

    // spacial_lookup.par_sort_unstable_by_key(|x| x.1);

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
