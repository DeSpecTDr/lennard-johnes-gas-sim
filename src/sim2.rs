// TODO: hooks for calc

use core::f32;
use std::{
    collections::VecDeque,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use average::{Estimate, Variance};
use glam::{IVec3, Vec3};
// use bevy::prelude::*;
// use bevy_egui::{EguiContexts, EguiPlugin, egui};
// use egui_plot::{PlotPoint, PlotPoints};
use itertools::izip;
use rayon::prelude::*;

use crate::utils::EasySum;

const CUTOFF: f32 = 2.5; // 4

// #[derive(Resource)]
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
        let state = &mut self.state;

        // TODO: check if maxwell-like

        let (p_e, _) = step(sim, state, true);
        let e0 = state.k_e() + p_e;
        println!("i: {} e: {} ({})", state.i, e0, state.T());

        let mut prev_e = 0.;
        let mut avg_e = crate::utils::SlidingMean::new(5);

        let mut nvt = true;

        loop {
            // let n = i * 100;
            // let mut e = 0.;
            let mut e = crate::utils::Mean::new();
            for _ in 0..5000 {
                let (p_e, _) = step(sim, state, true);
                // e += p_e + state.k_e();

                if nvt {
                    let lambda = (1.
                        + sim.dt / sim.tau * (sim.temp0 / state.T() - 1.))
                        .sqrt();
                    state.vs.iter_mut().for_each(|v| *v *= lambda);
                }
                e.add(p_e + state.k_e());
            }
            avg_e.add(e.avg);
            // avg_e.add()
            // e /= n as f32;

            // println!("i: {} e: {} mi: {} ma: {}", state.i, e.avg, e.min,
            // e.max);
            let T = state.T();
            println!("i: {} e: {} ({})", state.i, avg_e.avg, T);
            // let (p_e, _) = step(sim, state, true);
            // let e = e.mean() as f32;
            // let e = p_e + state.k_e();

            let dT = (T - sim.temp0).abs() / sim.temp0;

            if (prev_e - avg_e.avg).abs() / avg_e.avg < 0.0001 {
                if nvt {
                    nvt = false;
                } else {
                    break;
                }
            }

            if dT < 0.05 {
                nvt = false;
            }

            if dT > 0.1 {
                nvt = true;
            }

            prev_e = avg_e.avg;
        }
    }

    pub fn step(&mut self, calc_stats: bool) -> (f32, f32) {
        step(&self.params, &mut self.state, calc_stats)
    }

    pub fn get_t(&self) -> f32 {
        self.state.i as f32 * self.params.dt
    }

    pub fn rho(&self) -> f32 {
        self.params.n as f32 * self.params.size.powi(-3)
    }

    pub fn save_ovito(&self, p: impl AsRef<Path>) {
        use std::io::Write;

        let mut writer = BufWriter::new(File::create(p.as_ref()).unwrap());
        let s = self.params.size;
        let half = -s / 2.;
        writeln!(
            writer,
            "{}\nLattice=\"{} 0 0 0 {} 0 0 0 {}\" Origin=\"{} {} {}\" \
             Properties=pos:R:3:velo:R:3 Time = {}",
            self.params.n,
            s,
            s,
            s,
            half,
            half,
            half,
            self.get_t(),
        )
        .unwrap();

        for (x, v) in izip!(&self.state.xs, &self.state.vs) {
            writeln!(writer, "{} {} {} {} {} {}", x.x, x.y, x.z, v.x, v.y, v.z)
                .unwrap();
        }

        // writer
    }

    pub fn save(&self, p: impl AsRef<Path>) {
        let save = SaveSim {
            params: self.params.clone(),
            i: self.state.i,
            xs: self.state.xs.iter().map(|x| x.to_array()).collect(),
            vs: self.state.vs.iter().map(|x| x.to_array()).collect(),
        };

        let mut writer = BufWriter::new(File::create(p.as_ref()).unwrap());

        let nbytes = bincode::encode_into_std_write(
            save,
            &mut writer,
            bincode::config::standard(),
        )
        .unwrap();

        println!(
            "saved {} kB to {:?}",
            nbytes / 1024,
            p.as_ref().file_name().unwrap()
        );
    }

    pub fn load(p: impl AsRef<Path>) -> Self {
        let mut reader = BufReader::new(File::open(p.as_ref()).unwrap());
        let save: SaveSim = bincode::decode_from_std_read(
            &mut reader,
            bincode::config::standard(),
        )
        .unwrap();
        let n = save.params.n;
        let n_spacial = save.params.n_spacial;
        Self {
            params: save.params,
            state: SimState {
                i: save.i,
                xs: save.xs.into_iter().map(Vec3::from).collect(),
                vs: save.vs.into_iter().map(Vec3::from).collect(),
                fs: vec![Vec3::ZERO; n as usize].into_boxed_slice(),
                x2s: vec![Vec3::ZERO; n as usize].into_boxed_slice(),
                spacial_lookup: vec![(0, 0); n as usize].into_boxed_slice(),
                start_indicies: vec![u32::MAX; n_spacial as usize]
                    .into_boxed_slice(),
            },
        }
    }
}

#[derive(Clone, bincode::Encode, bincode::Decode)]
struct SaveSim {
    params: Sim,
    i: i32,
    xs: Box<[[f32; 3]]>,
    vs: Box<[[f32; 3]]>,
}

#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct Sim {
    pub n_row: u32,
    pub size: f32,
    pub dt: f32,
    pub temp0: f32,
    pub tau: f32,
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
            n_row: 14,
            size: 15.0,
            dt: 0.005,
            temp0: 1.0,
            tau: 0.3,
            disable_speed: false,
            ignore_cells: false,

            grid: 0.0,
            n_spacial: 0,
            n: 0,
            n_grid: 0,
            half_dt: 0.0,
        };
        // s.update();
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
        println!("grid size: {}", self.grid);

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
            xs: (0..n).map(|i| gen_point(i, sim.n_row, sim.size)).collect(),
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
                let v2sum: f32 = vs.iter().map(|x| x.length_squared()).ksum();
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
        // let a = (self.vs.iter().map(|x| x.length_squared() as
        // f64).sum::<f64>() / 2.0)     as f32;
        self.vs.iter().map(|x| x.length_squared()).ksum() / 2.0
    }

    #[allow(non_snake_case)]
    pub fn T(&self) -> f32 {
        self.k_e() * 2.0 / 3.0 / self.xs.len() as f32
    }

    pub fn p(&self) -> Vec3 {
        self.vs.ksum()
    }

    pub fn reset_x2(&mut self) {
        self.x2s.iter_mut().for_each(|x| *x = Vec3::ZERO);
    }
}

pub fn gen_point(i: u32, n_row: u32, size: f32) -> Vec3 {
    let s = n_row;
    let half = (s - 1) as f32 / 2.;

    let xi = i / s / s;
    let off: f32 = (xi % 2) as f32 * 0.5; // correction for high density

    let x = Vec3::new(
        xi as f32 - half,
        (i / s % s) as f32 - half + off,
        (i % s) as f32 - half + off,
    ) / s as f32
        * 2.;
    let x = x * size / 2.0;
    x - (x / size * 2.0).trunc() * size
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

fn step(sim: &Sim, state: &mut SimState, calc_stats: bool) -> (f32, f32) {
    let SimState { i, xs, vs, fs, x2s, spacial_lookup, start_indicies } = state;
    *i += 1;

    for (x, x2, v) in izip!(&mut *xs, &mut *x2s, &*vs) {
        let dx = v * sim.half_dt;
        *x += dx;
        *x2 += dx;
        *x -= (*x / sim.size * 2.0).trunc() * sim.size;
        if !x.is_finite() {
            println!("nan/inf detected");
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
