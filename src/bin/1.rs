use std::{
    ops::Deref,
    sync::{RwLock, atomic::AtomicU32},
};

use accurate::sum::{Kahan, traits::*};
use bevy_playground as my;
use inline_python::{Context, python};
use my::sim2::*;
use nalgebra as na;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

struct Hist {
    pub arr: Vec<u32>,
    size: f32,
}

impl Hist {
    fn new(size: f32) -> Self {
        Self { arr: Vec::new(), size }
    }

    fn n(&self) -> u32 {
        self.arr.iter().sum()
    }

    fn add(&mut self, x: f32) {
        let i = (x / self.size) as usize;

        if i + 1 > self.arr.len() {
            self.arr.resize(i + 1, 0);
            self.arr[i] = 1;
        } else {
            self.arr[i] += 1;
        }
    }

    fn extend(&mut self, arr: &[f32]) {
        arr.iter().for_each(|&x| self.add(x));
    }

    fn log(&self) -> Vec<f32> {
        let n = self.n() as f32;
        self.arr.iter().map(|&x| (x as f32 / n).ln()).collect()
    }

    fn val(&self) -> Vec<f32> {
        let n = self.n() as f32;
        self.arr.iter().map(|&x| x as f32 / n).collect()
    }

    fn bins(&self) -> Vec<f32> {
        (0..self.arr.len()).map(|x| (x as f32 + 0.5) * self.size).collect()
    }

    fn raw(&self) -> Vec<u32> {
        self.arr.clone()
    }
}

struct Plot {
    ctx: Context,
}

impl Deref for Plot {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl Plot {
    fn new() -> Self {
        Self {
            ctx: python! {
                import numpy as np
                import matplotlib.pyplot as plt
                import matplotlib as mpl
                import scipy.stats
                mpl.rcParams["figure.autolayout"] = True
                print(mpl.get_backend())
                mpl.use("Agg")

                // fig, axs = plt.subplots(squeeze=False)
            },
        }
    }

    fn save(&self, p: &str) {
        self.ctx.run(python! {
            plt.savefig('p), plt.close()
        });
    }

    fn show(&self) {
        self.ctx.run(python! {
            plt.show()
        });
    }

    fn plot(&self, x: &[f32], y: &[f32]) {
        self.ctx.run(python! {
            plt.plot('x, 'y, "."), plt.show()
        });
    }
}

pub trait EasySum {
    fn ksum(self) -> f32;
}

impl<I> EasySum for I
where
    I: IntoIterator<Item = f32>,
{
    fn ksum(self) -> f32 {
        Kahan::<f32>::zero().absorb(self).sum()
    }
}

fn main() {
    let mut sim = SimController::new(Sim::default());
    println!("T0: {}", sim.state.T());
    let mut Ts = vec![];
    let mut ps1 = vec![];
    let mut ps2 = vec![];
    let mut ps3 = vec![];
    let mut es = vec![];
    let mut ts = vec![];
    for i in 0..2 {
        println!("t: {i}");
        for _ in 0..1000 {
            let (p_e, _) = sim.step(true);
            let e = p_e + sim.state.k_e();
            let p = sim.state.p();
            ps1.push(p.x);
            ps2.push(p.y);
            ps3.push(p.z);
            es.push(e);
            Ts.push(sim.state.T());
            ts.push(sim.get_t());
        }
    }

    let plt = Plot::new();
    plt.plot(&ts, &es);
    plt.save("test.png");

    plt.plot(&ts, &Ts);

    // plt.plot(&ts, &ps1);
    // plt.plot(&ts, &ps2);
    // plt.plot(&ts, &ps3);
    plt.save("test1.png");
}

fn rdf() {
    let mut sim = SimController::new(Sim::default());
    sim.prepare();

    let n = sim.params.n as f32;

    let irq = sim.params.size / 2.;

    let meas_n = 300;

    let mut h = Hist::new(
        meas_n as f32
            * 2.
            * 4.
            * irq
            * ((sim.params.n as f32).powi(6) / 4.).powf(-1. / 3.),
    ); // 0.01
    // h.extend(); # 290kk
    let l = sim.state.xs.len();

    for i in 0..meas_n {
        println!("{i}");
        for _ in 0..10 {
            sim.step(false);
        }

        sim.state.xs.iter().enumerate().for_each(|(i, x1)| {
            for j in 0..(l - i - 1) {
                if i == j {
                    continue;
                }
                let x2 = sim.state.xs[j];

                let mut dr = x1 - x2;
                dr -= (dr / sim.params.size).round() * sim.params.size;
                let r = dr.length_squared();
                if r < sim.params.size.powi(2) / 28. {
                    h.add(r.sqrt());
                }
            }
        });

        // for i in 0..l {

        // }
    }
    // println!("histn: {}", h.n());

    // let irq = sim.state.vs.iter().map(|x|
    // x.length_squared()).sum_with_accumulator::<Kahan<_>>() / n; let mut h
    // = Hist::new(2. * irq * n.powf(-1./3.)); let mut h1 = Hist::new(2. *
    // irq * n.powf(-1./3.));

    //     let vels = sim.state.vs.iter().map(|x|
    // x.x.powi(2)).collect::<Vec<_>>(); h1.extend(&vels);

    // for i in 0..100 {
    //     println!("{i}");
    //     for _ in 0..100 {
    //         sim.step(false);
    //     }
    //     h.extend(&sim.state.vs.iter().map(|x|
    // x.x.powi(2)).collect::<Vec<_>>());

    // }

    // let vels = h.log();
    // let vels1 = h1.log();
    // let bins = h.bins();
    // let bins1 = h1.bins();

    // average::Histogram

    let b = h.bins();
    // let v = h.val();

    let rho = sim.params.n as f32 * sim.params.size.powi(-3);

    let v: Vec<f32> = h
        .raw()
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let gfac = 4. / 3. * std::f32::consts::PI * h.size.powi(3);
            let vb = gfac * ((3 * i + 3) * i + 1) as f32;
            let mut g = x as f32 / sim.params.n as f32 / vb / rho;
            g *= 2.; // twice per particle
            g /= meas_n as f32;
            g
        })
        .collect();

    let c = Plot::new();

    c.plot(&b, &v);
    c.show();

    c.save("plot.png");
}
