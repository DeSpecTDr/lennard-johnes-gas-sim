use std::collections::VecDeque;

use accurate::{sum::Kahan, traits::SumAccumulator};
use glam::Vec3;

pub struct Mean {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
    pub n: u64,
}

impl Default for Mean {
    fn default() -> Self {
        Self::new()
    }
}

impl Mean {
    pub fn new() -> Self {
        Self { avg: 0., max: f32::MIN, min: f32::MAX, n: 0 }
    }

    pub fn add(&mut self, x: f32) {
        self.n += 1;
        self.avg += (x - self.avg) / self.n as f32;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }
}

pub struct SlidingMean {
    pub avg: f32,
    arr: VecDeque<f32>,
}

impl SlidingMean {
    pub fn new(cap: usize) -> Self {
        assert_ne!(cap, 0);
        Self { avg: 0., arr: VecDeque::with_capacity(cap) }
    }

    pub fn add(&mut self, x: f32) {
        if self.arr.len() == self.arr.capacity() {
            let first = self.arr.pop_front().unwrap();
            self.avg += (self.avg - first) / self.arr.len() as f32;
        }
        self.arr.push_back(x);
        self.avg += (x - self.avg) / self.arr.len() as f32;
    }

    pub fn add2(&mut self, x: f32) {
        if self.arr.len() == self.arr.capacity() {
            self.arr.pop_front().unwrap();
        }
        self.arr.push_back(x);
        self.avg = self.calc();
    }

    pub fn calc(&self) -> f32 {
        self.arr.iter().copied().ksum() / self.arr.len() as f32
    }
}

pub trait EasySum<F> {
    fn ksum(self) -> F;
}

impl<I> EasySum<f32> for I
where
    I: IntoIterator<Item = f32>,
{
    fn ksum(self) -> f32 {
        Kahan::<f32>::zero().absorb(self).sum()
    }
}

impl EasySum<Vec3> for &[Vec3] {
    fn ksum(self) -> Vec3 {
        let zero = Kahan::<f32>::zero();
        Vec3::from(
            self.iter()
                .fold([zero, zero, zero], |acc, x| {
                    [acc[0] + x.x, acc[1] + x.y, acc[2] + x.z]
                })
                .map(|x| x.sum()),
        )
    }
}
