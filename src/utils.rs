use std::collections::VecDeque;

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
        use accurate::traits::*;

        // self.arr.iter().sum::<f32>() / self.arr.len() as f32

        self.arr
            .iter()
            .copied()
            .sum_with_accumulator::<accurate::sum::Kahan<_>>()
            / self.arr.len() as f32
    }
}
