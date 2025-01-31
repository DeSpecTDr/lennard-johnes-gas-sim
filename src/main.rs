#![allow(unused)]

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use egui_plot::{PlotPoint, PlotPoints};
use ops::FloatPow;
use rayon::prelude::*;

fn main() {
    let sim = Sim::default();
    App::new()
        .add_plugins((
            DefaultPlugins,
            bevy::diagnostic::LogDiagnosticsPlugin::default(),
            bevy::diagnostic::FrameTimeDiagnosticsPlugin,
            bevy_framepace::FramepacePlugin,
            EguiPlugin,
            // bevy_framepace::debug::DiagnosticsPlugin,
        ))
        .insert_resource(bevy_framepace::FramepaceSettings {
            limiter: bevy_framepace::Limiter::from_framerate(RENDER_FPS),
        })
        .add_systems(Startup, setup)
        // .add_systems(Update, reset_sim.run_if(resource_changed::<Sim>))
        .add_systems(
            Update,
            (draw, orbit_camera, simple_sim, reset_on_key, ui_system),
        )
        .run();
}

const RENDER_FPS: f64 = 30.0;

#[derive(Resource)]
struct Stats {
    real_time: VecDeque<(f32, f32)>,
    sim_stat: VecDeque<Stat>,
}

#[derive(Default, Clone)]
struct Stat {
    t: f32,
    k_e: f32,
    p_e: f32,
    imp: f32,
}

impl Default for Stats {
    fn default() -> Self {
        const CAP: usize = 10000;
        Self {
            real_time: VecDeque::with_capacity(CAP),
            sim_stat: VecDeque::with_capacity(CAP),
        }
    }
}

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
        ui.label(format!(
            "{fps:.1} fps (limit: {RENDER_FPS})\nTime factor: {:.2} ({})\nParticles: {}\nSubsteps: \
             {}\nIterations/s: {}",
            sim.target_dt * fps as f32,
            if sim.pause { "paused" } else { "running" },
            sim.n,
            sim.substeps,
            (sim.substeps as f32 * fps as f32 / 100.0).round() * 100.0
        ));
        let v = &stats.sim_stat;
        if sim.enable_stats && !v.is_empty() {
            let last = v.back().unwrap();
            ui.label(format!(
                "Temperature: {:.2}",
                last.k_e * 2.0 / 3.0 / sim.n as f32
            ));

            // make_plot(ui, "t_e", make_rolling(v, |x| x.p_e + x.k_e, true),
            // 0.0); make_plot(ui, "imp", make_rolling(v, |x| x.imp,
            // false), 0.0);

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
                plot_ui.bar_chart(egui_plot::BarChart::new(make_bar_chart(
                    &state.vs,
                )));
            });
        }
    });
}

fn make_bar_chart(vs: &[Vec3]) -> Vec<egui_plot::Bar> {
    use egui_plot::Bar;
    let mut vs: Vec<f32> = vs.iter().map(|x| x.x.squared()).collect();
    vs.sort_unstable_by(|a, b| a.total_cmp(b));
    let s = 30.0;
    let mut bars: Vec<Bar> = Vec::new();
    let mut it = vs.iter();
    'a: for i in 0..20 {
        let mut c = 0;
        for v in it.by_ref() {
            if *v > (i + 1) as f32 * s {
                break;
            }
            c += 1;
        }
        let lnc = if c == 0 { 0.0 } else { (c as f64).ln() };
        bars.push(Bar::new((0.5 + i as f64) * s as f64, lnc).width(s as f64));
    }
    bars
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
    v_start: f32,
    // walls: bool
    n_spacial: u32,
    n: u32,
    n_grid: u32,
    dt: f32,
    half_dt: f32,
}

impl Default for Sim {
    fn default() -> Self {
        let mut s = Self {
            is_3d: true,
            n_row: 7,
            size: 40.0,
            grid: 4.0,
            // target_dt: 1.0 / RENDER_FPS as f32,
            target_dt: 1. / RENDER_FPS as f32,
            substeps: 40,
            draw: true,
            draw_meshes: false,
            disable_speed: false,
            ignore_cells: false,
            draw_mirrors: false,
            enable_stats: true,
            pause: false,
            v_start: 20.0,

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
    spacial_lookup: Box<[(u32, u32)]>,
    start_indicies: Box<[u32]>,
}

impl SimState {
    fn new(sim: &Sim) -> Self {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let n = sim.n;
        let mut r = StdRng::seed_from_u64(0);
        // TODO: use normal distr with sigma = sqrt(temperature)
        let mut get_r = || r.gen_range(-1.0..=1.0);
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
                            ) * sim.v_start
                        }
                    })
                    .collect();

                let imp = vs.iter().sum::<Vec3>() / n as f32;
                vs.iter_mut().for_each(|x| *x -= imp);

                vs
            },
            fs: vec![Vec3::ZERO; n as usize].into_boxed_slice(),
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
        }
        if sim.enable_stats {
            let p_e = step(&sim, &mut state, true);
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
            v.push_back(Stat { t, k_e, p_e, imp });
            // push(&mut stats.sim_time, t);
            // push(&mut stats.k_e, k_e);
            // push(&mut stats.p_e, p_e);
            // push(&mut stats.t_e, k_e + p_e);
            // push(&mut stats.imp, imp);
        } else {
            step(&sim, &mut state, false);
        }
    }
}

fn step(sim: &Sim, state: &mut SimState, calc_potential: bool) -> f32 {
    let SimState { xs, vs, fs, spacial_lookup, start_indicies } = state;

    // leapfrog1 TODO: do i mirror the points here?

    for (x, v) in xs.iter_mut().zip(vs.iter()) {
        *x += v * sim.half_dt;
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

    let potential = fs
        .par_iter_mut()
        .zip(xs.par_iter())
        .enumerate()
        .map(|(i, (f, &x1))| {
            let cell = get_cell(x1, sim);
            let mut f1 = Vec3::ZERO;
            let mut pot1 = 0.0;
            for off in OFFSETS {
                let c = cell + off;
                let hash = get_hash(c, sim);
                let key = get_key(hash, sim);
                let start_i = start_indicies[key as usize] as usize;

                let mut f2 = Vec3::ZERO;
                let mut pot2 = 0.0;
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
                        const CUTOFF: f32 = 4.0; // 4 SIGMA

                        if r2 > CUTOFF * CUTOFF {
                            return;
                        }

                        if r2 == 0. {
                            // println!("");
                            // panic!("{i} {j}")
                        }

                        // assert!(r2 != 0.0);

                        let r = r2.sqrt();

                        f2 -= dr / r * force(r);
                        if calc_potential {
                            pot2 += energy(r);
                        }
                    });

                // *f = cell.as_vec3();
                f1 += f2;
                pot1 += pot2;
            }
            *f = f1; // + 10. * x1 / x1.length_squared();
            pot1
        })
        .sum::<f32>();

    for ((x, v), f) in xs.iter_mut().zip(vs.iter_mut()).zip(fs.iter()) {
        *v += f * sim.dt;
        *x += *v * sim.half_dt;
        *x -= (*x / sim.size * 2.0).trunc() * sim.size;
    }

    // added twice per particle
    potential / 2.0
}

const EPS: f32 = 2.0;

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

    commands.spawn((
        Camera3d::default(),
        // Projection::Orthographic(OrthographicProjection {
        //     scale: 0.05,
        //     ..OrthographicProjection::default_3d()
        // }),
        Transform::from_xyz(0.0, 50.0, 40.0).looking_at(Vec3::ZERO, Vec3::Y),
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
