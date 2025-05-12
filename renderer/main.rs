#![allow(unused)]

use core::f32;
use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_playground::sim;
use egui_plot::{PlotPoint, PlotPoints};
use itertools::izip;
use ops::FloatPow;
use rayon::prelude::*;
use sim::SimController;

const DRAW: bool = true;
const DRAW_MESHES: bool = true;
const RENDER_FPS: f64 = 30.0;

#[derive(Resource)]
struct Two {
    sim2: sim::SimController,
}

fn main() {
    // let mut sim2 = sim::SimController::new(sim::Sim { substeps: 20, ..default() });

    // sim1.simple_sim();
    // sim2.simple_sim();

    // println!("{}", sim1.state.xs.iter().zip(sim2.state.xs.iter()).map(|(x1, x2)| (x2 - x1).length_squared()).sum::<f32>() / sim1.params.n as f32);

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
        bevy_mod_outline::OutlinePlugin,
        // bevy_framepace::debug::DiagnosticsPlugin,
    ))
    .insert_resource(bevy_framepace::FramepaceSettings {
        limiter: bevy_framepace::Limiter::from_framerate(RENDER_FPS),
    })
    .add_systems(Startup, (setup_ui, setup_draw))
    .add_systems(
        Update,
        (
            draw,
            orbit_camera, // a
            do_steps,
            ui_system
        ),
    );
    // // .add_systems(Update, reset_sim.run_if(resource_changed::<Sim>))
    // // .add_systems(Update, (simple_sim, reset_on_key, ui_system));
    app.run();
}

fn ui_system(
    mut contexts: EguiContexts,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    sim_ctrl: Res<SimController>,
    localx: Local<usize>,
) {
    let sim = &sim_ctrl.params;
    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|f| f.average())
        .unwrap_or(0.);

    let ctx = contexts.ctx_mut();
    egui::Window::new("Stats").show(ctx, |ui| {
        let v = sim_ctrl.stats.sim_stat.back().cloned().unwrap_or_default();
        let temperature = v.k_e * 2.0 / 3.0 / sim.n as f32;

        ui.label(format!("FPS: {fps:.1}\nIter: {}\nT: {temperature}\nrho: {}", sim_ctrl.state.i, sim.n as f32 / sim.size.powi(3)));

        // let sim_stat = &sim_ctrl.stats.sim_stat;
        let Some(vels) = sim_ctrl.stats.vels.iter().last() else {
            return;
        };

        // egui_plot:: Plot::new("maxwell_plot").show(ui, |plot_ui| {
        //     plot_ui.points(egui_plot::Points::new(vels.iter().map(|x| [x.0 as f64, x.1 as f64]).collect::<Vec<_>>()));
        // });

        // egui_plot:: Plot::new("r2t").height(400.0).show(ui, |plot_ui| {
        //     plot_ui.points(egui_plot::Points::new(sim_ctrl.stats.r2t.iter().enumerate().map(|x| [(x.0 as f32 * sim.dt) as f64, *x.1 as f64]).collect::<Vec<_>>()));
        // });

        // egui_plot:: Plot::new("v0vt").include_y(1.0).include_y(0.0).height(400.0).show(ui, |plot_ui| {
        //     let v0 = sim_ctrl.stats.v0vt.first().copied().unwrap_or(0.0);
        //     plot_ui.points(egui_plot::Points::new(sim_ctrl.stats.v0vt.iter().enumerate().map(|x| [(x.0 as f32 * sim.dt) as f64, (*x.1 / v0) as f64]).collect::<Vec<_>>()));
        // });

        egui_plot:: Plot::new("aaaaa").height(400.0).show(ui, |plot_ui| {
            plot_ui.points(egui_plot::Points::new(sim_ctrl.stats.diff_x.iter().enumerate().map(|x| [(x.0 as f32 * sim.target_dt) as f64, *x.1 as f64]).collect::<Vec<_>>()));
        });


        // egui_plot::Plot::new("plot")
        //     .allow_zoom(false)
        //     .allow_drag(false)
        //     .allow_scroll(false)
        //     .legend(egui_plot::Legend::default())
        //     .include_y(-0.01)
        //     .include_y(0.01)
        //     .height(400.0)
        //     .x_axis_label("simulation time (s)")
        //     // .include_y(65.0)
        //     .show_grid(true)
        //     .show(ui, |plot_ui| {
        //         plot_line(
        //             plot_ui,
        //             "total_energy/total_energy0",
        //             make_rolling(v, |x| x.p_e + x.k_e, true),
        //         );
        //         plot_line(
        //             plot_ui,
        //             "impulse (sigma/s)",
        //             make_rolling(v, |x| x.imp, false),
        //         );
        //     });
        // egui_plot::Plot::new("bar").height(400.0).show(ui, |plot_ui| {
        //     let mut bars = [0.0; N_BARS];
        //     stats.vels.iter().for_each(|x| {
        //         x.iter().zip(bars.iter_mut()).for_each(|(v, s)| {
        //             *s += v;
        //         })
        //     });

        //     let bars = bars
        //         .iter()
        //         .enumerate()
        //         .map(|(i, x)| {
        //             egui_plot::Bar::new(
        //                 (0.5 + i as f64) * BARS_S,
        //                 *x as f64 / stats.vels.len() as f64,
        //             )
        //             .width(BARS_S)
        //         })
        //         .collect();

        //     plot_ui.bar_chart(egui_plot::BarChart::new(bars));

            // plot_ui.bar_chart(egui_plot::BarChart::new(make_bar_chart(
            //     &state.vs,
            // )));
        // });
    });
}

fn do_steps(mut sim_ctrl: ResMut<SimController>) {
    sim_ctrl.simple_sim();
}

fn reset_on_key(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    mut sim_ctrl: ResMut<SimController>,
) {
    // if input.just_pressed(KeyCode::KeyR) {
    //     // reset_sim(&mut commands);
    //     // sim_ctrl.params.update();
    //     commands.insert_resource(SimState::new(&sim));
    //     commands.insert_resource(Stats::default());
    // }

    // if input.just_pressed(KeyCode::KeyD) {
    //     sim.draw ^= true;
    // }

    // if input.just_pressed(KeyCode::KeyP) {
    //     sim.pause ^= true;
    // }
}

// fn reset_sim(
//     // sim: Res<Sim>,
//     commands: &mut Commands,
//     // mut query: Query<Entity, With<ParticleMarker>>,
//     // mut gizmos: Gizmos,
//     // input: Res<ButtonInput<KeyCode>>,
// ) {
//     // if input.just_pressed(KeyCode::KeyR) {
//     // commands.insert_resource(Stats::default());
//     // let sim = Sim::default();
//     // commands.insert_resource(SimState::new(&sim));
//     // commands.insert_resource(sim);
//     // }
//     // for p in &query {
//     //     commands.entity(p).despawn();
//     // }

//     // for _ in 0..sim.n {
//     //     commands.spawn()
//     // }

//     // query.
//     // println!("{}", sim.n)
// }

fn draw(
    mut commands: Commands,
    // sim: Res<Context>,
    sim_ctrl: Res<SimController>,
    mut query: Query<(Entity, &mut Transform, &ParticleMarker)>,
    mut gizmos: Gizmos,
    particle_mesh: Res<ParticleMesh>,
) {
    let sim = &sim_ctrl.params;
    let state = &sim_ctrl.state;

    if DRAW_MESHES {
        let it = query.iter_mut();
        if it.len() != sim.n as usize {
            for p in it {
                commands.entity(p.0).despawn();
            }
            for i in 0..sim.n {
                use bevy_mod_outline::*;
                commands.spawn((
                    Mesh3d(particle_mesh.mesh.clone()),
                    MeshMaterial3d(particle_mesh.mat.clone()),
                    ParticleMarker(i as usize),
                    OutlineVolume {
                        visible: true,
                        colour: Color::srgba(1.0, 1.0, 1.0, 1.0),
                        width: 2.0,
                    },
                ));
            }
        } else {
            query.iter_mut().for_each(|(_, mut transform, marker)| {
                transform.translation = state.xs[marker.0];
            });
        }
    } else {
        let mean_f =
            state.fs.iter().map(|x| x.length()).sum::<f32>() / sim.n as f32;

        for (&x, &f) in state.xs.iter().zip(state.fs.iter()) {
            let ff = ((f.length() / mean_f).powi(2)).tanh() * 0.5 + 0.5;
            // let c = Color::srgb_from_array(((f + 2.) / 4.).to_array());
            let c = Color::srgb(ff, ff, ff);
            gizmos
                .sphere(Isometry3d::from_translation(x), 0.5, c)
                .resolution(10);
        }
    }

    gizmos.cuboid(
        Transform::IDENTITY.with_scale(Vec3::splat(sim.size)),
        Color::WHITE,
    );
}

fn setup_ui(mut contexts: EguiContexts) {
    let ctx = contexts.ctx_mut();
    ctx.style_mut(|style| {
        style.text_styles.iter_mut().for_each(|(_text_style, font_id)| {
            font_id.size = 14.0;
        });
    });
}

fn setup_draw(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        // Projection::Orthographic(OrthographicProjection {
        //     scale: 0.05,
        //     ..OrthographicProjection::default_3d()
        // }),
        Transform::from_xyz(0.0, 50.0, 40.0).looking_at(Vec3::ZERO, Vec3::Y),
        MyCamera,
    ));

    let mesh = meshes.add(Sphere::new(0.5).mesh().ico(1).unwrap());

    let c = Color::srgba_u8(124, 144, 255, 255);
    let mut m = StandardMaterial::from_color(c);
    // m.unlit = true;
    let material = materials.add(m);

    commands.spawn((
        DirectionalLight { shadows_enabled: true, ..default() },
        Transform::from_xyz(0.0, 2.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(ParticleMesh { mesh, mat: material });

    let sim_ctrl = sim::SimController::new(sim::Sim { ..default() });
    commands.insert_resource(sim_ctrl);
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

fn orbit_camera(
    mut camera: Single<&mut Transform, With<MyCamera>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    mouse_scroll: Res<bevy::input::mouse::AccumulatedMouseScroll>,
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
