use criterion::{black_box, criterion_group, criterion_main, Criterion};
use koopman_dmd::*;

fn make_signal(n_vars: usize, n_time: usize) -> faer::Mat<f64> {
    let mut data = faer::Mat::<f64>::zeros(n_vars, n_time);
    for j in 0..n_time {
        let t = j as f64 * 0.05;
        for i in 0..n_vars {
            data[(i, j)] = ((i + 1) as f64 * t).sin() + 0.1 * ((i + 3) as f64 * t * 2.0).cos();
        }
    }
    data
}

fn bench_dmd(c: &mut Criterion) {
    let mut group = c.benchmark_group("dmd");

    for &(n_vars, n_time) in &[(5, 100), (10, 200), (20, 500), (50, 1000)] {
        let data = make_signal(n_vars, n_time);
        let config = DmdConfig::default();

        group.bench_function(format!("{n_vars}x{n_time}"), |b| {
            b.iter(|| dmd(black_box(&data), black_box(&config)).unwrap())
        });
    }

    group.finish();
}

fn bench_predict(c: &mut Criterion) {
    let data = make_signal(10, 200);
    let config = DmdConfig::default();
    let result = dmd(&data, &config).unwrap();

    let mut group = c.benchmark_group("predict");

    group.bench_function("modes_100", |b| {
        b.iter(|| predict_modes(black_box(&result), black_box(100), None).unwrap())
    });

    group.bench_function("matrix_100", |b| {
        b.iter(|| predict_matrix(black_box(&result), black_box(100), None).unwrap())
    });

    group.finish();
}

fn bench_hankel(c: &mut Criterion) {
    let mut group = c.benchmark_group("hankel_dmd");

    for &n_time in &[100, 500, 1000] {
        let mut data = faer::Mat::<f64>::zeros(1, n_time);
        for j in 0..n_time {
            let t = j as f64 * 0.05;
            data[(0, j)] = t.sin() + 0.5 * (3.0 * t).cos();
        }

        let config = HankelConfig {
            delays: Some(20),
            rank: Some(5),
            dt: 1.0,
        };

        group.bench_function(format!("1x{n_time}"), |b| {
            b.iter(|| hankel_dmd(black_box(&data), black_box(&config)).unwrap())
        });
    }

    group.finish();
}

fn bench_gla(c: &mut Criterion) {
    let mut data = faer::Mat::<f64>::zeros(2, 200);
    for j in 0..200 {
        let t = j as f64 * 0.1;
        data[(0, j)] = t.sin();
        data[(1, j)] = t.cos();
    }

    let config = GlaConfig {
        eigenvalues: None,
        n_eigenvalues: 3,
        tol: 1e-6,
        max_iter: None,
    };

    c.bench_function("gla_2x200_3eig", |b| {
        b.iter(|| gla(black_box(&data), black_box(&config)).unwrap())
    });
}

fn bench_mesochronic(c: &mut Criterion) {
    let map = StandardMap { epsilon: 0.12 };
    let obs = Observable::SinPi;

    c.bench_function("mesochronic_10x10_1000iter", |b| {
        b.iter(|| {
            mesochronic_compute(
                black_box(&map),
                (0.0, 1.0),
                (0.0, 1.0),
                black_box(10),
                black_box(&obs),
                0.1,
                black_box(1000),
            )
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_dmd,
    bench_predict,
    bench_hankel,
    bench_gla,
    bench_mesochronic
);
criterion_main!(benches);
