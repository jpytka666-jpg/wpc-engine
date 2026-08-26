use noworodek::{evaluate_math_model, generate_math_dataset, MathDataset};

fn train_epoch(weights: &mut [f32; 2], dataset: &MathDataset, learning_rate: f32) -> f32 {
    let mut loss_sum = 0.0f32;
    for s in &dataset.train {
        let pred = s.a * weights[0] + s.b * weights[1];
        let err = pred - s.target;
        loss_sum += err * err;
        let scale = 2.0 / dataset.train.len() as f32;
        weights[0] -= learning_rate * scale * err * s.a;
        weights[1] -= learning_rate * scale * err * s.b;
    }
    loss_sum / dataset.train.len() as f32
}

fn main() {
    let dataset = generate_math_dataset();
    let mut weights = [0.0f32, 0.0f32];

    let before = evaluate_math_model(&weights, &dataset.held_out).expect("valid math model");
    println!("NOWORODEK MATH TRAIN");
    println!("BEFORE mse={:.8} exact={:.4}", before.mse, before.exact_accuracy);

    let mut last_train_loss = f32::INFINITY;
    for step in 0..2000usize {
        last_train_loss = train_epoch(&mut weights, &dataset, 0.01);
        if step % 250 == 249 {
            let eval = evaluate_math_model(&weights, &dataset.held_out).expect("valid math model");
            println!("step={} train_mse={:.8} heldout_mse={:.8} heldout_exact={:.4} w=[{:.6},{:.6}]", step + 1, last_train_loss, eval.mse, eval.exact_accuracy, weights[0], weights[1]);
        }
    }

    let after = evaluate_math_model(&weights, &dataset.held_out).expect("valid math model");
    println!("AFTER mse={:.8} exact={:.4}", after.mse, after.exact_accuracy);
    println!("RESULT train_loss={:.8} learned_weights=[{:.6},{:.6}]", last_train_loss, weights[0], weights[1]);

    assert!(after.mse < before.mse, "held-out MSE did not improve");
}
