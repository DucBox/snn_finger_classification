from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras_spiking
from tensorflow.keras.models import load_model


def run(
    model_path: str,
    n_steps:    int   = 5,
    dt:         float = 0.3,
):
    # Required when TF_DETERMINISTIC_OPS=1 — SpikingActivation uses tf.random.uniform internally
    tf.random.set_seed(42)

    print(f"\n[STEP 1/2] Loading SNN model: {model_path}")
    model = load_model(
        model_path,
        custom_objects={"SpikingActivation": keras_spiking.SpikingActivation},
    )
    model.summary()
    print()

    print(f"[STEP 2/2] Estimating energy consumption  (n_steps={n_steps}, dt={dt}) ...")
    example_data = [
        np.ones((32, n_steps, 64), dtype=np.float32),
        np.ones((32, n_steps, 1),  dtype=np.float32),
    ]
    energy = keras_spiking.ModelEnergy(model, example_data=example_data)

    print("\n--- Energy by device (Joules / inference) ---")
    energy.summary(
        columns=(
            "name",
            "rate",
            "energy cpu",
            "energy gpu",
            "energy loihi",
            "energy spinnaker",
            "energy spinnaker2",
            "energy arm",
        ),
        dt=dt,
        timesteps_per_inference=n_steps,
        print_warnings=False,
    )

    print("\n--- Synaptic vs neuron energy breakdown ---")
    energy.summary(
        columns=(
            "name",
            "energy cpu",
            "energy gpu",
            "synop_energy cpu",
            "synop_energy gpu",
            "neuron_energy cpu",
            "neuron_energy gpu",
        ),
        print_warnings=False,
    )

    print("[STEP 2/2] Done.")
