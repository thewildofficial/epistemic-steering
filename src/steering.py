"""Steering vector computation and intervention.

Applies activation steering to modify LLM behavior based on
epistemic confidence signals from probe scoring.

Core functions:
- compute_steering_vector: Direction vector for intervention
- apply_steering: Modify hidden states during forward pass
- batch_steering: Apply steering to batch of inputs
- scale_steering: Adjust steering magnitude
"""