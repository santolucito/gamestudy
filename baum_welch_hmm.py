"""
Baum-Welch HMM Implementation using hmmlearn

This program demonstrates the Baum-Welch algorithm for learning Hidden Markov Model
parameters from observation sequences. Designed for analyzing behavioral data like
ARC puzzle solving traces.

The Baum-Welch algorithm (Expectation-Maximization for HMMs) iteratively:
1. E-step: Estimate hidden state probabilities given current parameters
2. M-step: Update parameters to maximize expected log-likelihood
"""

import numpy as np
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# =============================================================================
# 1. CATEGORICAL HMM - For discrete observations (actions, state transitions)
# =============================================================================

def train_categorical_hmm(observations, n_hidden_states=3, n_iter=100):
    """
    Train a Categorical HMM using Baum-Welch algorithm.

    Args:
        observations: List of observation sequences (each is array of integers)
        n_hidden_states: Number of hidden states to learn
        n_iter: Maximum Baum-Welch iterations

    Returns:
        Trained HMM model
    """
    # Concatenate all sequences
    X = np.concatenate(observations).reshape(-1, 1)
    lengths = [len(seq) for seq in observations]

    # Number of unique observation symbols
    n_symbols = len(np.unique(X))

    model = hmm.CategoricalHMM(
        n_components=n_hidden_states,
        n_iter=n_iter,
        tol=1e-4,  # Convergence threshold
        random_state=42,
        init_params="ste",  # Initialize: startprob, transmat, emissionprob
    )

    # Fit using Baum-Welch algorithm
    model.fit(X, lengths)

    return model


def demo_categorical_hmm():
    """Demo: Learning action patterns from behavioral traces."""
    print("=" * 60)
    print("CATEGORICAL HMM - Discrete Action Sequences")
    print("=" * 60)

    # Simulated action codes for ARC-like puzzle solving:
    # 0=select, 1=draw, 2=fill, 3=copy, 4=undo, 5=submit
    action_names = ["select", "draw", "fill", "copy", "undo", "submit"]

    # Synthetic sequences representing different solving strategies
    # "Exploratory" strategy: lots of select/undo
    exploratory = [
        np.array([0, 1, 4, 0, 2, 4, 0, 1, 1, 4, 0, 3, 5]),
        np.array([0, 4, 0, 1, 4, 0, 2, 2, 4, 0, 1, 5]),
        np.array([0, 0, 1, 4, 4, 0, 2, 1, 4, 0, 3, 5]),
    ]

    # "Confident" strategy: direct actions, few undos
    confident = [
        np.array([0, 1, 1, 2, 3, 5]),
        np.array([0, 2, 1, 1, 3, 5]),
        np.array([0, 1, 2, 2, 3, 5]),
    ]

    all_sequences = exploratory + confident

    print("\nTraining HMM on action sequences...")
    print(f"Number of sequences: {len(all_sequences)}")
    print(f"Actions: {action_names}")

    model = train_categorical_hmm(all_sequences, n_hidden_states=3, n_iter=100)

    print(f"\nBaum-Welch converged: {model.monitor_.converged}")
    print(f"Final log-likelihood: {model.monitor_.history[-1]:.2f}")

    # Learned parameters
    print("\n--- Learned Parameters ---")
    print("\nInitial state probabilities (π):")
    for i, p in enumerate(model.startprob_):
        print(f"  State {i}: {p:.3f}")

    print("\nTransition matrix (A):")
    print("  From\\To  ", end="")
    for j in range(model.n_components):
        print(f"  S{j}   ", end="")
    print()
    for i, row in enumerate(model.transmat_):
        print(f"  State {i}: ", end="")
        for p in row:
            print(f"{p:.3f} ", end="")
        print()

    print("\nEmission probabilities (which actions each hidden state emits):")
    for i, row in enumerate(model.emissionprob_):
        print(f"  State {i}:", end="")
        top_actions = np.argsort(row)[-3:][::-1]
        for idx in top_actions:
            print(f" {action_names[idx]}({row[idx]:.2f})", end="")
        print()

    # Decode a new sequence
    print("\n--- Decoding New Sequence ---")
    new_seq = np.array([0, 1, 4, 0, 2, 5]).reshape(-1, 1)
    hidden_states = model.predict(new_seq)
    log_prob = model.score(new_seq)

    print(f"Sequence: {[action_names[a] for a in new_seq.flatten()]}")
    print(f"Inferred hidden states: {hidden_states}")
    print(f"Log probability: {log_prob:.2f}")

    return model


# =============================================================================
# 2. GAUSSIAN HMM - For continuous observations (timing, durations)
# =============================================================================

def train_gaussian_hmm(observations, n_hidden_states=3, n_iter=100):
    """
    Train a Gaussian HMM using Baum-Welch algorithm.

    Args:
        observations: List of observation sequences (each is 2D array: time x features)
        n_hidden_states: Number of hidden states to learn
        n_iter: Maximum Baum-Welch iterations

    Returns:
        Trained HMM model
    """
    X = np.concatenate(observations)
    lengths = [len(seq) for seq in observations]

    model = hmm.GaussianHMM(
        n_components=n_hidden_states,
        covariance_type="full",  # Full covariance matrices
        n_iter=n_iter,
        tol=1e-4,
        random_state=42,
        init_params="stmc",  # Initialize: startprob, transmat, means, covars
    )

    model.fit(X, lengths)

    return model


def demo_gaussian_hmm():
    """Demo: Learning temporal patterns from timing data."""
    print("\n" + "=" * 60)
    print("GAUSSIAN HMM - Continuous Temporal Patterns")
    print("=" * 60)

    np.random.seed(42)

    # Simulated timing data: [action_duration, pause_before_action]
    # Three cognitive states: "thinking" (long pauses), "executing" (fast), "reviewing" (medium)

    def generate_sequence(state_pattern):
        """Generate timing observations based on cognitive state pattern."""
        state_params = {
            'thinking': {'duration': (0.5, 0.2), 'pause': (3.0, 1.0)},
            'executing': {'duration': (0.2, 0.1), 'pause': (0.3, 0.1)},
            'reviewing': {'duration': (1.0, 0.3), 'pause': (1.5, 0.5)},
        }
        observations = []
        for state in state_pattern:
            params = state_params[state]
            duration = max(0.1, np.random.normal(*params['duration']))
            pause = max(0.1, np.random.normal(*params['pause']))
            observations.append([duration, pause])
        return np.array(observations)

    # Generate training sequences with different patterns
    sequences = [
        generate_sequence(['thinking', 'thinking', 'executing', 'executing', 'executing', 'reviewing']),
        generate_sequence(['thinking', 'executing', 'executing', 'reviewing', 'executing', 'reviewing']),
        generate_sequence(['executing', 'executing', 'executing', 'reviewing', 'reviewing']),
        generate_sequence(['thinking', 'thinking', 'thinking', 'executing', 'reviewing']),
        generate_sequence(['thinking', 'executing', 'reviewing', 'executing', 'executing', 'reviewing']),
    ]

    print("\nTraining HMM on timing sequences...")
    print(f"Number of sequences: {len(sequences)}")
    print("Features: [action_duration, pause_before_action]")

    model = train_gaussian_hmm(sequences, n_hidden_states=3, n_iter=100)

    print(f"\nBaum-Welch converged: {model.monitor_.converged}")
    print(f"Final log-likelihood: {model.monitor_.history[-1]:.2f}")

    # Learned parameters
    print("\n--- Learned Parameters ---")
    print("\nInitial state probabilities (π):")
    for i, p in enumerate(model.startprob_):
        print(f"  State {i}: {p:.3f}")

    print("\nTransition matrix (A):")
    for i, row in enumerate(model.transmat_):
        print(f"  State {i}: [{', '.join(f'{p:.3f}' for p in row)}]")

    print("\nEmission parameters (Gaussian means for each state):")
    print("  [duration, pause]")
    for i, mean in enumerate(model.means_):
        print(f"  State {i}: duration={mean[0]:.2f}s, pause={mean[1]:.2f}s")

    # Decode a new sequence
    print("\n--- Decoding New Sequence ---")
    new_seq = np.array([[0.5, 2.8], [0.2, 0.3], [0.3, 0.4], [1.1, 1.6]])
    hidden_states = model.predict(new_seq)
    log_prob = model.score(new_seq)

    print(f"Observations: {new_seq.tolist()}")
    print(f"Inferred hidden states: {hidden_states}")
    print(f"Log probability: {log_prob:.2f}")

    return model


# =============================================================================
# 3. CUSTOM BAUM-WELCH MONITORING
# =============================================================================

def train_with_monitoring(observations, n_hidden_states=3, n_iter=100):
    """
    Train HMM with detailed Baum-Welch iteration monitoring.
    Shows the EM algorithm's convergence progress.
    """
    print("\n" + "=" * 60)
    print("BAUM-WELCH CONVERGENCE MONITORING")
    print("=" * 60)

    X = np.concatenate(observations).reshape(-1, 1)
    lengths = [len(seq) for seq in observations]

    model = hmm.CategoricalHMM(
        n_components=n_hidden_states,
        n_iter=n_iter,
        tol=1e-6,
        random_state=42,
        verbose=False,
    )

    model.fit(X, lengths)

    # Access convergence history (convert to list for compatibility)
    history = list(model.monitor_.history)

    print(f"\nIterations until convergence: {len(history)}")
    print(f"Converged: {model.monitor_.converged}")
    print("\nLog-likelihood progression:")

    display_count = min(10, len(history))
    for i in range(display_count):
        ll = history[i]
        improvement = "" if i == 0 else f" (Δ = {ll - history[i-1]:+.4f})"
        print(f"  Iter {i+1:3d}: {ll:.4f}{improvement}")

    if len(history) > 10:
        print(f"  ...")
        print(f"  Iter {len(history):3d}: {history[-1]:.4f}")

    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("BAUM-WELCH ALGORITHM DEMONSTRATION")
    print("Using hmmlearn for Hidden Markov Model training\n")

    # Run demos
    categorical_model = demo_categorical_hmm()
    gaussian_model = demo_gaussian_hmm()

    # Show convergence monitoring
    sample_data = [np.array([0, 1, 2, 1, 0, 2, 1]) for _ in range(5)]
    train_with_monitoring(sample_data)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The Baum-Welch algorithm learned:

1. CATEGORICAL HMM: Hidden cognitive strategies from discrete actions
   - Which actions are likely in each hidden state
   - How users transition between strategies

2. GAUSSIAN HMM: Hidden cognitive states from timing patterns
   - Characteristic timing signatures for each state
   - State transition dynamics

For your ARC research, you can:
- Use CategoricalHMM on action sequences to discover solving strategies
- Use GaussianHMM on timing data to identify cognitive states
- Compare hidden state patterns between humans and AI
""")
