import os
import sys
from pathlib import Path
import pytest

# Dynamically resolve project and src directories
CURRENT_FILE = Path(__file__).resolve()
TESTS_DIR = CURRENT_FILE.parent
ROOT_DIR = TESTS_DIR.parent
SRC_DIR = ROOT_DIR / "src"

for path_dir in [str(ROOT_DIR), str(SRC_DIR)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from fsm_observer import FSMObserver, Lane, State


def print_fsm_scorecard(title: str, metrics: dict) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    for key, val in metrics.items():
        print(f"  - {key:<26}: {str(val):<30}")
    print("=" * width + "\n")


def test_fsm_happy_path_single_patient():
    """Verify standard valid execution flow for Single Patient Lane and print telemetry."""
    observer = FSMObserver()
    
    assert observer.current_state == State.INCOMING_QUERY
    
    observer.transition_to(State.ROUTING)
    observer.set_lane(Lane.SINGLE_PATIENT)
    
    observer.transition_to(State.REACT_LOOP)
    observer.transition_to(State.EXTRACT_RECORD_INDEX)
    observer.transition_to(State.QDRANT_HYBRID_RRF)
    observer.transition_to(State.CROSS_ENCODER_RERANK)
    observer.transition_to(State.FORMAT_PATIENT_CONTEXT)
    observer.transition_to(State.COMPLETED)
    
    assert observer.current_state == State.COMPLETED
    
    # Print telemetry report for portfolio/blogs
    metrics = {
        "Active Lane": observer.active_lane.value,
        "Terminal State": observer.current_state.value,
        "Total State Transitions": len(observer.history) - 1,
        "Execution Trace": " -> ".join([s.value for s in observer.history]),
    }
    print_fsm_scorecard("HEALTHINSIGHT FSM // HAPPY PATH TELEMETRY", metrics)


def test_fsm_illegal_transition_raises_error():
    """Verify that skipping steps or illegal jumps throws a ValueError and prints security block log."""
    observer = FSMObserver()
    observer.transition_to(State.ROUTING)
    observer.set_lane(Lane.SINGLE_PATIENT)
    
    caught_error = None
    try:
        observer.transition_to(State.CROSS_ENCODER_RERANK)
    except ValueError as e:
        caught_error = str(e)

    assert caught_error is not None
    assert "Illegal transition" in caught_error

    metrics = {
        "Security Guardrail": "ACTIVE",
        "Blocked Illegal Jump": "ROUTING -> CROSS_ENCODER_RERANK",
        "Exception Caught": caught_error[:35] + "...",
    }
    print_fsm_scorecard("HEALTHINSIGHT FSM // SECURITY GUARDRAIL CHECK", metrics)


def test_fsm_self_healing_retry_cap():
    """Verify that healing increments retries and forces FAILED status once max retries are exceeded."""
    observer = FSMObserver(max_retries=2)
    observer.transition_to(State.ROUTING)
    observer.set_lane(Lane.SINGLE_PATIENT)
    observer.transition_to(State.REACT_LOOP)
    observer.transition_to(State.EXTRACT_RECORD_INDEX)
    observer.transition_to(State.QDRANT_HYBRID_RRF)
    
    observer.trigger_healing("Simulated transient error 1")
    assert observer.current_state == State.HEALING_RETRY
    assert observer.retry_count == 1
    
    observer.transition_to(State.QDRANT_HYBRID_RRF)
    observer.trigger_healing("Simulated transient error 2")
    assert observer.retry_count == 2
    
    observer.transition_to(State.QDRANT_HYBRID_RRF)
    observer.trigger_healing("Simulated transient error 3 (Breaks cap)")
    assert observer.current_state == State.FAILED

    metrics = {
        "Self-Healing Mechanism": "OPERATIONAL",
        "Max Retries Limit": observer.max_retries,
        "Recorded Retries": observer.retry_count,
        "Terminal Failure State": observer.current_state.value,
    }
    print_fsm_scorecard("HEALTHINSIGHT FSM // SELF-HEALING RESILIENCE REPORT", metrics)

def main():
    print("\n--- Starting HealthInsight RAG FSM Verification Runner ---")
    try:
        test_fsm_happy_path_single_patient()
        test_fsm_illegal_transition_raises_error()
        test_fsm_self_healing_retry_cap()
        print("\n[SUCCESS] All FSM state observer and self-healing tests completed successfully!\n")
    except Exception as e:
        print(f"\n[FAILED] FSM verification encountered an error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()