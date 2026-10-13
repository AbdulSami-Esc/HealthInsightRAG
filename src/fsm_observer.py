from enum import Enum
from typing import Optional
import logging
from typing import Optional

class Lane(str, Enum):
    GENERAL = "GENERAL"
    SINGLE_PATIENT = "SINGLE_PATIENT"
    COHORT = "COHORT"
    ANALYTICAL = "ANALYTICAL"


class State(str, Enum):
    # Ingestion & Routing
    INCOMING_QUERY = "INCOMING_QUERY"
    ROUTING = "ROUTING"

    # Lane 1: General
    DIRECT_LLM_RESPONSE = "DIRECT_LLM_RESPONSE"

    # Lane 2: Single Patient
    REACT_LOOP = "REACT_LOOP"
    EXTRACT_RECORD_INDEX = "EXTRACT_RECORD_INDEX"
    QDRANT_HYBRID_RRF = "QDRANT_HYBRID_RRF"
    CROSS_ENCODER_RERANK = "CROSS_ENCODER_RERANK"
    FORMAT_PATIENT_CONTEXT = "FORMAT_PATIENT_CONTEXT"

    # Lane 3: Cohort
    GLOBAL_HYBRID_SEARCH = "GLOBAL_HYBRID_SEARCH"
    QDRANT_SEARCH = "QDRANT_SEARCH"
    MARKDOWN_TABLE_OUTPUT = "MARKDOWN_TABLE_OUTPUT"

    # Lane 4: Analytical
    PANDAS_EXECUTION = "PANDAS_EXECUTION"
    QUANTITATIVE_RESULT = "QUANTITATIVE_RESULT"

    # Resilience & Exit States
    HEALING_RETRY = "HEALING_RETRY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# 2. HASH MAP RULES (O(1) CONSTANT TIME LOOKUPS)

# Maps each Lane to the explicit set of states it is permitted to execute.
LANE_ALLOWED_STATES: dict[Lane, set[State]] = {
    Lane.GENERAL: {
        State.DIRECT_LLM_RESPONSE,
        State.COMPLETED,
        State.FAILED,
    },
    Lane.SINGLE_PATIENT: {
        State.REACT_LOOP,
        State.EXTRACT_RECORD_INDEX,
        State.QDRANT_HYBRID_RRF,
        State.CROSS_ENCODER_RERANK,
        State.FORMAT_PATIENT_CONTEXT,
        State.HEALING_RETRY,
        State.COMPLETED,
        State.FAILED,
    },
    Lane.COHORT: {
        State.GLOBAL_HYBRID_SEARCH,
        State.QDRANT_SEARCH,
        State.MARKDOWN_TABLE_OUTPUT,
        State.HEALING_RETRY,
        State.COMPLETED,
        State.FAILED,
    },
    Lane.ANALYTICAL: {
        State.PANDAS_EXECUTION,
        State.QUANTITATIVE_RESULT,
        State.COMPLETED,
        State.FAILED,
    },
}

#A set within a dictionary
VALID_TRANSITIONS: dict[State, set[State]] = {
    State.INCOMING_QUERY: {State.ROUTING, State.FAILED},
    State.ROUTING: {
        State.DIRECT_LLM_RESPONSE,
        State.REACT_LOOP,
        State.GLOBAL_HYBRID_SEARCH,
        State.PANDAS_EXECUTION,
        State.EXTRACT_RECORD_INDEX,
        State.COMPLETED,
        State.FAILED,
    },
    # Lane 1: General
    State.DIRECT_LLM_RESPONSE: {State.COMPLETED, State.FAILED},

    # Lane 2: Single Patient
    State.REACT_LOOP: {State.EXTRACT_RECORD_INDEX, State.HEALING_RETRY, State.COMPLETED,State.FAILED}, # <--- FIX 2: Allow healing retry from ReAct loop
    State.EXTRACT_RECORD_INDEX: {State.QDRANT_HYBRID_RRF, State.FAILED},
    State.QDRANT_HYBRID_RRF: {State.QDRANT_HYBRID_RRF,State.CROSS_ENCODER_RERANK, State.HEALING_RETRY, State.FAILED},
    State.CROSS_ENCODER_RERANK: {State.FORMAT_PATIENT_CONTEXT, State.FAILED},
    State.FORMAT_PATIENT_CONTEXT: {State.COMPLETED, State.FAILED},
    
    # Lane 3: Cohort
    State.GLOBAL_HYBRID_SEARCH: {State.QDRANT_SEARCH, State.FAILED},
    State.QDRANT_SEARCH: {State.QDRANT_SEARCH,State.MARKDOWN_TABLE_OUTPUT, State.HEALING_RETRY, State.FAILED},
    State.MARKDOWN_TABLE_OUTPUT: {State.COMPLETED, State.FAILED},

    # Lane 4: Analytical
    State.PANDAS_EXECUTION: {State.QUANTITATIVE_RESULT, State.HEALING_RETRY, State.FAILED},
    State.QUANTITATIVE_RESULT: {State.COMPLETED, State.FAILED},

    # Healing Loop
    State.HEALING_RETRY: {State.QDRANT_HYBRID_RRF, State.QDRANT_SEARCH, State.PANDAS_EXECUTION, State.COMPLETED,State.FAILED},

    # Terminal States
    State.COMPLETED: set(),
    State.FAILED: set(),
}


logger = logging.getLogger(__name__)


def can_transition(current_state: State, next_state: State, active_lane: Optional[Lane] = None) -> bool:
    allowed_next = VALID_TRANSITIONS.get(current_state, set())
    if next_state not in allowed_next:
        return False

    if active_lane is not None:
        if next_state not in LANE_ALLOWED_STATES[active_lane]:
            return False

    return True


class FSMObserver:
    def __init__(self, max_retries: int = 3):
        self.current_state: State = State.INCOMING_QUERY
        self.active_lane: Optional[Lane] = None
        self.retry_count: int = 0
        self.max_retries: int = max_retries
        self.history: list[State] = [self.current_state]

    def set_lane(self, lane: Lane) -> None:
        if self.active_lane is not None:
            raise ValueError(f"Lane already set to {self.active_lane}")
        if self.current_state != State.ROUTING:
            raise ValueError("Lane can only be set while in ROUTING state")
        self.active_lane = lane

    def transition_to(self, next_state: State) -> None:
        if not can_transition(self.current_state, next_state, self.active_lane):
            raise ValueError(
                f"Illegal transition: {self.current_state} -> {next_state} "
                f"(lane: {self.active_lane})"
            )

        if next_state == State.HEALING_RETRY:
            if self.retry_count >= self.max_retries:
                next_state = State.FAILED
            else:
                self.retry_count += 1

        self.current_state = next_state
        self.history.append(next_state)

    def trigger_healing(self, reason: str) -> bool:
        logger.warning("Healing triggered: %s", reason)
        self.transition_to(State.HEALING_RETRY)
        return self.current_state == State.HEALING_RETRY