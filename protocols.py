# protocols.py

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List

# A structure to represent the adjustment or reciprocal commitment
@dataclass
class Adjustment:
    """Represents a shift in students from one slot to another."""
    slot_minutes: int  # e.g., -4, -2, 0, +2
    students: int      # number of students affected

@dataclass
class Commitment:
    """Represents a historical agreement between two agents."""
    # 1. Non-default fields MUST come first
    partner_id: str
    type: str          # "MADE" (C_i promises C_j) or "RECEIVED" (C_i receives promise from C_j)
    adjustment: Adjustment # The immediate action taken (e.g., C_i moves 20 students to slot -2)
    reciprocal_due: Adjustment # The commitment to be honored later (e.g., C_i owes C_j 2 min extra time for 20 students)
    due_episode: int   # The episode where the reciprocal must be fulfilled (e.g., episode 3)
    status: str        # "PENDING", "FULFILLED", "VIOLATED"
    created_episode: int
    
    # 2. Default fields MUST come last
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class Message:
    """Base message structure for agent communication."""
    sender_id: str
    receiver_id: str # Can be 'BROADCAST'
    type: str        # e.g., 'CapacityUpdate', 'CommitRequest', 'CommitBroadcast'
    epoch: int       # Simulation episode number (from Agent B)
    timestamp: float # For tie-breaking
    payload: Dict[str, Any]