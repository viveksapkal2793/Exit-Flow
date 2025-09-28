import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Adjustment:
    slot_minutes: int
    students: int

@dataclass
class Commitment:
    partner_id: str
    type: str
    adjustment: Adjustment
    reciprocal_due: Adjustment
    due_episode: int
    status: str
    created_episode: int
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    type: str
    epoch: int
    timestamp: float
    payload: Dict[str, Any]