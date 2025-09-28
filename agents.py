from autogen import ConversableAgent
import uuid
import time
import random
from typing import List, Dict, Callable

from protocols import Message, Commitment, Adjustment
from config import (
    N_CLASSROOMS, MAX_STUDENTS_PER_SLOT, BOTTLENECK_CAPACITY_PER_MINUTE,
    SLOT_DURATION_MINUTES, INITIAL_ATTENDANCE, VIOLATION_LIMIT,
    SIMULATION_EPISODES, LECTURE_NOMINAL_END_TIME
)

def compute_initial_slots(attendance: int) -> Dict[int, int]:
    """
    Calculates the minimal exit slots required for a classroom's attendance.
    """
    slots = {}
    remaining = attendance
    slot_index = 0 
    
    while remaining > 0:
        if slot_index == 0:
            slot_offset = 0
        elif slot_index % 2 == 1:
            slot_offset = -SLOT_DURATION_MINUTES * ((slot_index + 1) // 2)
        else:
            slot_offset = SLOT_DURATION_MINUTES * (slot_index // 2)

        students_in_slot = min(remaining, MAX_STUDENTS_PER_SLOT)
        slots[slot_offset] = students_in_slot
        remaining -= students_in_slot
        slot_index += 1

    return slots

class BottleneckAgent(ConversableAgent):
    """
    Agent B: Monitors the bottleneck and broadcasts capacity to classroom agents.
    """
    def __init__(self, name="B", **kwargs):
        super().__init__(
            name=name,
            llm_config=False,
            human_input_mode="NEVER",
            system_message=(
                "You are the Bottleneck Monitor Agent (B). "
                "Announce traffic capacity and estimated congestion to all Classroom Agents."
            )
        )
        self.capacity_per_slot = MAX_STUDENTS_PER_SLOT
        self.classroom_agents: List['ClassroomAgent'] = []
        self.all_agents: List[ConversableAgent] = []
        self.current_episode = 0

    def set_agents(self, agents: List[ConversableAgent]):
        self.all_agents = agents
        self.classroom_agents = [a for a in agents if a.name.startswith('C')]

    def broadcast_capacity(self, estimated_total_students: int) -> Message:
        self.current_episode += 1
        payload = {
            "capacity_per_slot": self.capacity_per_slot,
            "estimated_total_students": estimated_total_students,
            "lecture_end_time": LECTURE_NOMINAL_END_TIME
        }
        msg_obj = Message(
            sender_id=self.name,
            receiver_id="BROADCAST",
            type="CapacityUpdate",
            epoch=self.current_episode,
            timestamp=time.time(),
            payload=payload
        )
        return msg_obj

class ClassroomAgent(ConversableAgent):
    """
    Agent C_i: Manages classroom exit slots, maintains commitments, and negotiates.
    """
    def __init__(self, cid: int, initial_attendance: int, prof_policy: Callable[[Adjustment], bool], **kwargs):
        name = f"C{cid}"
        super().__init__(
            name=name,
            llm_config=False,
            human_input_mode="NEVER",
            system_message=(
                f"You are Classroom Agent {name}. "
                f"Ensure your {initial_attendance} students exit safely by coordinating with peers."
            )
        )
        self.cid = cid
        self.attendance = initial_attendance
        self.prof_policy = prof_policy
        self.commitment_history: List[Commitment] = []
        self.planned_exit_slots: Dict[int, int] = compute_initial_slots(self.attendance)
        self.violations_count = 0
        self.current_epoch = 0
        self.negotiation_status = "IDLE"
        self.global_slot_load: Dict[int, int] = {}
        self.active_proposals: Dict[str, Message] = {}

    def get_total_attendance(self) -> int:
        return sum(self.planned_exit_slots.values())

    def check_congestion_risk(self) -> bool:
        for students in self.planned_exit_slots.values():
            if students > MAX_STUDENTS_PER_SLOT:
                return True
        return False

    def update_history_status(self, commitment_id: str, status: str):
        for c in self.commitment_history:
            if c.id == commitment_id:
                c.status = status
                return

    def handle_message(self, message: Message):
        if message.type == "CapacityUpdate":
            self.current_epoch = message.epoch
            self.global_slot_load = {}
        elif message.type == "CommitBroadcast":
            pass
        elif message.type == "ViolationReport":
            pass

def simple_prof_policy(cid: int) -> Callable[[Adjustment], bool]:
    if cid == 1:
        return lambda adjustment: -4 <= adjustment.slot_minutes <= 6
    elif cid == 3:
        return lambda adjustment: adjustment.slot_minutes >= 0
    else:
        return lambda adjustment: abs(adjustment.slot_minutes) <= 2
