import pandas as pd
import time
import random
import sys
from typing import List, Dict, Tuple, Optional

from agents import BottleneckAgent, ClassroomAgent, simple_prof_policy, compute_initial_slots
from protocols import Message, Commitment, Adjustment
from config import N_CLASSROOMS, INITIAL_ATTENDANCE, MAX_STUDENTS_PER_SLOT, VIOLATION_LIMIT
from config import SIMULATION_EPISODES, MAX_NEGOTIATION_ROUNDS

SIMULATION_LOG = []
COMMITMENT_COUNTER = 0

def get_total_slot_load(all_agents: List['ClassroomAgent']) -> Dict[int, int]:
    global_load: Dict[int, int] = {}
    for agent in all_agents:
        for slot, students in agent.planned_exit_slots.items():
            global_load[slot] = global_load.get(slot, 0) + students
    return global_load

def check_for_congestion(global_load: Dict[int, int]) -> Tuple[bool, Optional[int]]:
    for slot, load in global_load.items():
        if load > MAX_STUDENTS_PER_SLOT:
            return True, slot
    return False, None

def find_available_slot(global_load: Dict[int, int]) -> Optional[int]:
    MIN_CAPACITY_REQUIRED = 20
    
    slots_to_check = [-6, -4, -2, 0, 2, 4, 6]
    
    for offset in slots_to_check:
        current_load = global_load.get(offset, 0)
        remaining_capacity = MAX_STUDENTS_PER_SLOT - current_load
        
        if remaining_capacity >= MIN_CAPACITY_REQUIRED:
            return offset
    return None

def find_best_target_agent(
    all_agents: List['ClassroomAgent'], congested_slot: int, current_agent: 'ClassroomAgent'
) -> Optional['ClassroomAgent']:
    contributors = []
    for agent in all_agents:
        if agent.name != current_agent.name and agent.planned_exit_slots.get(congested_slot, 0) > 0:
            contributors.append(agent)
    
    if contributors:
        contributors.sort(key=lambda a: a.planned_exit_slots.get(congested_slot, 0), reverse=True)
        return contributors[0]
    return None

def attempt_adjustment(agent: 'ClassroomAgent', adjustment: Adjustment, destination_slot: int) -> bool:
    source_slot = adjustment.slot_minutes
    students_to_move = adjustment.students
    current_students = agent.planned_exit_slots.get(source_slot, 0)
    
    if current_students < students_to_move:
        return False

    agent.planned_exit_slots[source_slot] -= students_to_move
    if agent.planned_exit_slots[source_slot] == 0:
        del agent.planned_exit_slots[source_slot]
        
    agent.planned_exit_slots[destination_slot] = agent.planned_exit_slots.get(destination_slot, 0) + students_to_move
    
    print(f"  > [Action] {agent.name}: Adjusted {students_to_move} students from slot {source_slot} to {destination_slot}.")
    return True

def process_commitment_fulfillment(agent: ClassroomAgent, current_episode: int):
    global COMMITMENT_COUNTER
    
    due_commitments = [c for c in agent.commitment_history if c.status == "PENDING" and c.due_episode <= current_episode]
    
    for commitment in due_commitments:
        print(f"  > [Commitment Check] {agent.name} checking commitment {commitment.id} (Type: {commitment.type}).")
        
        if commitment.type == "RECEIVED":
            pass
            
        elif commitment.type == "MADE":
            if agent.prof_policy(commitment.reciprocal_due):
                agent.update_history_status(commitment.id, "FULFILLED")
                print(f"  > [FULFILLED] {agent.name} honored a past commitment: {commitment.reciprocal_due}.")
            else:
                print(f"  > [REFUSED] {agent.name}'s Professor refused to honor commitment: {commitment.reciprocal_due}.")
                
        if commitment.type == "RECEIVED" and commitment.status != "FULFILLED":
            pass

def check_violations(all_agents: List[ClassroomAgent]):
    for agent in all_agents:
        if not isinstance(agent, ClassroomAgent): continue
        
        unfulfilled_received = [
            c for c in agent.commitment_history 
            if c.type == "RECEIVED" and c.status == "PENDING" and c.due_episode < agent.current_epoch
        ]
        
        if unfulfilled_received:
            agent.violations_count += len(unfulfilled_received) 
            print(f"  > [VIOLATION] {agent.name} raises {len(unfulfilled_received)} violation(s) against partners. Total: {agent.violations_count}")
        
        if agent.violations_count >= VIOLATION_LIMIT:
            print(f"!!! {agent.name} HAS EXCEEDED VIOLATION LIMIT ({VIOLATION_LIMIT})!!!")
        
        for c in unfulfilled_received:
            if c.status == "PENDING":
                c.status = "PENDING_VIOLATION_TRACKED"

def run_negotiation_round(all_agents: List['ClassroomAgent'], current_episode: int) -> bool:
    global_load = get_total_slot_load(all_agents)
    congested, congested_slot = check_for_congestion(global_load)
    if not congested:
        return False

    successful_deal = False
    
    negotiators = [a for a in all_agents if isinstance(a, ClassroomAgent) and a.planned_exit_slots.get(congested_slot, 0) > 0]
    random.shuffle(negotiators) 
    
    if not negotiators:
        print("  > [Negotiation] Congestion detected but no agents contributing to it are available to negotiate.")
        return False

    for proposer in negotiators:
        target_agent = find_best_target_agent(all_agents, congested_slot, proposer)
        if not target_agent: continue
        
        available_slot = find_available_slot(global_load)
        if available_slot is None:
            continue 
            
        students_to_move = min(
            proposer.planned_exit_slots.get(congested_slot, 0),
            20
        )
        
        if students_to_move == 0: continue
        
        target_immediate_adjustment = Adjustment(
            slot_minutes=congested_slot, 
            students=students_to_move
        )
        
        proposer_reciprocal = Adjustment(
            slot_minutes=-2,
            students=students_to_move 
        )

        if target_agent.prof_policy(target_immediate_adjustment):
            
            if attempt_adjustment(target_agent, target_immediate_adjustment, available_slot):
            
                new_commitment = Commitment(
                    partner_id=target_agent.name,
                    type="MADE",
                    adjustment=target_immediate_adjustment,
                    reciprocal_due=proposer_reciprocal,
                    due_episode=current_episode + random.randint(1, 2), 
                    status="PENDING",
                    created_episode=current_episode
                )
                proposer.commitment_history.append(new_commitment)
                
                reciprocal_commitment = Commitment(
                    partner_id=proposer.name,
                    type="RECEIVED",
                    adjustment=target_immediate_adjustment,
                    reciprocal_due=proposer_reciprocal,
                    due_episode=new_commitment.due_episode,
                    status="PENDING",
                    created_episode=current_episode
                )
                target_agent.commitment_history.append(reciprocal_commitment)

                msg_broadcast = Message(
                    sender_id=proposer.name,
                    receiver_id="BROADCAST",
                    type="CommitBroadcast",
                    epoch=current_episode,
                    timestamp=time.time(),
                    payload={
                        "agents": [proposer.name, target_agent.name],
                        "adjustment": {target_agent.name: target_immediate_adjustment},
                        "new_commit_id": new_commitment.id,
                    }
                )
                
                print(f"  > [DEAL] {proposer.name} <-> {target_agent.name}: Target moves {students_to_move} from {congested_slot} to the next slot (+2). BROADCASTING!")
                
                successful_deal = True
                break
            else:
                pass 
        else:
            pass 
            
    return successful_deal

def get_commitment_history_summary(all_agents: List['ClassroomAgent']) -> pd.DataFrame:
    all_records = []
    for agent in all_agents:
        if not isinstance(agent, ClassroomAgent): continue
        for commitment in agent.commitment_history:
            record = {
                'Agent_ID': agent.name,
                'Commitment_ID': commitment.id,
                'Type': commitment.type,
                'Partner_ID': commitment.partner_id,
                'Status': commitment.status,
                'Created_Episode': commitment.created_episode,
                'Due_Episode': commitment.due_episode,
                'Reciprocal_Adj': f"{commitment.reciprocal_due.students} students @ {commitment.reciprocal_due.slot_minutes} min",
            }
            all_records.append(record)
    
    df = pd.DataFrame(all_records).sort_values(by=['Created_Episode', 'Agent_ID'])
    return df

def run_simulation():
    random.seed(42)

    bottleneck_agent = BottleneckAgent()
    classroom_agents: List[ClassroomAgent] = []
    
    for i in range(N_CLASSROOMS):
        cid = i + 1
        prof_policy = simple_prof_policy(cid)
        agent = ClassroomAgent(
            cid=cid,
            initial_attendance=INITIAL_ATTENDANCE[i],
            prof_policy=prof_policy
        )
        classroom_agents.append(agent)
    
    all_agents = classroom_agents + [bottleneck_agent]
    bottleneck_agent.set_agents(all_agents)
    
    print("--- Starting Multiagent Traffic Coordination Simulation ---")
    
    for episode in range(1, SIMULATION_EPISODES + 1):
        print(f"\n================ EPISODE {episode}: Monday 11:00 AM ================")
        
        print("--- Resetting Attendance for New Episode ---")
        current_attendance = []
        for agent in classroom_agents:
            base_attendance = INITIAL_ATTENDANCE[agent.cid - 1]
            variation = int(base_attendance * random.uniform(-0.1, 0.1))
            new_attendance = base_attendance + variation
            
            agent.attendance = new_attendance
            agent.planned_exit_slots = compute_initial_slots(new_attendance)
            current_attendance.append(new_attendance)
            print(f"  > {agent.name} new attendance: {new_attendance} -> Initial Slots: {agent.planned_exit_slots}")
        
        estimated_total = sum(INITIAL_ATTENDANCE)
        capacity_msg = bottleneck_agent.broadcast_capacity(estimated_total)
        print(f"[B] Broadcasts CapacityUpdate (Epoch {episode}, Max Slot Load: {MAX_STUDENTS_PER_SLOT})")
        
        for agent in classroom_agents:
            agent.handle_message(capacity_msg)
            
        for agent in classroom_agents:
            process_commitment_fulfillment(agent, episode)
        check_violations(classroom_agents)
        
        rounds = 0
        while rounds < MAX_NEGOTIATION_ROUNDS:
            print(f"--- Negotiation Round {rounds + 1} ---")
            
            current_global_load = get_total_slot_load(classroom_agents)
            congested, congested_slot = check_for_congestion(current_global_load)
            
            if not congested:
                print("--- CONGESTION RESOLVED. Negotiation Ends. ---")
                break
            
            print(f"*** Congestion Detected at slot {congested_slot} (Load: {current_global_load[congested_slot]}, Max: {MAX_STUDENTS_PER_SLOT}) ***")
            
            if not run_negotiation_round(classroom_agents, episode):
                print("  > No successful deal completed in this round.")
                rounds += 1
            else:
                rounds += 1 
                
        final_load = get_total_slot_load(classroom_agents)
        congested, _ = check_for_congestion(final_load)
        
        total_delay_minutes = 0
        for agent in classroom_agents:
            for slot, students in agent.planned_exit_slots.items():
                if slot > 0:
                    total_delay_minutes += students * slot
                    
        total_commitments = sum(len(a.commitment_history) for a in classroom_agents)

        log_entry = {
            'episode': episode,
            'initial_congestion': congested and rounds == 0,
            'final_congested': congested,
            'max_load': max(final_load.values()) if final_load else 0,
            'total_delay_minutes': total_delay_minutes,
            'negotiation_rounds': rounds,
            'agent_slots': {agent.name: agent.planned_exit_slots.copy() for agent in classroom_agents},
            'agent_violations': {agent.name: agent.violations_count for agent in classroom_agents}
        }
        SIMULATION_LOG.append(log_entry)
        
        print(f"\n[SUMMARY EPISODE {episode}] Final Congested: {congested}. Total Delay: {total_delay_minutes} student-minutes.")
        print("-" * 60)

    print("\n\n--- SIMULATION COMPLETED ---")
    
    df = pd.DataFrame(SIMULATION_LOG)
    
    agent_slot_data = []
    for index, row in df.iterrows():
        episode_df = pd.DataFrame.from_dict(row['agent_slots'], orient='index').fillna(0)
        episode_df['episode'] = row['episode']
        agent_slot_data.append(episode_df)

    dist_df = pd.concat(agent_slot_data).reset_index().rename(columns={'index': 'agent'})
    pivot_dist_df = dist_df.pivot(index='episode', columns='agent').fillna(0).astype(int)

    violations_df = pd.json_normalize(df['agent_violations']).fillna(0).astype(int)
    violations_df['episode'] = df['episode']
    violations_df.set_index('episode', inplace=True)
    violations_df.columns = [f"Violations_{c}" for c in violations_df.columns]

    core_metrics_df = df[['episode', 'final_congested', 'max_load', 'total_delay_minutes', 'negotiation_rounds']].set_index('episode')
    
    pivot_dist_df.columns = [f"{agent}_Slot_{slot}" for slot, agent in pivot_dist_df.columns]
    
    final_report_df = core_metrics_df.join(pivot_dist_df).join(violations_df).reset_index()

    slot_columns = [col for col in final_report_df.columns if 'Slot' in col]
    final_report_df['Total Students Exited'] = final_report_df[slot_columns].sum(axis=1)
    
    print("\n\n--- FINAL EPISODE SUMMARY & DETAILED TRAFFIC DISTRIBUTION ---")
    sorted_slot_cols = sorted([c for c in final_report_df.columns if 'Slot' in c])
    sorted_violation_cols = sorted([c for c in final_report_df.columns if 'Violations' in c])
    
    display_cols = ['episode', 'final_congested', 'max_load', 'negotiation_rounds'] + sorted_slot_cols + sorted_violation_cols + ['Total Students Exited']
    
    print(final_report_df[display_cols].to_string(index=False))

    history_df = get_commitment_history_summary(classroom_agents)
    print("\n\n--- DETAILED COMMITMENT HISTORY (By Episode and Status) ---")
    print(history_df.to_string(index=False))

if __name__ == "__main__":
    run_simulation()