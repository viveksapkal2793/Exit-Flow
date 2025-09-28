# driver.py

import pandas as pd
import time
import random
import sys
from typing import List, Dict, Tuple, Optional

# Import agents, protocols, and config
from agents import BottleneckAgent, ClassroomAgent, simple_prof_policy, compute_initial_slots
from protocols import Message, Commitment, Adjustment
from config import N_CLASSROOMS, INITIAL_ATTENDANCE, MAX_STUDENTS_PER_SLOT, VIOLATION_LIMIT
from config import SIMULATION_EPISODES, MAX_NEGOTIATION_ROUNDS

# --- GLOBAL TRACKERS ---
SIMULATION_LOG = [] # For storing episodic results
COMMITMENT_COUNTER = 0

# --- CORE LOGIC FUNCTIONS ---

def get_total_slot_load(all_agents: List['ClassroomAgent']) -> Dict[int, int]:
    """Calculates the total student load for every planned exit slot across all agents."""
    global_load: Dict[int, int] = {}
    for agent in all_agents:
        for slot, students in agent.planned_exit_slots.items():
            global_load[slot] = global_load.get(slot, 0) + students
    return global_load

def check_for_congestion(global_load: Dict[int, int]) -> Tuple[bool, Optional[int]]:
    """Checks for congestion and returns the first congested slot, if any."""
    for slot, load in global_load.items():
        if load > MAX_STUDENTS_PER_SLOT:
            return True, slot
    return False, None

# --- Find Slot Fix ---
def find_available_slot(global_load: Dict[int, int]) -> Optional[int]:
    """
    Finds a slot (earlier or later) that has at least MIN_CAPACITY_REQUIRED capacity.
    This makes negotiation much more flexible.
    """
    MIN_CAPACITY_REQUIRED = 20 # Allow smaller batches (up to 20 students in a deal)
    
    # Check slots from -6 to +6 (common range for negotiation)
    # Note: 0 is included to handle cases where 0 is congested but has some remaining capacity.
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
    """Finds a peer agent contributing the most to the congested slot for negotiation."""
    contributors = []
    for agent in all_agents:
        if agent.name != current_agent.name and agent.planned_exit_slots.get(congested_slot, 0) > 0:
            contributors.append(agent)
    
    # Strategy: pick the agent contributing the most to the congestion
    if contributors:
        contributors.sort(key=lambda a: a.planned_exit_slots.get(congested_slot, 0), reverse=True)
        return contributors[0]
    return None

def attempt_adjustment(agent: 'ClassroomAgent', adjustment: Adjustment, destination_slot: int) -> bool:
    """
    Applies a local slot adjustment to an agent's planned_exit_slots.
    MODIFIED: Now accepts a 'destination_slot' to move students to a known available slot.
    """
    source_slot = adjustment.slot_minutes
    students_to_move = adjustment.students
    current_students = agent.planned_exit_slots.get(source_slot, 0)
    
    # Students must be moved *from* the source slot
    if current_students < students_to_move:
        return False # Not enough students to move

    # Adjust the source slot
    agent.planned_exit_slots[source_slot] -= students_to_move
    if agent.planned_exit_slots[source_slot] == 0:
        del agent.planned_exit_slots[source_slot]
        
    # Adjust the destination slot using the provided parameter
    agent.planned_exit_slots[destination_slot] = agent.planned_exit_slots.get(destination_slot, 0) + students_to_move
    
    print(f"  > [Action] {agent.name}: Adjusted {students_to_move} students from slot {source_slot} to {destination_slot}.")
    return True

# --- EPISODIC LOGIC ---

def process_commitment_fulfillment(agent: ClassroomAgent, current_episode: int):
    """Checks and handles commitments due in this episode."""
    global COMMITMENT_COUNTER
    
    due_commitments = [c for c in agent.commitment_history if c.status == "PENDING" and c.due_episode <= current_episode]
    
    for commitment in due_commitments:
        print(f"  > [Commitment Check] {agent.name} checking commitment {commitment.id} (Type: {commitment.type}).")
        
        if commitment.type == "RECEIVED":
            # This is a promise made TO this agent. We expect the partner to fulfill it.
            # We don't act, but track if it was honored in the negotiation phase.
            pass
            
        elif commitment.type == "MADE":
            # This agent owes the reciprocal adjustment. Attempt to fulfill.
            
            # The agent must now apply the reciprocal_due adjustment (e.g., extend lecture by 2 min)
            # For simplicity: fulfilling a MADE commitment means the agent extends/shortens 
            # its lecture slot for the partner's benefit. This is complex to model directly
            # by changing planned_exit_slots here.
            
            # Instead, we model fulfillment as: agent *tries* to adjust its schedule
            # to honor the adjustment, but first consults the current professor.
            if agent.prof_policy(commitment.reciprocal_due):
                # The professor agrees to honor the commitment (e.g., extend by 2 minutes)
                # We update the commitment status and assume the external world accounts for this.
                agent.update_history_status(commitment.id, "FULFILLED")
                print(f"  > [FULFILLED] {agent.name} honored a past commitment: {commitment.reciprocal_due}.")
            else:
                # The professor rejects, resulting in a violation for THIS agent's partner
                # The *partner* (the receiver) will flag the violation, not this agent.
                # Here, we leave it PENDING, and the partner will raise the violation flag.
                print(f"  > [REFUSED] {agent.name}'s Professor refused to honor commitment: {commitment.reciprocal_due}.")
                
        # Check for non-fulfillment (3 times rule)
        # This is more accurately done by the agent that *received* the commitment, 
        # but we centralize the violation check here for simplicity.
        if commitment.type == "RECEIVED" and commitment.status != "FULFILLED":
            # Assume we only check if the reciprocal slot adjustment was provided.
            # In a real MAS, we would need to check if the partner broadcasted an honoring message.
            # For simulation, we'll track the *refusals*.
            pass # Violation logic is better done within the main loop based on refusal.

def check_violations(all_agents: List[ClassroomAgent]):
    """Central check for violations after negotiation and fulfillment attempts."""
    for agent in all_agents:
        if not isinstance(agent, ClassroomAgent): continue
        
        # Check all pending MADE commitments that should have been honored by the partner
        # (i.e., those received by THIS agent that are still PENDING after the due episode)
        unfulfilled_received = [
            c for c in agent.commitment_history 
            if c.type == "RECEIVED" and c.status == "PENDING" and c.due_episode < agent.current_epoch
        ]
        
        # In a multi-episode run, we'd need a more nuanced tracking of refusal counts.
        # Simple rule: if it's due and pending, we count a violation attempt.
        if unfulfilled_received:
            # For simplicity, we assume one violation per unfulfilled commitment per due episode.
            agent.violations_count += len(unfulfilled_received) 
            print(f"  > [VIOLATION] {agent.name} raises {len(unfulfilled_received)} violation(s) against partners. Total: {agent.violations_count}")
        
        if agent.violations_count >= VIOLATION_LIMIT:
            print(f"!!! {agent.name} HAS EXCEEDED VIOLATION LIMIT ({VIOLATION_LIMIT})!!!")
            # Agent can now be penalized (e.g., removed from negotiation pool)
        
        # Reset PENDING commitments that are fulfilled to avoid double counting
        for c in unfulfilled_received:
            if c.status == "PENDING":
                c.status = "PENDING_VIOLATION_TRACKED" # Mark as tracked
                
# --- NEGOTIATION PROTOCOL ---

def run_negotiation_round(all_agents: List['ClassroomAgent'], current_episode: int) -> bool:
    """
    Executes one round of negotiation (proposals, 2PC, broadcasts).
    MODIFIED: Uses a smaller negotiation batch size and a more flexible reciprocal commitment
              to avoid the professor rejection spiral.
    """
    global_load = get_total_slot_load(all_agents)
    congested, congested_slot = check_for_congestion(global_load)
    if not congested:
        return False

    successful_deal = False
    
    # Agents who might make a move (contributing to congestion)
    negotiators = [a for a in all_agents if isinstance(a, ClassroomAgent) and a.planned_exit_slots.get(congested_slot, 0) > 0]
    random.shuffle(negotiators) 
    
    if not negotiators:
        print("  > [Negotiation] Congestion detected but no agents contributing to it are available to negotiate.")
        return False

    for proposer in negotiators:
        # 1. Identify Target and Proposal
        target_agent = find_best_target_agent(all_agents, congested_slot, proposer)
        if not target_agent: continue
        
        # Find an available slot (used primarily as a congestion indicator, not a direct negotiation target in current logic)
        available_slot = find_available_slot(global_load)
        if available_slot is None:
            continue 
            
        # Negotiate using a smaller, fixed batch size (20) to increase transaction success
        students_to_move = min(
            proposer.planned_exit_slots.get(congested_slot, 0),
            20 # FIXED: Use a smaller negotiation chunk size (was MAX_STUDENTS_PER_SLOT=60)
        )
        
        if students_to_move == 0: continue
        
        # 2. Craft Proposal (CommitRequest) - Phase 1 Start
        
        # Target's immediate adjustment: C_target moves students from congested_slot to the next slot (+2 min)
        target_immediate_adjustment = Adjustment(
            slot_minutes=congested_slot, 
            students=students_to_move
        )
        
        # Proposer's commitment (reciprocal) for the future.
        # FIXED: Offer a small, commonly accepted shift (-2 minutes) to avoid the -6 minute rejection spiral.
        proposer_reciprocal = Adjustment(
            slot_minutes=-2, # Offer to shift their lecture 2 min earlier next time (a desirable, modest change)
            students=students_to_move 
        )

        # 3. Target Agent (Target) checks for acceptance
        # Target consults its professor for the *immediate adjustment* (moving students to +2)
        if target_agent.prof_policy(target_immediate_adjustment):
            # Phase 1: ACCEPT
            
            # Target applies the change locally (Two-Phase Commit: Prepare success)
            if attempt_adjustment(target_agent, target_immediate_adjustment, available_slot):
            
                # 4. Phase 2: COMMIT / BROADCAST
                # Proposer logs the MADE commitment
                new_commitment = Commitment(
                    partner_id=target_agent.name,
                    type="MADE",
                    adjustment=target_immediate_adjustment,
                    reciprocal_due=proposer_reciprocal,
                    # Schedule fulfillment sooner (1-2 episodes) to test durability
                    due_episode=current_episode + random.randint(1, 2), 
                    status="PENDING",
                    created_episode=current_episode
                )
                proposer.commitment_history.append(new_commitment)
                
                # Target logs the RECEIVED commitment
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

                # Broadcast the successful deal!
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
                break # Successful deal, recalculate load and start new round
            else:
                pass 
        else:
            # Phase 1: REJECT (Target's professor says NO)
            pass 
            
    return successful_deal

# driver.py (New Reporting/Debugging Function)

def get_commitment_history_summary(all_agents: List['ClassroomAgent']) -> pd.DataFrame:
    """Collects and summarizes the commitment history for all agents."""
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
    
    # Sort for clearer debugging flow
    df = pd.DataFrame(all_records).sort_values(by=['Created_Episode', 'Agent_ID'])
    return df

# --- MAIN SIMULATION LOOP ---

def run_simulation():
    """Initializes agents and runs the episodic simulation."""
    random.seed(42) # Ensure reproducibility

    # 1. Initialization
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
    
    # 2. Episodic Simulation
    for episode in range(1, SIMULATION_EPISODES + 1):
        print(f"\n================ EPISODE {episode}: Monday 11:00 AM ================")
        
        # --- NEW: DYNAMIC ATTENDANCE ---
        # At the start of each episode, vary the attendance and reset slots.
        # This makes each episode a new, unique challenge.
        print("--- Resetting Attendance for New Episode ---")
        current_attendance = []
        for agent in classroom_agents:
            # Vary attendance by +/- 10% of the initial value
            base_attendance = INITIAL_ATTENDANCE[agent.cid - 1]
            variation = int(base_attendance * random.uniform(-0.1, 0.1))
            new_attendance = base_attendance + variation
            
            agent.attendance = new_attendance
            agent.planned_exit_slots = compute_initial_slots(new_attendance)
            current_attendance.append(new_attendance)
            print(f"  > {agent.name} new attendance: {new_attendance} -> Initial Slots: {agent.planned_exit_slots}")
        
        # A. Start Episode & Broadcast
        estimated_total = sum(INITIAL_ATTENDANCE) # B uses initial attendance as a proxy estimate
        capacity_msg = bottleneck_agent.broadcast_capacity(estimated_total)
        print(f"[B] Broadcasts CapacityUpdate (Epoch {episode}, Max Slot Load: {MAX_STUDENTS_PER_SLOT})")
        
        # Agents process CapacityUpdate
        for agent in classroom_agents:
            agent.handle_message(capacity_msg)
            
        # B. Commitment Fulfillment & Violation Check (at start of the day)
        for agent in classroom_agents:
            process_commitment_fulfillment(agent, episode)
        check_violations(classroom_agents) # Check if any past commitment was broken
        
        # C. Negotiation Phase
        rounds = 0
        while rounds < MAX_NEGOTIATION_ROUNDS:
            print(f"--- Negotiation Round {rounds + 1} ---")
            
            # Recalculate global load at the start of the round
            current_global_load = get_total_slot_load(classroom_agents)
            congested, congested_slot = check_for_congestion(current_global_load)
            
            if not congested:
                print("--- CONGESTION RESOLVED. Negotiation Ends. ---")
                break
            
            print(f"*** Congestion Detected at slot {congested_slot} (Load: {current_global_load[congested_slot]}, Max: {MAX_STUDENTS_PER_SLOT}) ***")
            
            if not run_negotiation_round(classroom_agents, episode):
                # No successful deal in this round (either no target, no available slot, or no acceptance)
                print("  > No successful deal completed in this round.")
                rounds += 1
            else:
                # A deal was made, continue to the next round with updated load
                rounds += 1 
                
        # D. Final Metrics Collection
        final_load = get_total_slot_load(classroom_agents)
        congested, _ = check_for_congestion(final_load)
        
        total_delay_minutes = 0
        for agent in classroom_agents:
            # Simple metric: calculate delay for students in +2, +4, ... slots
            for slot, students in agent.planned_exit_slots.items():
                if slot > 0:
                    total_delay_minutes += students * slot
                    
        total_commitments = sum(len(a.commitment_history) for a in classroom_agents)

        # Log results
        log_entry = {
            'episode': episode,
            'initial_congestion': congested and rounds == 0, # Placeholder, needs proper initial check
            'final_congested': congested,
            'max_load': max(final_load.values()) if final_load else 0,
            'total_delay_minutes': total_delay_minutes,
            'negotiation_rounds': rounds,
            # NEW: Store detailed slot distribution for each agent
            'agent_slots': {agent.name: agent.planned_exit_slots.copy() for agent in classroom_agents},
            # NEW: Store violation counts for all agents
            'agent_violations': {agent.name: agent.violations_count for agent in classroom_agents}
        }
        SIMULATION_LOG.append(log_entry)
        
        print(f"\n[SUMMARY EPISODE {episode}] Final Congested: {congested}. Total Delay: {total_delay_minutes} student-minutes.")
        print("-" * 60)

    # 3. Final Output
    # 3. Final Output (MODIFIED SECTION)
    print("\n\n--- SIMULATION COMPLETED ---")
    
    df = pd.DataFrame(SIMULATION_LOG)
    
    # --- Generate Detailed Student Distribution & Violation Table ---

    # 1. Process Agent Slot Data
    # This will create a list of DataFrames, one for each episode's agent distribution
    agent_slot_data = []
    for index, row in df.iterrows():
        # Create a DataFrame from the nested dictionary {'C1': {slot: students}, 'C2': ...}
        episode_df = pd.DataFrame.from_dict(row['agent_slots'], orient='index').fillna(0)
        episode_df['episode'] = row['episode']
        agent_slot_data.append(episode_df)

    # Combine all episodes into one DataFrame with agent names as the index
    dist_df = pd.concat(agent_slot_data).reset_index().rename(columns={'index': 'agent'})
    # Pivot to get the desired structure: episode | agent | Slot -X | Slot Y ...
    pivot_dist_df = dist_df.pivot(index='episode', columns='agent').fillna(0).astype(int)

    # 2. Process Violation Data
    violations_df = pd.json_normalize(df['agent_violations']).fillna(0).astype(int)
    violations_df['episode'] = df['episode']
    violations_df.set_index('episode', inplace=True)
    violations_df.columns = [f"Violations_{c}" for c in violations_df.columns]

    # 3. Combine into the Final Report
    # Combine core metrics with the new detailed slot and violation data
    core_metrics_df = df[['episode', 'final_congested', 'max_load', 'total_delay_minutes', 'negotiation_rounds']].set_index('episode')
    
    # The columns of pivot_dist_df are a MultiIndex, e.g., ('-2', 'C1'), ('0', 'C2'). Let's flatten it.
    pivot_dist_df.columns = [f"{agent}_Slot_{slot}" for slot, agent in pivot_dist_df.columns]
    
    # Join everything together
    final_report_df = core_metrics_df.join(pivot_dist_df).join(violations_df).reset_index()

    # 4. Calculate Total Students Exited for verification
    slot_columns = [col for col in final_report_df.columns if 'Slot' in col]
    final_report_df['Total Students Exited'] = final_report_df[slot_columns].sum(axis=1)
    
    # 5. Print the Cleaned, Detailed Table
    print("\n\n--- FINAL EPISODE SUMMARY & DETAILED TRAFFIC DISTRIBUTION ---")
    # Sort columns for readability: episode, core metrics, slots per agent, violations, total
    sorted_slot_cols = sorted([c for c in final_report_df.columns if 'Slot' in c])
    sorted_violation_cols = sorted([c for c in final_report_df.columns if 'Violations' in c])
    
    display_cols = ['episode', 'final_congested', 'max_load', 'negotiation_rounds'] + sorted_slot_cols + sorted_violation_cols + ['Total Students Exited']
    
    print(final_report_df[display_cols].to_string(index=False))

    history_df = get_commitment_history_summary(classroom_agents)
    print("\n\n--- DETAILED COMMITMENT HISTORY (By Episode and Status) ---")
    print(history_df.to_string(index=False))
if __name__ == "__main__":
    run_simulation()