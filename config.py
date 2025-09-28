# config.py (Updated for debugging)

# --- ENVIRONMENT & SCENARIO CONSTANTS ---
N_CLASSROOMS = 3 # Reduced to 3 agents
LECTURE_NOMINAL_END_TIME = "11:00 AM" 

# Bottleneck Agent B Parameters
BOTTLENECK_CAPACITY_PER_MINUTE = 30 
SLOT_DURATION_MINUTES = 2          
MAX_STUDENTS_PER_SLOT = BOTTLENECK_CAPACITY_PER_MINUTE * SLOT_DURATION_MINUTES # 60 students

# Classroom Agent C_i Parameters
# Use a subset of the original attendance for N=3
INITIAL_ATTENDANCE = [40, 55, 60] # C1 (40), C2 (55), C3 (60)
VIOLATION_LIMIT = 7 
MAX_NEGOTIATION_ROUNDS = 10 

# --- SIMULATION PARAMETERS ---
SIMULATION_EPISODES = 4 # Reduced to 3 episodes