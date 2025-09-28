# Exit-Flow
A Multi-agent system for managing student flows in and out of classrooms.

A distributed simulation system that models classroom exit flow coordination through autonomous agent negotiation. The system solves bottleneck congestion problems where multiple classrooms must coordinate student exit times through a shared corridor with limited capacity.

## What It Does

The simulation models three classroom agents (C1, C2, C3) that must coordinate student exit times to avoid congestion at a shared bottleneck. Agents negotiate with each other using a commitment-based protocol, redistributing exit slots while respecting individual professor scheduling constraints. The system demonstrates distributed problem-solving, social contract enforcement, and adaptive behavior under realistic constraints.

## Dependencies

- Python 3.8+
- pandas
- autogen

## Setup and Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Exit-Flow
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the simulation:**
```bash
python driver.py
```

## File Structure

- driver.py - Main simulation orchestration and analysis
- agents.py - Agent classes (BottleneckAgent, ClassroomAgent)
- protocols.py - Data structures (Message, Commitment, Adjustment)
- config.py - System parameters and constants

## Output

The simulation generates:
- Episode-by-episode negotiation logs
- Final traffic distribution table with per-agent slot assignments
- Commitment history tracking social contracts and violations
- Performance metrics (congestion resolution, delay times, violation counts)

## Key Features

- **Dynamic Attendance**: ±10% variation per episode
- **Heterogeneous Constraints**: Different professor policies per classroom
- **Social Contracts**: Commitment tracking with violation penalties
- **Distributed Coordination**: No central authority, pure agent negotiation