# ChronoCore™ Tutorial

**Learn ChronoCore in 30 Minutes**

This tutorial walks you through creating, simulating, and analyzing your first narrative using ChronoCore.

---

## Prerequisites

- Python 3.8 or higher
- Basic understanding of JSON format
- Familiarity with command line

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/PaniclandUSA/chronocore.git
cd chronocore
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If Plotly isn't installed, ChronoCore will automatically fall back to matplotlib.

---

## Part 1: Understanding the Basics

### What Am I Simulating?

ChronoCore simulates **narrative spacetime** - the physics of how story events, characters, and themes interact.

**Key Concepts:**

| Element | What It Is | Example |
|---------|-----------|---------|
| **Chronoton** | A story event with emotional mass | "Alice discovers Bob's betrayal" (M=8.5) |
| **Fermion** | A character following a path through the story | Alice, Bob |
| **Motif** | A theme existing in quantum superposition | "Love" (enduring vs. fleeting) |

---

## Part 2: Your First Simulation

Let's create a simple love story with 3 events and 2 characters.

### Step 1: Create the JSON File

Create a file called `my_story.json`:

```json
{
  "chronotons": [
    {
      "id": 0,
      "t": 0.1,
      "M": 5.0,
      "psi_M": [0.8, 0.2],
      "tags": ["first_meeting", "hope", "romantic"],
      "description": "Alice and Bob meet at a coffee shop",
      "observer": "omniscient"
    },
    {
      "id": 1,
      "t": 0.5,
      "M": 8.5,
      "psi_M": [0.3, 0.7],
      "tags": ["betrayal", "grief", "conflict"],
      "description": "Alice discovers Bob has been lying",
      "observer": "omniscient"
    },
    {
      "id": 2,
      "t": 0.9,
      "M": 6.0,
      "psi_M": [0.6, 0.4],
      "tags": ["reunion", "forgiveness", "resolution"],
      "description": "Alice and Bob reconcile",
      "observer": "omniscient"
    }
  ],
  "fermions": [
    {
      "name": "Alice",
      "shell": 1,
      "orbit": [0.5, 0.0, 0.0],
      "state": {
        "archetype": "dreamer",
        "motivation": "seeking_connection",
        "valence": 0.7,
        "agency": 0.6
      }
    },
    {
      "name": "Bob",
      "shell": 1,
      "orbit": [-0.5, 0.0, 0.0],
      "state": {
        "archetype": "realist",
        "motivation": "protecting_self",
        "valence": 0.4,
        "agency": 0.5
      }
    }
  ],
  "motifs": [
    {
      "name": "Love",
      "amplitude": [0.7, 0.7],
      "basis": ["enduring", "fleeting"]
    }
  ]
}
```

### Step 2: Understanding the Parameters

**Chronoton Fields:**
- `t` - Narrative time (0 = beginning, 1 = end)
- `M` - Emotional mass (0-10, higher = more impactful)
- `psi_M` - Motif superposition state (quantum weights)
- `tags` - Semantic labels for entanglement calculation

**Fermion Fields:**
- `shell` - PEPC energy level (characters in same shell can't be too similar)
- `orbit` - Starting position in narrative space [x, y, z]
- `state.archetype` - Character type
- `state.agency` - How much control they have (0-1)

**Motif Fields:**
- `amplitude` - Quantum superposition weights
- `basis` - Possible outcomes (will collapse to one)

### Step 3: Run the Simulation

```bash
python chronocore_v0_2_patched.py my_story.json
```

**Expected Output:**

```
======================================================================
ChronoCore™ v0.2 — Grindhouse Relativism Engine
======================================================================

✓ Loaded from my_story.json:
  3 chronotons
  2 fermions
  1 motifs

Initializing ChronoCore with 3 chronotons, 2 fermions, 1 motifs...
✓ Spacetime mesh computed: (100, 100, 50)
✓ Entanglement matrix built: (3, 3)

======================================================================
RUNNING SIMULATION
======================================================================

1. Computing character trajectories through curved spacetime...
   ✓ Alice: trajectory computed (150 points)
   ✓ Bob: trajectory computed (150 points)

2. Detecting PEPC violations...
   ✓ No PEPC violations detected

3. Collapsing motif wavefunctions...
   ✓ Love collapsed to: enduring (probabilities: [0.73 0.27])

4. Computing narrative coherence...
   ✓ Global Coherence Score: 0.885 (88.5%)

5. Entanglement statistics:
   Total possible entanglements: 3
   Strong entanglements (E > 0.7): 2
   Entanglement density: 66.7%

======================================================================
VISUALIZATION
======================================================================

✓ ChronoCore v0.2 simulation complete!
```

### Step 4: Interpret the Results

**Coherence Score: 88.5%**
- **>90%** = Exceptional (rare)
- **80-90%** = Strong (most good stories)
- **70-80%** = Acceptable (mainstream)
- **<70%** = Problems detected

**Motif Collapse: "enduring"**
- The quantum superposition collapsed to "enduring love"
- Probability: 73% (based on amplitude weights)

**Entanglement Density: 66.7%**
- 2 out of 3 possible event pairs are strongly entangled
- High entanglement = tight plot structure

---

## Part 3: Understanding the Visualization

The interactive 3D plot shows:

**Spacetime Curvature (Translucent Surface)**
- High-mass events create "gravity wells"
- Characters are pulled toward these wells

**Character Trajectories (Colored Lines)**
- Curved paths show how characters respond to events
- Straight line = unaffected by narrative gravity
- Steep curve = strong emotional response

**How to Interact:**
- **Rotate:** Click and drag
- **Zoom:** Scroll or pinch
- **Pan:** Shift + drag
- **Hover:** See event/character details

---

## Part 4: Diagnosing Problems

### Problem 1: Low Coherence (<70%)

**Possible Causes:**

1. **PEPC Violations** - Characters too similar
   ```
   ⚠ PEPC violation: Alice ↔ Bob (d=0.09)
   → Auto-resolving: Bob excited to shell 2
   ```
   **Fix:** Make characters more distinct (different motivations, archetypes)

2. **Weak Entanglement** - Events disconnected
   ```
   Entanglement density: 15.2%
   ```
   **Fix:** Add shared tags between related chronotons

3. **Causality Violations** - Time order broken
   ```
   Causality component: 0.43
   ```
   **Fix:** Ensure chronoton `t` values match story order

### Problem 2: PEPC Violations

**Example:**
```json
{
  "fermions": [
    {"name": "Alice", "shell": 1, "orbit": [0.1, 0, 0], "state": {"archetype": "hero"}},
    {"name": "Bob", "shell": 1, "orbit": [0.12, 0, 0], "state": {"archetype": "hero"}}
  ]
}
```

**Issue:** Both are shell 1, distance = 0.02 (threshold = 0.15)

**Fix Options:**
1. Change shell: Make Bob shell 2
2. Increase distance: Move orbits farther apart
3. Change archetype: Make Bob a different character type

### Problem 3: Flat Trajectories

If character trajectories are nearly straight lines:

**Possible Causes:**
1. **Low emotional mass** - Increase chronoton `M` values
2. **Characters too far** - Move fermion `orbit` closer to events
3. **Low grid resolution** - Increase in code (default 100)

---

## Part 5: Advanced Techniques

### Technique 1: Fine-Tuning Entanglement

**Goal:** Create strong setup-payoff relationships

```json
{
  "chronotons": [
    {
      "id": 0,
      "tags": ["promise", "trust", "commitment"]  ← Setup
    },
    {
      "id": 5,
      "tags": ["betrayal", "trust", "violation"]  ← Payoff (shares "trust")
    }
  ]
}
```

**Result:** High entanglement between events 0 and 5

### Technique 2: Quantum Motif Tuning

**Goal:** Control collapse probabilities

```json
{
  "motifs": [
    {
      "name": "Justice",
      "amplitude": [0.9, 0.3],  ← 90% chance "served", 10% chance "denied"
      "basis": ["served", "denied"]
    }
  ]
}
```

**Math:** Probability = |amplitude|² / Σ|amplitude|²

### Technique 3: Hierarchical Character Shells

**Goal:** Create character relationships through PEPC

```json
{
  "fermions": [
    {"name": "King", "shell": 3},      ← Highest authority
    {"name": "Prince", "shell": 2},    ← Mid authority
    {"name": "Peasant", "shell": 1}    ← Low authority
  ]
}
```

**Result:** Shell number encodes power dynamics

---

## Part 6: Common Patterns

### Pattern 1: Three-Act Structure

```json
{
  "chronotons": [
    {"id": 0, "t": 0.15, "M": 5.0, "tags": ["setup"]},
    {"id": 1, "t": 0.25, "M": 6.0, "tags": ["inciting_incident"]},
    {"id": 2, "t": 0.50, "M": 8.5, "tags": ["midpoint", "reversal"]},
    {"id": 3, "t": 0.75, "M": 9.0, "tags": ["climax"]},
    {"id": 4, "t": 0.95, "M": 4.0, "tags": ["resolution"]}
  ]
}
```

**Key Points:**
- `t` values follow act structure (0-0.25, 0.25-0.75, 0.75-1.0)
- `M` peaks at climax (id 3)
- Resolution has lower mass (denouement)

### Pattern 2: Character Arc

```json
{
  "fermions": [
    {
      "name": "Hero",
      "orbit": [-1.0, -0.5, 0.0],  ← Starting position (negative)
      // Trajectory will curve toward high-M chronotons
      // Ending position will be more positive (character growth)
    }
  ]
}
```

### Pattern 3: Dual Protagonists

```json
{
  "fermions": [
    {"name": "Alice", "shell": 1, "orbit": [0.8, 0.0, 0.0]},   ← Start far apart
    {"name": "Bob", "shell": 1, "orbit": [-0.8, 0.0, 0.0]}
  ],
  "chronotons": [
    {"id": 5, "t": 0.5, "M": 9.0, "tags": ["convergence"]}  ← Meeting point
  ]
}
```

**Result:** Trajectories converge at t=0.5

---

## Part 7: Using Python Directly

### Basic Python Usage

```python
from chronocore import ChronoCore

# Load story
core = ChronoCore.from_json("my_story.json")

# Run simulation
for character in core.fermions:
    core.geodesic_drift(character, steps=150)

# Check for violations
violations = core.detect_pepc_violations()
if violations:
    print(f"Found {len(violations)} PEPC violations")

# Collapse motifs
for i, motif in enumerate(core.motifs):
    outcome = core.collapse_motif(i, observer_bias=0.0)
    print(f"{motif.name} → {outcome}")

# Calculate coherence
C = core.coherence_score()
print(f"Narrative Coherence: {C:.1%}")

# Visualize
core.visualize()
```

### Advanced: Manual Construction

```python
import numpy as np
from chronocore import ChronoCore, Chronoton, CharacterFermion, MotifBoson

# Create chronotons
chronotons = [
    Chronoton(
        id=0,
        t=0.1,
        M=5.0,
        psi_M=np.array([0.8, 0.2]),
        tags=["first_meeting"],
        description="They meet"
    )
]

# Create characters
fermions = [
    CharacterFermion(
        name="Alice",
        shell=1,
        orbit=np.array([0.5, 0.0, 0.0]),
        state={"archetype": "hero"}
    )
]

# Create motifs
motifs = [
    MotifBoson(
        name="Love",
        amplitude=np.array([0.7, 0.7]),
        basis=["enduring", "fleeting"]
    )
]

# Initialize engine
core = ChronoCore(chronotons, fermions, motifs, grid_resolution=100)

# Run simulation
# ... (same as above)
```

### Advanced: Custom Analysis

```python
# Access entanglement matrix
E = core.entanglement_matrix
print(f"Strongest bond: {E.max():.3f}")

# Access spacetime curvature
R = core.spacetime_mesh
print(f"Max curvature: {R.max():.1f}")

# Access character trajectories
for f in core.fermions:
    traj = f.trajectory
    distance_traveled = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    print(f"{f.name} traveled: {distance_traveled:.2f} narrative units")

# Export results
core.export_json("simulation_results.json")
```

---

## Part 8: Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'chronocore'"

**Solution:**
```bash
# Make sure you're in the chronocore directory
cd chronocore

# Run with full path
python chronocore_v0_2_patched.py my_story.json
```

### Issue: "Plotly not available"

**Solution:**
```bash
pip install plotly
```

Or accept the matplotlib fallback (static visualization).

### Issue: "Geodesic integration failed"

**Cause:** Invalid initial conditions or extreme curvature

**Solution:**
1. Check fermion `orbit` values are reasonable ([-2, 2] range)
2. Reduce chronoton `M` values if too high (>10)
3. Increase grid resolution in code

### Issue: Simulation is slow

**Solutions:**
1. Reduce grid resolution: `ChronoCore(..., grid_resolution=50)`
2. Reduce integration steps: `core.geodesic_drift(f, steps=50)`
3. Wait for v0.4 GPU acceleration

---

## Part 9: Next Steps

### Explore Example Files

```bash
python chronocore_v0_2_patched.py examples/love_story.json
python chronocore_v0_2_patched.py examples/pride_and_prejudice.json
python chronocore_v0_2_patched.py examples/grindhouse_genesis.json
```

### Read the Theory

- **[PHYSICS.md](docs/PHYSICS.md)** - Mathematical formalism
- **[ROADMAP.md](ROADMAP.md)** - Future features (v0.3-v1.0)
- **[White Paper](docs/WHITEPAPER.md)** - Complete theory (coming soon)

### Join the Community

- **GitHub Issues:** Report bugs or request features
- **GitHub Discussions:** Ask questions, share stories
- **Esper Stack:** Explore VSE and PICTOGRAM integration

### Contribute

- Add example narratives
- Improve documentation
- Submit bug fixes
- Propose new features

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon)

---

## Part 10: Pro Tips

### Tip 1: Start Small

Don't try to simulate your entire novel on day 1. Start with:
- 3-5 chronotons (key events only)
- 2-3 fermions (main characters)
- 1-2 motifs (core themes)

Then expand once you understand the patterns.

### Tip 2: Use Tags Strategically

Tags drive entanglement. Shared tags = strong connections.

**Good tagging:**
```json
{"tags": ["promise", "trust"]}        ← Setup
{"tags": ["betrayal", "trust"]}       ← Payoff (shares "trust")
```

**Bad tagging:**
```json
{"tags": ["event_1"]}                 ← No semantic meaning
{"tags": ["thing_happens"]}
```

### Tip 3: Mass = Audience Impact

Chronoton `M` should reflect how much the audience *feels* the event:
- **M = 2-4** - Minor events (conversations, transitions)
- **M = 5-7** - Significant events (revelations, conflicts)
- **M = 8-10** - Major events (deaths, betrayals, climaxes)

### Tip 4: Iterate

First run almost always reveals issues. That's the point!

**Process:**
1. Run simulation
2. Check coherence score
3. Fix violations
4. Run again
5. Repeat until C > 0.85

### Tip 5: Trust the Physics

If ChronoCore says there's a PEPC violation, there probably is character redundancy even if you can't see it yet.

If it says entanglement is weak, that setup-payoff relationship probably isn't landing.

**The math doesn't lie.**

---

## Conclusion

You now know how to:
- ✅ Create narrative JSON files
- ✅ Run ChronoCore simulations
- ✅ Interpret coherence scores
- ✅ Diagnose structural problems
- ✅ Visualize narrative spacetime
- ✅ Use ChronoCore in Python

**Next:** Try simulating your own story and see what ChronoCore reveals!

---

**Questions?** Open an issue on GitHub or start a discussion.

**Want to contribute?** We'd love your help! See CONTRIBUTING.md.

---

*"Stories obey physics. ChronoCore computes the laws."*

**Happy simulating!** 🚀✨
