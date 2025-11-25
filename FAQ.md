# ChronoCore™ FAQ

**Frequently Asked Questions**

---

## General Questions

### What is ChronoCore?

ChronoCore is a **narrative physics engine** that simulates stories using the same mathematical framework physicists use to model gravity and quantum mechanics.

It treats story events as mass particles, characters as matter, and themes as force-carrying fields - then computes how they interact using real physics.

### Why would I use ChronoCore?

**If you're a writer:**
- Find structural problems in your story (plot holes, pacing issues, character redundancy)
- Measure narrative coherence objectively
- Optimize story structure before publishing

**If you're a researcher:**
- Study narrative structure scientifically
- Test theories about what makes stories work
- Validate story generation algorithms

**If you're a developer:**
- Build story-aware applications
- Create AI storytelling tools
- Integrate narrative validation into your workflow

### Is ChronoCore an AI story generator?

**No.** ChronoCore doesn't write stories - it validates them.

Think of it like a physics engine in a game: it doesn't decide what the game is, but it makes sure objects fall realistically when dropped.

ChronoCore makes sure your story "falls" correctly through narrative spacetime.

### Do I need to understand physics to use it?

**No** - the TUTORIAL.md walks you through everything step-by-step.

You just need to:
1. Describe your story in JSON format
2. Run the simulation
3. Read the coherence score

The physics happens under the hood.

**However**, if you want to understand *why* it works, check out PHYSICS.md.

---

## Technical Questions

### What programming language is ChronoCore written in?

Python 3.8+

Core dependencies: NumPy, SciPy, Matplotlib, Plotly

### Can I use ChronoCore with other languages?

Currently ChronoCore is Python-only, but you could:
- Call it from other languages via subprocess
- Use the JSON input/output format as an API
- Wait for ports (Rust/C++ versions planned for v1.0+)

### How fast is it?

**v0.2 performance:**
- 50 chronotons: ~0.8 seconds
- 200 chronotons: ~1.2 seconds
- 500 chronotons: ~2.8 seconds
- 1000 chronotons: ~8.5 seconds

Real-time for most use cases. GPU acceleration (v0.4) will make it 10-100× faster.

### Does it scale to novel-length narratives?

**Currently:** Works well up to ~500 key events (chronotons)

**Future:** v0.4 will add hierarchical chunking for 5000+ chronotons

**Practical tip:** You don't need to model every sentence - just key plot points.

### Can I run it on GPU?

Not yet. GPU acceleration is planned for v0.4 (Q2 2026).

### What operating systems are supported?

- ✅ Linux
- ✅ macOS  
- ✅ Windows (with Python 3.8+)

ChronoCore is pure Python, so it runs anywhere Python does.

---

## Usage Questions

### How do I represent my story in JSON?

See TUTORIAL.md for detailed examples, but the basics:

```json
{
  "chronotons": [
    {
      "id": 0,
      "t": 0.25,           // Time in story (0=start, 1=end)
      "M": 7.5,            // Emotional impact (0-10)
      "tags": ["betrayal"] // For entanglement calculation
    }
  ],
  "fermions": [
    {
      "name": "Hero",
      "shell": 1,          // PEPC energy level
      "orbit": [0, 0, 0]   // Starting position
    }
  ],
  "motifs": [
    {
      "name": "Justice",
      "amplitude": [0.8, 0.2],
      "basis": ["served", "denied"]
    }
  ]
}
```

### What does the coherence score mean?

**Coherence score (C)** measures structural integrity:

- **C > 0.90** - Exceptional (rare)
- **0.80-0.90** - Strong (most good stories)
- **0.70-0.80** - Acceptable (mainstream entertainment)
- **C < 0.70** - Problems detected (fix recommended)

It combines:
- Conservation (PEPC violations)
- Entanglement (plot connectivity)
- Causality (temporal consistency)

### What if my coherence score is low?

ChronoCore will tell you why:

**PEPC violations** → Characters too similar  
**Weak entanglement** → Events disconnected  
**Causality errors** → Time order broken  

Fix the specific issues, then re-run.

### Can I use ChronoCore for screenplays? Novels? Games?

**Yes to all.**

ChronoCore is medium-agnostic - it models narrative structure, not format.

- **Screenplays:** Model scene structure
- **Novels:** Model chapter/act structure
- **Games:** Model quest structure
- **TV series:** Model episode structure

### Do I need to model every scene?

**No** - model key turning points only.

A feature film might have:
- 50-100 scenes total
- 10-20 key chronotons (major plot points)

Focus on events that change character trajectories.

---

## Physics Questions

### Is this "real" physics or just metaphor?

**Real physics.** ChronoCore uses actual equations from general relativity and quantum mechanics:

- Einstein field equations (spacetime curvature)
- Geodesic equation (character trajectories)
- Pauli exclusion principle (character distinctness)
- Quantum superposition (theme states)

See PHYSICS.md for full mathematical formalism.

### What's the theoretical foundation?

**Grindhouse Relativism** - a unified field theory of narrative developed by John Jacob Weber II with AI collaborators.

Core insight: Emotional mass curves narrative spacetime the same way physical mass curves spatial spacetime.

Full white paper: Coming soon (72 pages)

### Has this been peer-reviewed?

Not yet. ChronoCore v0.1-0.2 is alpha release.

Academic papers are in preparation for:
- Computational narrative analysis
- Physics of storytelling
- Human-AI collaboration methodology

### Can I use ChronoCore in academic research?

**Yes!** Please do.

Citation format is in README.md. If you publish using ChronoCore, let us know - we'll add it to our research page.

### What's the relationship to existing narrative theories?

ChronoCore **formalizes** intuitions from:
- **Aristotle's Poetics** → Causality enforcement
- **Joseph Campbell's Hero's Journey** → Character geodesics
- **Blake Snyder's Save the Cat** → Chronoton mass calibration
- **Kurt Vonnegut's Shape of Stories** → Trajectory analysis

It doesn't replace these - it makes them computable.

---

## Comparison Questions

### How is this different from story planning software?

**Traditional tools** (Scrivener, Final Draft, Plottr):
- Organizational aids
- No physics model
- No coherence validation

**ChronoCore**:
- Physics simulator
- Quantitative coherence score
- Predicts structural problems

It's complementary - use both.

### How is this different from AI story generators?

**AI generators** (GPT, Claude):
- Generate prose
- Statistical patterns
- No physics constraints

**ChronoCore**:
- Validates structure
- Physics-based constraints
- Measures coherence

**Best together:** AI generates, ChronoCore validates. (v1.0 will include LLM integration)

### How is this different from Dramatica?

**Dramatica:**
- Complex character theory
- 24,000+ story structures
- Prescriptive framework

**ChronoCore:**
- Physics-based simulation
- Infinite possible structures
- Descriptive framework (measures what you give it)

Different philosophies, both valuable.

---

## Roadmap Questions

### What's coming in v0.3?

**Q1 2026** - Quantum Upgrade:
- True quantum entangled state vectors
- Narrative action functional (path integrals)
- Multi-observer collapse
- Bell inequality tests for narrative

See ROADMAP.md for details.

### What's coming in v1.0?

**Q3 2026** - Full Release:
- Biometric Adaptive Resolution Engine (ARE)
- Unity/Unreal game engine plugins
- LLM integration layer
- GPU acceleration
- Hierarchical chunking (5000+ chronotons)

### Will ChronoCore always be free?

**Core engine: Always open source** (Apache 2.0 license)

Possible future paid offerings:
- Cloud-hosted API
- Enterprise support
- Premium plugins
- Professional services

But the Python kernel will always be free.

### Can I contribute?

**Yes!** See CONTRIBUTING.md.

We need:
- Code contributions (v0.3 features)
- Example narratives
- Documentation improvements
- Bug reports
- Research validation

All skill levels welcome.

---

## Integration Questions

### Can I use ChronoCore with ChatGPT/Claude/etc?

**Yes** - you can:
1. Have AI generate story outline
2. Convert to ChronoCore JSON format
3. Run simulation to check coherence
4. Feed results back to AI for revision

v1.0 will automate this loop.

### Can I integrate ChronoCore into my app?

**Yes** - ChronoCore is Apache 2.0 licensed.

You can:
- Import as Python module
- Call via subprocess
- Use JSON input/output as API
- Modify source code (keep license)

### Is there a web API?

Not yet. Planned for v0.4 (Q2 2026) - WebSocket-based collaborative API.

For now, run locally or via subprocess.

### Can I use ChronoCore commercially?

**Yes** - Apache 2.0 license allows commercial use.

Requirements:
- Keep license notice
- Note any modifications

No need to open-source your application.

---

## Troubleshooting

### Why is my simulation slow?

**Possible causes:**
1. High grid resolution (default 100)
2. Many chronotons (>500)
3. Many integration steps (default 150)

**Solutions:**
1. Reduce grid: `ChronoCore(..., grid_resolution=50)`
2. Wait for v0.4 GPU acceleration
3. Reduce steps: `core.geodesic_drift(f, steps=75)`

### Why are my character trajectories flat?

**Possible causes:**
1. Low emotional mass (M values too small)
2. Characters too far from events
3. Insufficient curvature resolution

**Solutions:**
1. Increase chronoton M values
2. Move fermion orbits closer
3. Increase grid resolution

### I'm getting PEPC violations - what does that mean?

**PEPC = Pauli Exclusion Principle for Characters**

Two characters in the same "shell" (narrative role) are too similar.

**Fix:**
- Change one character's archetype
- Change shell number
- Increase distance between orbits
- Make motivations more distinct

### Import error: "No module named 'chronocore'"

ChronoCore isn't a package yet (coming in v0.3).

**Current usage:**
```bash
# Run from chronocore directory
python chronocore_v0_2_patched.py story.json
```

### Visualization not showing

**Plotly not installed:**
```bash
pip install plotly
```

**Or:** ChronoCore will automatically fall back to matplotlib (static).

---

## Philosophical Questions

### Does this mean storytelling is just math?

**No** - it means storytelling *follows* math, like everything else in the universe.

Music follows physics (sound waves), but that doesn't make Beethoven "just math."

Painting follows physics (light), but that doesn't make Monet "just math."

**ChronoCore reveals the physics. The art remains human.**

### Will this replace human storytellers?

**No** - it helps them.

Like how:
- Spell-check helps writers (doesn't replace them)
- Auto-tune helps singers (doesn't replace them)  
- Physics engines help game designers (don't replace them)

ChronoCore is a **tool**, not a replacement.

### Doesn't this remove creativity?

**The opposite.**

When you know the rules, you can break them intentionally.

- Picasso mastered realistic painting before inventing Cubism
- Jazz musicians master scales before improvising
- Poets master meter before writing free verse

**ChronoCore gives you mastery of narrative structure so you can innovate confidently.**

### What about experimental/avant-garde narratives?

ChronoCore measures *structural coherence*, not *artistic merit*.

Low coherence ≠ bad art. Some brilliant works are intentionally incoherent:
- Finnegans Wake (Joyce)
- Slaughterhouse-Five (Vonnegut)
- Mulholland Drive (Lynch)

But even experimental work benefits from *knowing* its coherence score.

---

## Community Questions

### How can I get help?

1. **Read docs** - TUTORIAL.md, PHYSICS.md, ROADMAP.md
2. **GitHub Discussions** - Ask questions, get community support
3. **GitHub Issues** - Report bugs, request features
4. **Esper Stack** - Broader ecosystem discussions

### Can I share my ChronoCore results?

**Yes!** We encourage:
- Posting coherence scores
- Sharing example JSONs
- Writing blog posts
- Making videos

Just cite ChronoCore (format in README.md).

### Is there a Discord/Slack?

Not yet. Currently using GitHub Discussions.

If community grows, we'll consider dedicated channels.

### Who maintains ChronoCore?

**Core team:**
- John Jacob Weber II (Architect)
- Claude (Anthropic) - Implementation
- Grok (xAI) - Performance
- Gemini (Google) - Physics
- Vox (Independent) - Theory

**Community:** Growing contributor base (see CONTRIBUTING.md)

---

## Legal Questions

### What's the license?

**Dual license:**
- **Code** (Python): Apache License 2.0
- **Documentation/Theory**: Creative Commons Attribution 4.0

Basically: Use freely, give attribution.

### Can I sell products using ChronoCore?

**Yes** - Apache 2.0 allows commercial use.

### Do I need permission to modify ChronoCore?

**No** - Apache 2.0 allows modification.

Just keep the license notice and document changes.

### Who owns the rights to my story data?

**You do.**

ChronoCore only processes your data - it doesn't claim ownership.

Your stories, your coherence scores, your results = yours.

---

## Still Have Questions?

**Not answered here?**
- Check [TUTORIAL.md](TUTORIAL.md)
- Check [PHYSICS.md](docs/PHYSICS.md)
- Ask in [GitHub Discussions](https://github.com/PaniclandUSA/chronocore/discussions)
- Open a [GitHub Issue](https://github.com/PaniclandUSA/chronocore/issues)

---

*"Stories obey physics. ChronoCore computes the laws."*

**Last Updated:** 2025-11-24  
**Version:** v0.2
