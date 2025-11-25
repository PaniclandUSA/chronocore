# Changelog

All notable changes to ChronoCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned for v0.3 (Q1 2026)
- Quantum entangled state vectors
- Narrative action functional
- Multi-observer collapse mechanics
- Bell inequality validation tests

### Planned for v0.4 (Q2 2026)
- GPU acceleration with CuPy
- Hierarchical chunking for N > 1000
- WebSocket collaborative API
- Motif field lattice (QFT-style)

### Planned for v1.0 (Q3 2026)
- Biometric Adaptive Resolution Engine (ARE)
- Unity/Unreal engine plugins
- LLM integration layer
- Production deployment tools

---

## [0.2.0] - 2025-11-24

### Added - Major Features
- **Interactive Plotly visualization** - Rotatable 3D spacetime plots with isosurfaces
- **Universal JSON loader** - `ChronoCore.from_json(path)` class method
- **JSON export** - `core.export_json(path)` for saving results
- **Command-line interface** - Run simulations directly from terminal
- **Comprehensive documentation** - README, TUTORIAL, PHYSICS, ROADMAP, CONTRIBUTING, FAQ

### Changed - Performance Improvements
- **50Ã— faster curvature computation** - Vectorized NumPy operations (Grok contribution)
- **Smooth interpolated geodesics** - scipy.ndimage.map_coordinates (no more blocky artifacts)
- **Optimized Plotly sampling** - Caps at 40Â³ points to prevent lag
- **Fixed boundary overflow** - Coordinate mapping edge case resolved

### Changed - Code Quality
- **Type hints** - Added throughout codebase
- **Enhanced error handling** - Graceful fallbacks and warnings
- **Modular architecture** - Separated concerns for maintainability
- **Comprehensive docstrings** - All public methods documented

### Changed - User Experience
- **Rich console output** - Progress indicators and formatted results
- **Automatic fallback** - matplotlib if Plotly unavailable
- **Better error messages** - Clear guidance when something fails

### Validation
- âœ… **Grok (xAI)** - Performance validation ("Seal of Approval")
- âœ… **Gemini (Google)** - Physics validation ("Production-ready")
- âœ… **Claude (Anthropic)** - Architecture and implementation

---

## [0.1.0] - 2025-11-15

### Added - Initial Release
- **Core physics engine** - Narrative spacetime simulation
- **Chronotons** - Event particles with emotional mass
- **Character Fermions** - Matter following geodesics
- **Motif Bosons** - Force-carrying thematic fields
- **PEPC enforcement** - Pauli Exclusion Principle for Characters
- **Entanglement matrix** - Non-local event correlations
- **Coherence scoring** - Quantitative narrative integrity metric
- **Matplotlib visualization** - Static 3D plots
- **Example narratives** - love_story.json, pride_and_prejudice.json
- **JSON schema** - Validation schema for story files

### Implementation Details
- **Ricci scalar proxy** - R â‰ˆ 8Ï€GT approximation
- **Geodesic integration** - scipy.odeint solver
- **PEPC detection** - Distance threshold violations
- **Motif collapse** - Quantum measurement simulation

### Documentation
- Basic README.md
- LICENSE files (Apache 2.0 + CC BY 4.0)
- schema.json specification

### Known Limitations
- Loop-based curvature computation (slow)
- Blocky geodesic trajectories (rounding artifacts)
- Static visualization only
- Manual JSON authoring required
- No GPU acceleration
- Limited to ~200 chronotons before slowdown

---

## [0.0.1] - 2025-11-01

### Proof of Concept
- Initial theoretical framework
- Grindhouse Relativism white paper (72 pages)
- Basic Python prototype
- Concept validation with simple narratives

---

## Version Numbering

ChronoCore follows [Semantic Versioning](https://semver.org/):

**MAJOR.MINOR.PATCH**

- **MAJOR** - Incompatible API changes
- **MINOR** - New features (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

### Version Milestones

| Version | Status | Focus | Target Date |
|---------|--------|-------|-------------|
| 0.1.0 | âœ… Released | Proof of concept | 2025-11-15 |
| 0.2.0 | âœ… Released | Performance + UX | 2025-11-24 |
| 0.3.0 | ðŸ”„ In Progress | Quantum mechanics | Q1 2026 |
| 0.4.0 | ðŸ“‹ Planned | Production features | Q2 2026 |
| 1.0.0 | ðŸ“‹ Planned | Stable release | Q3 2026 |

---

## Release Philosophy

### Alpha (0.1-0.2)
- **Purpose:** Validate core concepts
- **Stability:** Experimental, expect breaking changes
- **Audience:** Early adopters, researchers
- **Focus:** Proof of physics model

### Beta (0.3-0.4)
- **Purpose:** Production features
- **Stability:** API stabilizing, fewer breaking changes
- **Audience:** Professional users, studios
- **Focus:** Performance, scalability, integrations

### Stable (1.0+)
- **Purpose:** Production deployment
- **Stability:** Stable API, semantic versioning guarantees
- **Audience:** Everyone
- **Focus:** Reliability, ecosystem, long-term support

---

## Deprecation Policy

Starting with v1.0:
- **Deprecated features** will be marked in docs
- **Removal** will occur at next MAJOR version
- **Warning period** of at least 6 months
- **Migration guides** will be provided

---

## Contributing to Changelog

When submitting PRs:
1. Add entry to **[Unreleased]** section
2. Use categories: Added, Changed, Deprecated, Removed, Fixed, Security
3. Link to issue/PR number
4. Describe user-facing impact

Example:
```markdown
### Added
- Interactive visualization with Plotly (#123) - @username
```

---

## Links

- **Repository:** https://github.com/PaniclandUSA/chronocore
- **Issues:** https://github.com/PaniclandUSA/chronocore/issues
- **Releases:** https://github.com/PaniclandUSA/chronocore/releases
- **Esper Stack:** https://github.com/PaniclandUSA/esper-stack

---

*"Stories obey physics. ChronoCore computes the laws."*
