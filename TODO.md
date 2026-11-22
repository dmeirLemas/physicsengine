# TODO

Status as of 2026-11-20.

## Where we are

- **Module 1, massive-body systems:** done. Scenes, scene editor (Ctrl+click
  placement, Make orbit with the Hill-radius rule, live editing), trails,
  look-ahead paths, adaptive time steps, merging on contact, Restart / Full
  reset. Tested (`tests/test_nbody*.py`).
- **Module 2, galaxy collisions:** working and tested
  (`tests/test_galaxy_*.py`).
  - **Physics:** self-gravitating galaxies (Hernquist halo and bulge,
    exponential disc, Jeans and Toomre-Q kinematics, rotation curve measured
    from the tree), massless tracer stars, four encounter scenes plus a
    single galaxy.
  - **Forces:** Barnes-Hut tree, built in parallel on the CPU and walked on
    the GPU.
  - **Rendering:** HDR star renderer with bloom.
  - **Speed:** 10 time units/s (47 Myr/s) for a two-galaxy merger, the top of
    the time-scale slider.
  - **Tried and dropped:** block (individual) time steps gave no speed-up for
    worse energy conservation. Code deleted, measurements in the session notes.
- **Module 2:** closed 2026-11-20 (README skipped by decision; known
  limitations listed under "Galaxies" below).
- **Module 3, black holes:** working (Claude under override; review the
  physics). Schwarzschild and Kerr ray tracing with a Page-Thorne disc;
  bodies on exact geodesics, lensed, redshifted, with light-travel time;
  plunging body and tidal disruption; 1PN and black holes in the N-body
  module. Details and checks in "Module 3, black holes" below.
- **Module 4, particle physics:** not started.

## Galaxies: still to do

- [x] Customisable galaxy setup (2026-11-20, Claude under override; review
      the physics): `GalaxyModel` per galaxy (galaxy_components.py), `Encounter`
      with any eccentricity and Toomre angles (i, omega) per disc
      (galaxy_encounters.py), "Galaxy setup" window with Apply / Revert /
      Preset (galaxy_editor.py). Restart keeps edits, Full reset drops them.
- [x] Bound orbit: "Bound prograde" scene (e = 0.8, start 70 kpc) and the
      eccentricity slider.
- [x] Speed (2026-11-20): GPU_THETA 1.0 and dt 0.16 (was 0.6 / 0.08). The
      app now holds time scale 10 (10 time units/s, 47 Myr/s; was 3.5-3.8).
      Energy error over 905 Myr 1.5% (was 0.34%); separations, tails and
      pictures match the reference run. Accuracy setting if ever needed:
      0.6 / 0.08. At time scale 1 the picture updates ~6 times a second.
- [x] Core loop: a step slower than `frame_budget` no longer drops the
      fraction of a step left in the accumulator; the speed readout is
      (time asked for - time dropped) / wall time over 0.5 s.
- [x] Orbit set from the galaxies' mass profiles (galaxy_orbit.py, Claude
      under override; review the physics): centres as test bodies,
      g = G (M1(<d) + M2(<d)) / d^2. Pericentre 10 kpc gives a bulge passage
      at 8.6 kpc (Kepler set-up: 18.7; rigid spheres: 4.5). Prograde merges
      after 1.3 Gyr, Bound prograde after 0.5 Gyr. Bound orbits use
      e = (r_a - r_p) / (r_a + r_p).
- [ ] An isolated galaxy's bulge wanders 1-3 kpc around its halo's centre of
      mass (4000 halo particles): limits how exactly a passage can be set.
- [ ] Bulge cores puff up (10% radius 0.27 -> 0.6 kpc in 900 Myr), also in
      the reference run: a resolution effect like the disc heating below.
- [ ] Camera option to follow the colliding pair (centre of mass), as in the
      N-body module.
- [ ] dt = 0.16 is fixed; the editor lets you build denser bulges than it was
      checked for (no warning shown).
- [ ] Decide on vsync (off measured about +7% simulation speed; core setting).
- [ ] Disc heating from the 10x heavier halo particles (disc thickness doubles
      in ~240 Myr in isolation). More halo particles would reduce it, at a
      speed cost.

## General

- [ ] Commit the work since the last commit (N-body editor onwards, the whole
      galaxy module, GPU code, tests).
- [ ] Short README per module: what it shows, controls, physics model with
      references.
- [ ] Module 1, optional: first-order post-Newtonian correction for perihelion
      precession.

## Module 3, black holes: next

Decided 2026-11-20: black-hole module = exact Schwarzschild geodesics plus
Newtonian pulls between bodies; N-body module = 1PN (EIH) with a black-hole
body type; bodies drawn lensed; both a plunging body and a tidal disruption.

- [x] Full reset restores every setting to its default in all modules
      (core/app.py; N-body and black-hole editor settings reset too).
- [x] Step A: bodies on exact geodesics (`geodesics.py`), lensed and
      redshifted images (`blackhole_images.py`), scenes (precession, ISCO,
      plunging body, custom) and a Bodies window to add orbits. Checked:
      precession equals the exact integral to 1e-7, E and L conserved to
      1e-13, ISCO stability, images cross-checked against the shader's disc
      pixels (1e-3 deg, g to 4e-6), flat-space size limit.
- [x] Light-travel time (on/off): each image shows the body where it was
      when its light left (exact delay by quadrature over u; body histories
      prefilled by backward integration).
- [x] Tidal disruption ("Star torn apart"): Plummer star of 2000 particles
      with Newtonian self-gravity, each on its exact geodesic. Mass ratio
      1e-3, r_t = 20 M, pericentre 4.5 M (beta = 4.4): ~45% swallowed during
      the passage, ~42% flung out, ~13% bound (checked). A star swallowed
      whole was tried and dropped: such a dense star relaxes (swells) in a
      particle model long before it arrives.
- [x] "Star torn apart, with gas": eccentric orbit (e = 0.6, pericentre
      8 M); superseded on 2026-11-22 by the SPH version below.
- [x] N-body module: 1PN (EIH) gravity (default; Newton selectable), black-
      hole bodies in the editor (horizon radius, absorbed on contact), scenes
      "S2 around Sgr A*", "Star near a black hole", "Black hole flyby".
      Checked: Mercury 42.99"/century, S2 12.26'/orbit (both = GR), 1PN
      energy conserved to 2PN order.
- [x] N-body fixes, 2026-11-22 (Claude; review the integrator change in
      nbodystate.py): substep level may change inside a step (binary grid,
      up to dt / 2^40; before: one level per step, max 256 substeps, so deep
      encounters jumped); at most 600 substeps per step, then slow motion.
      1PN refinement moved from a per-step ETA switch into the timescale
      (the switch broke time symmetry: energy errors of order one). Black
      holes are flagged (NBodyState.black_hole): capture within 6 G M / c^2,
      always survive a merger, horizon grows with mass. Predicted path of the
      selected body is cached and extended (was recomputed every frame, 30-
      400 ms). pair_timescales and first_contact compiled (steps 3-4x faster).
- [ ] Not reproduced: crash after hovering the Follow button (probably the
      stall above; get the terminal output if it happens again).
- [x] 2026-11-22 (Claude under override; review the physics):
      - Black-hole module: "Hole mass" slider (1e4-1e9 M_sun) and star type
        (Sun-like, white dwarf) set the star's size and mass in units of M.
        The star moves as one body until its tidal radius, then its
        particles are released (blackhole_star.py); inside the capture
        radius it goes in whole. "Star torn apart" is now a plunge (L = 3.9).
      - Gas: SPH (blackhole_sph.py; pressure, shock heating, Balsara switch,
        optional beta cooling) replaces the sticky-particle model
        (blackhole_gas.py deleted). Measured, Sun at 1e6 M_sun on the
        eccentric orbit: ~3% of the star eaten per 1000 M, with or without
        cooling; a disc forms with cooling, a hot cloud without.
      - N-body: bodies that collide hard shatter (Leinhardt & Stewart 2012
        scaling, 8 fragments); a hole's tide tears a body into 40 pieces at
        r_t, or swallows it whole if r_t is inside the capture radius;
        swallowed bodies fade and redden at the hole (stylised, 1.5 s).
        Debris pieces do not interact with each other (nbodystate.DEBRIS).
        Each step is capped at 12 ms of wall time (slow motion beyond).
        Two bodies at the same point no longer crash the 1PN code.
- [x] 2026-11-22, later (Claude under override; review the physics):
      - Gravitational waves: 2.5PN radiation reaction in the relativity
        force (postnewtonian.rr_acc; Peters' da/dt reproduced to 0.03%);
        merging holes lose 4.8% (4 nu)^2 of their mass. Scene "Black hole
        merger": 2 x 5e9 M_sun from 20 G M / c^2, merges after 18.8 yr and
        28 orbits (Peters: 19.5 yr). PN is only indicative for the last
        orbits (v/c 0.2 to 0.4).
      - Spacetime sheet (nbody_sheet.py; setting Auto/On/Off, wave-height
        gain): dips at the masses are an illustration; the ripples are the
        quadrupole formula's h_+ at the retarded time, drawn as r h.
      - Scenes "Planets colliding" (head-on at 30 km/s after 2 yr) and
        "Star torn by a black hole" (supergiant, 1e4 M_sun hole).
      - Looks: body glow, black holes as a dark disc with a bright ring,
        collision and merger flashes, molten fragments, black-hole-module
        gas coloured by how much it was heated (display only).
- [x] 2026-11-22, star scenes reworked (the star should look like a star,
      stretch into a strand, wind round the hole and be eaten):
      - Black-hole module: default hole 10^5.5 M_sun (Sun radius 1.5 M);
        "Star torn apart" is a bound orbit with L = 3.8 and uses the gas;
        81% swallowed (measured), the rest stays in orbit. Star particles
        are drawn as soft blobs; the star comes in from the right, clear of
        the settings panel.
      - N-body: tidal debris is 300 pieces of "gas" (soft blobs, no trails),
        forces linear in the pieces (nbody_debris.py; test-body 1PN on the
        pieces). Scenes can run at other than a year per second
        (Scene.years_per_second). "Star torn by a black hole" is now the Sun
        at a 10^5.5 M_sun hole at true sizes; 79% swallowed (measured).
- [x] 2026-11-22, "Gas ring" scene (black-hole module): 2000 SPH particles
      on circular orbits at 12 M, locally isothermal (H/r = 0.03), shear
      viscosity on (no Balsara factor). Measured at viscosity 1: 4.3% eaten
      by 1000 M, 26% by 6000 M, nothing thrown out (outermost 61 M); the
      "Gas viscosity" slider changes the rate (2.4% / 7.3% / 13% by 1500 M
      at 0.2 / 1 / 5). With "Gas cooling: None" it heats and swells instead.
- [x] 2026-11-22, N-body: captured bodies no longer vanish at the capture
      radius with a dot in the hole's centre. They are drawn falling on to
      the horizon (nbody_plunge.py: Schwarzschild coordinate-time infall,
      redshift dimming); display only, slowed to be visible when a step is
      much longer than r_s / c.
- [x] 2026-11-22, N-body, bodies near holes (replaces the plunge display):
      - A light body (<= 1% of the hole) within 20 G M / c^2 is handed to
        the exact Schwarzschild geodesic (nbody_plunge.py) until it is
        swallowed or leaves; so capture follows the exact rule (tested:
        L = 3.9 G M / c swallowed, 4.1 escapes). Heavier bodies and hole
        pairs still merge at 6 G M / c^2.
      - Automatic slow motion near holes (SLOW_MOTION: a step advances at
        most 0.1 of the shortest hole-pair timescale); module.elapsed is the
        simulated time.
      - "Star torn by a black hole" starts as one Sun outside its tidal
        radius; 63% swallowed (measured). New-body kind "Star"; rings for
        tidal and capture radius while placing or selecting; holes drawn
        on top; thin gas drawn wider and fainter.
- [x] 2026-11-22, N-body Custom mode near holes:
      - Fixed "frozen body": slow motion locked on to one fragment a
        millionth of an AU from a small hole (factor ~1e9). Slow motion is
        now bounded by what is visible (VISIBLE_PIXELS, handover radius,
        MAX_SLOW_MOTION), and bodies on screen near a hole move at most
        PIXELS_PER_STEP pixels per step.
      - Debris bound to a hole with a dynamical time below half a step is
        counted as accreted (_accrete_unresolved); whole bodies never are.
      - Swallowed bodies are shown frozen at the hole's drawn edge,
        reddening and fading over 1.6 s (stylised: FADE_SECONDS).
      - Mass presets for new black holes; tear radius given as text when
        its ring is too small to draw.
- [ ] Open: the star scenes' swallowed fractions with spin are unmeasured;
      collision fragments use a recipe for speeds (not from the paper);
      grazing impacts are treated as head-on.
- [x] Kerr (spin slider): ZAMO-camera ray tracer (Mino time), shadow matches
      Bardeen's curve; isco and Page-Thorne flux (now also used at a = 0);
      bodies on Kerr geodesics (Lense-Thirring checked) with exact Kerr
      images for up to 50 bodies (Newton shooting); more (the star) and trail
      points use Schwarzschild lensing as an approximation.
- [x] Tests: tests/test_blackhole_physics.py, test_blackhole_gpu.py,
      test_postnewtonian.py (207 tests in all pass).
- [ ] Possible: Kerr images of many particles (GPU solver), a rotating disc
      texture, binary black holes (needs 2.5PN radiation reaction).
