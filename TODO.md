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
- **Module 4, particle physics:** first version working (Claude under
  override, 2026-11-22; review the physics). Event display in a CMS-like
  detector with invariant-mass histogram. See "Module 4" below.

- **Removed 2026-11-26** (user's request): the particle module's Custom event
  type (`particle_custom.py`, `particle_editor.py`, including the pair -> Z
  collisions and particle save/load) and the atom module's "Decay of a sample"
  and "Decay chain" scenes. Mentions of them further down are history. Now
  unused by the app but still tested: the B, D and Ds physics,
  `breit_wigner_formation`, and `build_chain` / `DecaySample` / `bateman`.

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

## Home screen, save and load (2026-11-23)

- [x] Home screen (homemodule.py), first in the registry: a description of
      each module with an Open button, and each module's saved scenarios.
      Needed one addition to the core, approved: BaseModule.switch_requested
      (a registry name), honoured by the app loop after draw_ui.
- [x] Save / load of custom scenarios (scenario_files.py; JSON files in
      saves/<kind>/ at the top of the repository): N-body (whatever is on
      screen, loaded as the Custom scene), black hole (scene, spin, hole
      mass, star and the bodies added in the Bodies window), galaxies (the
      setup window's models, orbit and angles). tests/test_scenarios.py
      (6 round-trip tests).
- N-body Restart after an edit now keeps the scene's pace, camera tilt and
  sheet setting (the restart point used to drop them).
- [ ] `saves/` and `data/` are not in .gitignore: decide before committing.

## Module 4, particle physics

Files: particle_data.py (PDG values), particle_kinematics.py (four-momenta,
boosts, two-body and n-body phase-space decays, leptonic V-A decays,
Breit-Wigner), particle_material.py (Bethe energy loss, Highland
scattering, bremsstrahlung, conversions, shower profiles, the detector's
materials), particle_tracks.py (helix in the solenoid and return field,
stepping through the material), particle_events.py (scenes, Higgs,
reconstruction), particle_renderer.py, particlemodule.py,
particle_opendata.py, tests/test_particles.py (29 tests; 2 need the
downloaded data and are skipped without it).

- [x] Scenes: Z -> mu mu, Z -> e e, Z -> tau tau, J/psi -> mu mu, strange
      particles (K0_S, Lambda), busy collision, Higgs -> Z Z* -> 4 leptons.
- [x] Checked against published values: p* of K0_S -> pi pi (206 MeV),
      Lambda -> p pi (101 MeV), Z -> mu mu (45.594 GeV); helix radius 3.336 m
      for 1 GeV in 1 T; mean decay length beta gamma c tau; Michel spectrum
      (mean x = 0.7); minimum ionisation in PbWO4 (10.2 MeV/cm); Highland
      angle; shower maximum; four-momentum in every decay and energy over
      whole events (ionisation included); Z peak 91.19, K0_S 0.4976, Higgs
      125.25 GeV; Z -> e e recovered through the radiated photons; muons
      need ~4.5 GeV to cross the iron.
- [x] Display: labels on particles and detector layers, legend, commentary
      of each event in words, continuous playback (next event prepared in a
      background thread), playback ends when 97% of the energy has settled.
- [x] 2026-11-22: click a particle (or the collision point, for a boson)
      for its properties and what happened to it in this event; a list of
      the event's particles to click; incoming protons and a flash before
      each event; a close-up diagram of the production mechanism (an
      illustration); real CMS events (particle_opendata.py: CERN Open Data
      record 545, CC0; files go to data/opendata/, 15 MB, ignored by git):
      Z, J/psi and all muon pairs, the last with a logarithmic spectrum
      showing J/psi, psi', Upsilon and Z.
- Still not simulated: the collision itself (momenta and particle mix are
  drawn from distributions matched to measured averages; no jets, no
  correlations: that is Pythia's job, out of scope by CLAUDE.md). Showers
  are parametrised profiles, not particle cascades; energy loss is the mean
  only (no Landau fluctuations, no delta rays); hadron interactions in the
  tracker are ignored; H -> 4 leptons has no angular correlations.
- [ ] Possible next: B and D decay chains with zoomable displaced vertices,
      real CERN Open Data dimuon events through the same display, tau decays
      through the rho and a1, an "end view" camera preset.

## Black-hole speed (2026-11-25)

Claude under override (geodesics.py, kerr_bodies.py, blackholemodule.step);
review the integrator change. Measured headless and in an offscreen GL
context at 1920 x 1080: NOT YET SEEN IN THE REAL WINDOW.

- [x] Kick steps run on across frames (`advance(carry=True)`, used by the
      module for clouds of more than 20 bodies; every other caller and the
      tests use the old path). Before, each frame closed its step and paid
      for a force evaluation, and so did each history sample. Force
      evaluations over 1200 frames: "Star torn apart" 1187 -> 379 (spin 0.6:
      2376 -> 335), "Gas ring" 750 -> 484 over 500, "with gas" 1422 -> 1350
      over 1500 (a frame there is already one step long). Mean step 11 ->
      1.5 ms in "Star torn apart". Outcomes unchanged: 97.0% / 97.1% eaten,
      96.1% with spin, with gas 1.4% / 1.2% by t = 12490 M.
- [x] Display: the picture is traced at half resolution while the camera
      moves and for 0.2 s after (MOVING_SCALE, MOVING_HOLD in
      blackholemodule.py; set MOVING_SCALE = 1 to switch it off). A camera
      move re-traces every pixel: 15-16 ms at full resolution, 7-9 at half.
- Whole frame, "with gas", camera still: step 5.5 + sprites and GPU 5 ms.
- [ ] The lower limit on OUTER_STEP (options.outer_step) lets the kick step
      exceed the gas's Courant time: down to 1% of the step at pericentre in
      "Star torn apart" before this change (4% after), 11-14% "with gas".
      Not changed; honouring it would cost several evaluations a frame there.
- [ ] Between kicks w is half a kick behind x (display and history read it).
- N-body "Star torn by a black hole": step 0.4 ms, render 4-7 ms offscreen;
  no frame-time problem found there. If it looks slow, it is the scene's
  pace (slow motion), not the frame rate.

## Verification and fixes (2026-11-23)

An independent review of modules 1-4, the home screen and save/load found
the problems below; four agents fixed them (physics included, under the
user's override). Headless runs and offscreen renders only: NOBODY HAS
CLICKED THROUGH THE REAL WINDOW since.

Black hole (blackhole_scenes.py, blackhole_star.py, blackholemodule.py):
- [x] Star scenes are laid out in units of the tidal radius r_t: the star
      starts at max(1.6 r_t, 40 M), is one object until r_t (release within
      0.3% of r_t for 10^4..10^8 Msun), and is swallowed whole from 10^8.2.
      Before, it started at 40 M, inside r_t below 10^6.2, and was released
      on frame 1. Default mass now: 97% eaten (was 78-81%).
- [x] Scene clock (r_t reached after 4 s at Speed 40, slows near the hole),
      camera follows the star in (setting, lets go when the user zooms),
      star drawn at least 6 px (factor shown in the UI), painted disc hidden
      in star scenes, dotted r_t circle.
- [x] Released gas star is supported by pressure only (half-mass radius
      0.73 -> 0.68 over 6 dynamical times; was 0.73 -> 1.61).
- [x] Load is atomic; malformed saves fall back with a notice.
- Limits: 10^7..10^8 the tear is within 10 M and over in under a second;
  "with gas" still over the 20 ms frame budget; the SPH temperature readout
  gives odd values (0 K, 1e10 K), not looked at.

N-body (nbodymodule.py, nbodystate.py, nbody_plunge.py, nbody_editor.py):
- [x] Slow motion is for visible events only and ends (hole_pace replaces
      hole_timescale; bound debris that cannot be resolved is accreted,
      Rees 1988). The two stalls (star dropped at 0.1 AU from a 10 Msun
      hole; body on a 0.005 AU orbit) are back at normal pace within 30
      steps. A planet on a stable orbit near a hole now runs at true pace.
- [x] One whole-or-torn rule: torn inside r_t unless r_t is inside the
      horizon, also inside the handover sphere.
- [x] Every swallowing path fades, continuously (within ~2%).
- [x] New holes default to 1e6 Msun, are drawn at horizon size (4 px
      minimum), with placement text saying torn or swallowed.
- [x] Torn-star scene: star cross-fades into its gas, blobs stretched along
      the stream.
- [x] Stable body ids (identity, selection, follow kept across close
      passes; unique names); saves include close-pass bodies; malformed
      saves fall back.
- [ ] 10 Msun preset: the tear is 1-2 px at the default zoom (announced,
      visible after Follow).
- [ ] A compact body or second hole on a sub-pixel orbit still exhausts the
      step budget (15 ms/step).
- [ ] Trails vanish at handover and at the tear (needs nbody_trails.py).

Particles:
- [x] Tau visible mass no longer double-counts converted photons; bad data
      rows and worker errors no longer hang; bulk fill in the background.
- [x] Every listed decay is drawn at its PDG fraction (rest in a labelled
      stand-in mode); tau -> pi pi0 nu via rho, 3 pi nu via a1.
- [x] Highland summed over steps 8.76 mrad (was 7.4); minimum ionisation
      within 0.07% of PDG (Sternheimer density effect).
- [x] Momentum resolution from Gluckstern + multiple scattering with the
      real field (35 um points, 13 layers: chosen, not from a reference);
      momenta are still smeared truth, not fitted from hits. Tracks with
      fewer than 3 layers give no mass (11% of Z -> mu mu).
- [x] Production step labelled "Illustration (not simulated)"; disclaimer
      always visible; recorded events carry no mechanism.
- [ ] CHECK AGAINST THE PDG: the numbers added here were written from
      memory (rarer tau fractions, K modes, rho and a1 widths, Sternheimer
      parameters).

Save/load, home, galaxy:
- [x] Files carry kind and format version; wrong-kind or broken files give
      a message; names are exact, overwrite asks; galaxy Save stores the
      running setup; home text within the accuracy scope; saves/ and data/
      in .gitignore.
- [ ] Galaxy saves accept absurdly small particle counts (bounds are a
      physics choice).

Looks, 2026-11-28 (Claude; display and UI only):
- [x] Spacetime sheet: dip depth goes as mass^(1/4) (was the square root),
      lighter bodies get narrower dips; dip and wave heights follow the
      camera distance smoothly instead of the grid's power-of-two half-width
      (no jump on zooming).
- [x] Recording window: label column with radio rows, red Record button,
      Discard apart from Stop and save, fixed window size in every state.
- [ ] Sheet: the near-source fade of the waves (`soft` in the vertex shader)
      still uses the snapped half-width; left alone because it sits in the
      wave formula. Earth-mass dips beside a star are still only ~3 px.
