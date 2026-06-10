# MadGraph Generation

`run.sh` generates `pp -> t t~` events with `t -> b j j` (hadronic) and
`t~ -> b~ l- vl~` (leptonic) using MadGraph5_aMC@NLO v3.5.5.

## Usage

```
./run.sh <dataset_tag> <polarization> <generation> <num_runs> <num_events_per_run> <max_cpu_cores> <seed> <top_pT_cut>
```

| Arg | Description |
|---|---|
| `dataset_tag` | Output directory suffix, e.g. `unpolarized_10k` -> `pp_tt_semi_full_unpolarized_10k` |
| `polarization` | `L`, `R`, or `U` (see below) |
| `generation` | `inclusive`, `first`, or `second` (see below) |
| `num_runs` | Number of independent `multi_run` jobs |
| `num_events_per_run` | Unweighted events per run |
| `max_cpu_cores` | Passed through to `multi_run` |
| `seed` | Initial RNG seed (`iseed`) |
| `top_pT_cut` | Minimum hadronic-top pT in GeV (see below) |

## Polarization

Edits the `p p > t t~` part of the `generate` line in `proc_card_mg5.dat`:

| Value | Resulting subprocess | Meaning |
|---|---|---|
| `U` | `p p > t t~` | Unpolarized (no spin correlation imposed) |
| `L` | `p p > t{L} t~` | Top quark forced to negative helicity (left-handed) |
| `R` | `p p > t{R} t~` | Top quark forced to positive helicity (right-handed) |

The `t~` polarization is left unconstrained in all cases.

## Generation (hadronic top decay channel)

Edits the `t > b j j` part of the `generate` line:

| Value | Resulting decay | Meaning |
|---|---|---|
| `inclusive` | `t > b j j` | `j` = any light quark/gluon (`g u c d s u~ c~ d~ s~`) |
| `first` | `t > b u d~` | First-generation decay only (`W+ -> u d~`) |
| `second` | `t > b c s~` | Second-generation decay only (`W+ -> c s~`) |

The leptonic side, `t~ > b~ l- vl~`, is unaffected by this argument.

## Config files (`config/`)

| File | Purpose |
|---|---|
| `proc_card_mg5.dat` | Base MG5 command file (process definition, model setup) |
| `multi_run.config` | Template for `multi_run`/`nevents`/`iseed`, edited per-run |
| `run_card.dat` | Reference run card (beam energy, PDF set, scale choice, generic cuts). Not auto-copied into the generated process by `run.sh` |
| `pt_cut.f` | Custom generator-level cut, spliced into the generated `cuts.f` (see below) |

## Custom pT(top) cut

MadGraph auto-generates a large `cuts.f` (~1750 lines) into every
`SubProcesses/` directory. Rather than committing and copying a full
custom version of that file just to add one extra cut, `run.sh` uses
`sed` to splice the small snippet `config/pt_cut.f` into the
freshly-generated stock `cuts.f`, anchored on a unique MadWeight tag
comment that sits in a safe spot within `PASSCUTS` (after the basic
momentum sanity checks, before the run_card-driven cuts).

`pt_cut.f` reconstructs the hadronic top's 4-momentum from its decay
products (`b j j`) and rejects the event if its transverse momentum is
below `top_pT_cut` GeV. The `__TOP_PT_CUT__` placeholder in `pt_cut.f`
is filled in by `run.sh` via `sed` before splicing the snippet into
`cuts.f`.
