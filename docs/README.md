# scopex documentation

**[blueprint_scopex.html](blueprint_scopex.html) is the documentation.** Open it first. Everything
here is an appendix to it, and the blueprint links each one from the section it belongs to.

| file | audience | what it is |
|---|---|---|
| **[blueprint_scopex.html](blueprint_scopex.html)** | you are profiling a jitted JAX program | the two routes, the naming contract, the silent failures, the API, the recipe routing table, reading a dump, and the limits |
| [DEFICITS.md](DEFICITS.md) | you are deciding whether to trust a number scopex just produced | per instrument: what validates it, where that validation fails, what it cannot see, what it costs |
| [HARDENING.md](HARDENING.md) | you are changing a parser, or you just upgraded jax | every place scopex reads text a compiler printed, what produces it, and how the self-checks cover it |
| [INVESTIGATIONS.md](INVESTIGATIONS.md) | you want the evidence behind a claim | 28 arms worked end to end: what each instrument did and did not show, and the routes tried and rejected |

Two more live next to the code they describe rather than here, because that is where you need them:

| file | audience |
|---|---|
| [../README.md](../README.md) | you just arrived at the repository |
| [../tests/degenerate/SUITE.md](../tests/degenerate/SUITE.md) | you are adding a corpus case, or running the suite |

None of these is scratch. Everything under `docs/` is maintained, every number in it was measured on
jax 0.10.2, and where a re-run later disagreed the file says so and keeps both.

## Runnable, not prose

- `examples/marked_framework.py` — the program every number in the blueprint comes from
- `examples/recipes/` — one runnable file per question, routed by blueprint §7
- `scopex.selftest()` / `scopex.conformance()` — run after any jax upgrade; they fail loudly
