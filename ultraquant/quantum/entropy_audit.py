"""The entropy black box: judge any randomness source by its output alone.

The chaos experiment ended with a revert and one durable lesson: the first
jitter harvester produced a constant, and hashing that constant made it look
statistically perfect. What survives is the *auditor's* stance — never trust a
source's story, only its bytes — and this module is that stance made
mechanical. Feed it bytes from anything (timing jitter, ``os.urandom``, QPU
measurement counts, a PRNG, a suspected-broken harvester) and it scores them
with a battery of tests that make no assumptions about where they came from.
A black box in the proper sense: internals invisible, verdict earned at the
boundary.

The battery is chosen so that each known failure mode has a test that catches
it, including the one that actually happened here:

============== ============================================================
bit balance     ones-rate; catches stuck-at sources
runs rate       transition frequency; catches 0101... oscillators
serial 1-1      adjacent-bit correlation; catches momentum
byte entropy    Shannon bits/byte over the byte histogram
delta entropy   Shannon of successive byte *differences*; catches counters,
                which sail through every static test (all bytes distinct,
                balance f