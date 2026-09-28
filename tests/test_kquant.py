"""K-quant weights, decoded exactly as llama.cpp decodes them.

11.123: the reader refused Q4_K, Q5_K and Q6_K by name, and the user's
Command-R 08-2024 is almost nothing else. These pins hold the decoder to
ggml bit for bit - everywhere, through golden blocks captured from ggml
itself; and on a machine that has the model and LM Studio, through the
whole gate.
"""

from __future__ import annotations

import io
import struct
import unittest

from ultraquant.convert import gguf

#: One real block of each k-quant type from Command-R 08-2024 (Q4_K_S),
#: with ggml's own float32 decode of it - captured from ggml-base.dll.
GOLDEN = {
    'Q4_K': ('blk.0.attn_q.weight',
        ('e503300fffb8b3efb9ffbbff11089e3089a6c782b880787bc69f86b298c8a78a'
         '76a5a76578a6064775647795f71487a7a89e78c8667abf8d6a05aad9beabd4c9'
         'aabd67c08afd7b7bdaf9a978777a895da1c5db64b54539730483e43ee4a89476'
         'f09773558182f9e5b68645d560a56166667a7184777776d8c754988a0863c5a5'
         '76825fa427a3c85798f3c0c457787775'),
        ('c05c0e3c009e26bb007a9d3a40818fbc000ca23b00d8ccbc000ca23b2085843c'
         '009e26bba032ff3c009e26bb40818fbc000ca23b000ca23b007a9d3a80b34b3c'
         '009e26bb80fccdbb007a9d3a80fccdbb000ca23b009e26bb009e26bb007a9d3a'
         '80fccdbb005524bc007a9d3a80fccdbb007a9d3a005524bc007a9d3a007a9d3a'
         '000085ba00d8b83b0078493c000085ba00f2123c000085ba004c8ebb004c8ebb'
         '0078493c0098173b000085ba00f2123c0098173b0078493c00d8b83b000085ba'
         '004c8ebb00d8b83b00d8b83b0058fbbb004c8ebb00d8b83b0068e2bc00b86abc'
         '004c8ebb0058fbbb004c8ebb0098173b0085b63c0025c7bc000085ba00d8b83b'
         '0090d6ba408e873c0090d6ba0090d6ba0043fcbb00fb903b2062a03cc0745d3c'
         '00fb903b40c92fbc00fb903b00aeb63a408e873c804af43b007161bc00aeb63a'
         '00fb903bc0745d3c80f398bb0008d4bc00fb903bc0745d3c804af43b804af43b'
         '00fb903b00aeb63a00aeb63a0090d6ba80f398bb00fb903b00aeb63ac0745d3c'
         '0070993900dd23bbc07c04bc00a2c03b803f32bcc07c04bc00394a3b0074adbb'
         '803f32bc0068e2bc00709939c0130e3c00394a3b00709939c0130e3c00a2c03b'
         '0070993900394a3b803f32bc00a2c03b0074adbb4099693cc07c04bcc07c04bc'
         'c0130e3c4099693c00709939c07c04bcc07c04bcc07c04bc0074adbb400260bc'
         'c07a3dbc0020ad38e0cc8f3c006c39bb0020ad380020ad38402f403c8020bcbb'
         '006c39bb8020bcbb006c39bbc05cd73c006c39bb007a103c006c39bb003e443b'
         '00306dbc8089c13b8020bcbb0020ad38c07a3dbc80c50dbc402f403c0020ad38'
         '003e443b003e443b0020ad380020ad3800306dbc0020ad38c07a3dbc003e443b'
         '00d0313b00b0f43b004a213c00a8debb00cca63b00383dbc002a64bc00c490bb'
         '0080acbc00c005bb003c483c002a64bc003c483c00d0313b0040b03900c490bb'
         '002e6f3c0040b03900c490bb004616bc00c005bb00c005bb002e6f3c003c483c'
         '00cca63b00c005bb00383dbc004a213c00a8debb00d0313b00a8debb00a8debb'
         '00b8cfba002f193c80e679bc0014e7bb0094963a0094963a00b8cfba00707e3b'
         '0094963a0014e7bb00707e3b002f193c00707e3b805320bc00818dbb00818dbb'
         '00b8cfba001d4dbc408fbc3c0014e7bb0094963a805320bc00707e3b0094963a'
         '00707e3b805320bc005893bc0014e7bb0094963a00707e3b0094963a00818dbb'
         '0050acbb00b01dbb00b01dbb0000ea3900b01dbb00b01dbb00b01dbb00fc703c'
         '0040423c00e404bc0030583b0000ea390048b7bc0050acbb0040423c0090c93b'
         '00b01dbb0000ea3900e404bc0090c93b008c88bc0090c93b0040423c00e404bc'
         '0030583b003aa73c0040423c0040423c00e404bc00b01dbb00b01dbb00b01dbb'),
    ),
    'Q5_K': ('blk.0.attn_v.weight',
        ('8a01820ebbeef2f2bfe1e2f68720cd2f3931a37bcfa9ed7347e9e5e4e3512268'
         '293d3a8321f7c071562909217b29401f68f652a1f7b8ff151b9191ea2a244eeb'
         '72f407361214cdd4c0a980e7150faf022dfcde5de36c21cb60e5e1af0fcefdbd'
         'fe05d0f6c905eeeac11522ccedc6ee48d7630f5507978225a4908daabfea1f15'
         '8a7e62f808ef0a7693a8c39f196a17d3dfdf8657c22f7351905023543065d085'
         'c83eca6bcce25e01fbdcfaf9c4ead889'),
        ('00d6063c00deb23b0080afb80094c0ba0045e03b00d6063c40df923c0077853b'
         '80f04a3c0094c0ba0094c0ba00fb36bc003d343c0020303b005ab8bb804720bc'
         '0080afb80020303b80157bbc00deb23b0080afb80020303b00c1e5bb0020303b'
         '00ffccbc80891d3c00183bbb0045e03b0077853b40df923c00f38abb0080afb8'
         '0020d9bb00ec4a3b00ed1c3c006c753c80f5a63c0088a0ba00ec4a3b0042ac3b'
         '0042ac3b00dc5dbb00dc5dbb0020043b00a8cf3b005c33bc003a0b3c0020043b'
         '00bab5bb00ec4a3b00dc883b000ef33b000f45bc0042ac3b008017b90050753a'
         '00698c3c001017bb005492bb0020043b0042ac3b00c256bc001017bb00dc883b'
         '00a8e63a0080193a0048403b00a8e63a0043103c0080193a0092d33b00501aba'
         '0018ad3b0026fabb0092d33b808fbb3c009e863b0048403b00a8e63a00a8e63a'
         '0048403b00bd363c00445dbc00acd3bb007c40bb00bd363c0048403b0010e7ba'
         '0092d33b0026fabb00ca36bc0080193a00a8e63a00acd3bb0048403b0074703c'
         '0040a4b900647bbb00a6cabb0054523b00ba613c00a48f3b0040a4b90020f1bb'
         '00fe6bbc00ba613c002ca4bb000a1fbc00b6afbc0020f1bb00647bbb0003283c'
         '00f7743c00702ebb007d4e3c00647bbb0020f1bb00b6afbc002ca4bb002ca4bb'
         '0020f1bb00f8c2ba0040a4b90020f1bb00ba613c00403b3c002ca4bb0060053b'
         '00a5a93b0064c63a00e60cbb003e5b3b80451bbc80451bbc004d66bc003e5b3b'
         '004a48bc002882bc007982bb0082dcbb00e60cbb00d7013c00e60cbb804839bc'
         '0082dcbb00dd3d3c00b01c3a00440cbc00440cbc80de4c3c0082dcbb00a28b3b'
         '0064c63a00440cbc804b57bc00e60cbb00abe53b0082dcbb80451bbc0064c63a'
         '0034523c00d8a13b0020efba00d0793b00b2a2bc0054083c00b8eb3b0080e139'
         '00b40cbc0054083c00b8eb3b00cc1a3c00442d3c00a885bb00c036ba00c036ba'
         '00b8eb3b00c8c63b00d8a13b007041bb0020efba00ac643c00b2a2bc00c8c63b'
         '002c1fbc00cc1a3c0088cfbb0054083c00c036ba00d8a13b007699bc0098aabb'
         '00a219bb00a219bb00a579bc805b153c0004003b00a219bb00e65d3b0088083a'
         '00006fba00006fba00e65d3b00e49d3b00006fba00d5cc3b003cc3bc00d5cc3b'
         '00b44abc008477bb00c31bbc804a04bc00a4d9bb0004003b80d39c3c0088083a'
         '803d733c00a4d9bb00c31bbc803b33bc00e49d3b00c31bbc00d42c3c803b33bc'
         '004824ba004824ba0030803c80304cbc00abb03c00dedd3b8022683c80304cbc'
         '0075d6bb80a7373c00dedd3b80a7373c802c073c00f333bc004824ba007803bc'
         '00080abb80ab7cbc00080abb00e54f3c00080abb80e8c83c80a7373c00b2a2bc'
         '00da183b004824ba00da183b00da183b00080abb00905f3a004824ba007803bc'),
    ),
    'Q6_K': ('token_embd.weight',
        ('e0ea2c115fe9f0bf0d091181d74df00649f16cf85700c0daf359d5d1bccec212'
         'e10e748f18219e5170de201de3c5026f033821465c0bb124ec1d08f5f308171b'
         'fc2d028f7e10303693dd5158d0096bd40f85ed88eeaddfe38694af70b1d314e1'
         'd1c4b70c76518532b2fea00f9d7c5f4fb19fef4e7f8ed30190f02defa21ef0d1'
         '4b91be7a22f8929a8c75a2a668a494baa988bdaeac855254469db17a5929bab6'
         '59596da5b5a0baaa6b55ba15568595944966b6b49565591a4b6ba4d41be55ab3'
         '0ceaeaf0e8eac49f21f143801416d0151601'),
        ('0080503b00609cba00601c3b008050390078433b00dc95bb000000000078433b'
         '009877bb0070b6ba008050390080503900e4a2bb009877bb0080d0bb00609c3a'
         '003c273b0027393c0020bf3a00203fbb0051153c0020bf3b000000800066033c'
         '00588fba003c273b0066833b0020bfb90020bf3a00203f3a00203fba00203fba'
         '0020bfb90004d73b00e8eebb002eb3bb00580f3c0020bfb90004d73b0020bfb9'
         '0020bfbb00203f3a00203f3c00588f3a00588fba0066833b003ca73b002eb3bb'
         '008050ba00000bbb00b093bb0020bfbb0040f3bb00c0ad3a00a8063c0080503b'
         '00008b3a00f0fbbb0080d03b00c0adba008050ba00000bbb0040f3ba00c0ad3a'
         '0090ea3b0080503a0090eabb0088ddbb005002bb007843bc0080d0390050023b'
         '0080503c0080d0bb0080d0b9008050bb0068a9bb0080d0ba0080d0390080d0bb'
         '0020bfba0012cb3b006603bc002eb3bb00e8eeba00203f3c0020bf3a00588f3a'
         '0012cb3b0066833b00352dbc00352dbc00e8ee3a00588fbb003c27bc0012cbbb'
         '0050023b00000080000ce4bb0050023c007bfc3c009a92bc009a12bc00e4a2bb'
         '000ce4bb0078433b005002bb005082ba0050023b007843bc000000800078c3bb'
         '0000008000019ebb00ac52bb00acd2bb80ab03bc0000008080ab033c8056b83c'
         '00ac523b00acd2ba0000008000acd23a00acd23a00ac523d00acd2ba00acd2ba'
         '00580fbb0004d7ba00dafabb00580fba00588fba00588fbc000000000004573b'
         '80382a3c0004d7ba00580f3a00588fbb0000000000da7abb002e33bb00da7abc'
         '0050823900e4a2ba00c253bb0078c33b0050023a0078433a00508239007843ba'
         '002eb3bb00e4a2bb00758a3b0050023c00758abb00c2533b005082ba00758abb'
         '0084913a0084913bc02dd13c008491bb00e535bce0f70cbd00e5b53b0084113b'
         '0084113b008411bb00000000008491ba00465abb008491bb008491ba008491ba'
         '00000bbb00000b3b00000b3b00008b3b00000b3b00008b3b0080d0bb00000bbb'
         '00000080000000800080d03b00000b3b00008bbb00008b3b0000008000a8863d'
         '00c0adb9000898bb0000000000c02d3b00c4f93b00c0ad390054ce3b0050823a'
         '0078433b005082ba000ce43b00e86ebb005082ba00c02dbc003059bb005082ba'
         '00203fbc00203f3b002e333c00580f3c00203fba00e86e3b00588fba00203fba'
         '00580fbc0004573b00e86e3b000457bb00e8eeba004a9b3b002eb3bb002e333c'
         '00601c3b0080503b0050823b000000800070b6bb005082bb0080d0bb00601cbb'
         '0050823b0080503a005002bc0080d03c0070b63b0070b6bb005082bb008050bb'
         '000ce4ba00a21fbb00a29f3b0070b63a00a21f3b007036bb00d488ba007036bc'
         '00a21fbb0070b6b90070363a00092b3c00dafabb00d7c13b0070b6b9003b943b'),
    ),
}


def _floats(hexes: tuple) -> list:
    raw = bytes.fromhex("".join(hexes))
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class GoldenBlockTests(unittest.TestCase):
    """One real block of each type, against ggml's own decode of it."""

    def test_each_type_decodes_bit_exactly(self) -> None:
        for kind, (name, raw_hex, out_hex) in GOLDEN.items():
            with self.subTest(kind=kind, tensor=name):
                raw = bytes.fromhex("".join(raw_hex))
                ours = gguf._decode_rows(io.BytesIO(raw), 0, kind, 256, 0, 1)[0]
                theirs = _floats(out_hex)
                self.assertEqual(len(ours), 256)
                self.assertEqual(
                    struct.pack("<256f", *ours), struct.pack("<256f", *theirs))
                self.assertEqual(ours, theirs)      # exact values, not just bits

    def test_values_are_exact_float32(self) -> None:
        """A float64 that merely rounds to the right float32 is not enough."""
        for kind, (_name, raw_hex, _out) in GOLDEN.items():
            raw = bytes.fromhex("".join(raw_hex))
            for value in gguf._decode_rows(io.BytesIO(raw), 0, kind, 256, 0, 1)[0]:
                self.assertEqual(value,
                                 struct.unpack("<f", struct.pack("<f", value))[0])

    def test_the_three_types_are_readable_and_the_rest_refused(self) -> None:
        for kind in ("Q4_K", "Q5_K", "Q6_K"):
            self.assertIn(kind, gguf._READABLE)
        for kind in ("Q2_K", "Q3_K", "IQ4_XS"):
            self.assertNotIn(kind, gguf._READABLE)


class WholeGateTests(unittest.TestCase):
    """The full exam, where the model and ggml exist."""

    def setUp(self) -> None:
        from ultraquant.experiments import kquant_gate
        self.gate = kquant_gate
        self.doc = " ".join(kquant_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Bit-exact", "The exam can fail",
                       "Rows are blocks, joined", "Nothing else moves",
                       "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("PASSED on all four measured criteria", self.doc)
        self.assertIn("428", self.doc)

    def test_the_gate_passes_here(self) -> None:
        if not (self.gate.COMMAND_R.exists() and self.gate.GGML_DIR.exists()):
            self.skipTest("Command-R or LM Studio's ggml is not on this machine")
        self.assertTrue(self.gate.run_gate().passes)


if __name__ == "__main__":
    unittest.main()
