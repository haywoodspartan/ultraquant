"""LM Studio and the LLMLS teacher panel.

These tests never contact a real server. Every request is answered by a stub, so
the suite passes on a machine with no LM Studio installed — which is the normal
case for an optional tier, and the same discipline
``tests/test_bluequbit.py`` applies to the cloud backend.

The lineage cases are drawn from a real catalogue rather than invented, because
the traps that matter are the ones real model naming actually produces:

    qwen/qwen3-coder-30b      arch=qwen3moe    publisher=qwen
    qwen/qwen3-coder-next     arch=qwen3next   publisher=qwen      <- arch differs
    openai/gpt-oss-20b        arch=gpt-oss     publisher=openai
    openai-gpt-oss-...-neo    arch=gpt-oss     publisher=DavidAU   <- publisher differs

Neither field alone separates a lineage from an unrelated model, which is the
whole reason :func:`lineage_key` uses three signals and over-groups.
"""

from __future__ import annotations

import json
import io
import unittest
import urllib.error
from unittest import mock

from ultraquant.interpreter.lmstudio import (
    Answer,
    LMStudioClient,
    LMStudioUnavailable,
    quarantine_answer,
)
from ultraquant.interpreter.llmls import (
    Consensus,
    ModelCard,
    TeacherPanel,
    _correlated,
    independent_groups,
    lineage_key,
)


#: A real catalogue shape, trimmed to the interesting rows.
_CATALOGUE = {
    "data": [
        {"id": "qwen/qwen3-coder-30b", "arch": "qwen3moe", "publisher": "qwen",
         "type": "llm", "state": "loaded", "max_context_length": 262144},
        {"id": "qwen/qwen3-coder-next", "arch": "qwen3next", "publisher": "qwen",
         "type": "llm", "state": "not-loaded", "max_context_length": 262144},
        {"id": "openai/gpt-oss-20b", "arch": "gpt-oss", "publisher": "openai",
         "type": "llm", "state": "not-loaded", "max_context_length": 131072},
        {"id": "openai-gpt-oss-20b-abliterated-neo", "arch": "gpt-oss",
         "publisher": "DavidAU", "type": "llm", "state": "not-loaded",
         "max_context_length": 131072},
        {"id": "google/gemma-4-31b", "arch": "gemma4", "publisher": "google",
         "type": "vlm", "state": "not-loaded", "max_context_length": 131072},
        {"id": "text-embedding-nomic-embed-text-v1.5", "arch": "nomic-bert",
         "publisher": "nomic-ai", "type": "embeddings", "state": "not-loaded",
         "max_context_length": 2048},
    ]
}


def _card(model_id, arch="", publisher="", kind="llm"):
    return ModelCard(id=model_id, arch=arch, publisher=publisher, kind=kind)


class _Reply(io.BytesIO):
    """A urlopen context-manager result."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StubServer:
    """Answers HTTP the way LM Studio does, without a socket."""

    def __init__(self, replies=None, fail=None):
        self.replies = replies or {}
        self.fail = fail
        self.requests = []

    def __call__(self, request, timeout=None):
        # urlopen takes a Request or a bare URL string; both are used here.
        url = request if isinstance(request, str) else request.full_url
        data = None if isinstance(request, str) else request.data
        body = json.loads(data) if data else None
        self.requests.append((url, body))
        if self.fail:
            raise self.fail
        for fragment, payload in self.replies.items():
            if fragment in url:
                data = payload(body) if callable(payload) else payload
                return _Reply(json.dumps(data).encode())
        raise urllib.error.HTTPError(url, 404, "no such route", {}, None)


def _openai_models(ids):
    return {"data": [{"id": name} for name in ids]}


def _chat(text, model="stub-model"):
    return {"model": model,
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"total_tokens": 7}}


class ClientTests(unittest.TestCase):
    """The transport, and every failure mode named separately."""

    def test_localhost_is_the_default(self):
        self.assertIn("127.0.0.1", LMStudioClient().base_url)

    def test_a_remote_endpoint_is_refused_by_default(self):
        """Prompts and quarantined claims must not leave the machine by typo."""
        with self.assertRaises(ValueError) as caught:
            LMStudioClient("http://192.168.1.50:1234/v1")
        self.assertIn("leave this machine", str(caught.exception))

    def test_a_remote_endpoint_is_allowed_when_asked_for_explicitly(self):
        client = LMStudioClient("http://192.168.1.50:1234/v1", allow_remote=True)
        self.assertTrue(client.is_remote)

    def test_ipv6_loopback_counts_as_local(self):
        self.assertFalse(LMStudioClient("http://[::1]:1234/v1").is_remote)

    def test_models_are_listed(self):
        stub = StubServer({"/models": _openai_models(["a", "b-embed"])})
        with mock.patch("urllib.request.urlopen", stub):
            self.assertEqual(LMStudioClient().models(), ["a", "b-embed"])

    def test_a_refused_connection_says_the_server_is_not_running(self):
        stub = StubServer(fail=urllib.error.URLError("Connection refused"))
        with mock.patch("urllib.request.urlopen", stub):
            with self.assertRaises(LMStudioUnavailable) as caught:
                LMStudioClient().models()
        self.assertIn("Start Server", str(caught.exception))

    def test_available_never_raises(self):
        stub = StubServer(fail=urllib.error.URLError("nope"))
        with mock.patch("urllib.request.urlopen", stub):
            self.assertFalse(LMStudioClient().available())

    def test_a_non_json_reply_is_a_named_failure(self):
        class Garbage(StubServer):
            def __call__(self, request, timeout=None):
                return _Reply(b"<html>not json</html>")

        with mock.patch("urllib.request.urlopen", Garbage()):
            with self.assertRaises(LMStudioUnavailable) as caught:
                LMStudioClient().models()
        self.assertIn("non-JSON", str(caught.exception))

    def test_completion_carries_provenance(self):
        stub = StubServer({"/models": _openai_models(["m1"]),
                           "/chat/completions": _chat("42", model="m1")})
        with mock.patch("urllib.request.urlopen", stub):
            answer = LMStudioClient().complete("what is six times seven?")
        self.assertEqual(answer.text, "42")
        self.assertEqual(answer.source_url, "lmstudio://m1")
        self.assertEqual(answer.prompt, "what is six times seven?")

    def test_temperature_defaults_to_zero(self):
        """Sampling noise would look like two sources disagreeing."""
        stub = StubServer({"/models": _openai_models(["m1"]),
                           "/chat/completions": _chat("ok")})
        with mock.patch("urllib.request.urlopen", stub):
            LMStudioClient().complete("hello")
        body = [b for url, b in stub.requests if "chat" in url][0]
        self.assertEqual(body["temperature"], 0.0)

    def test_embeddings_are_returned_in_input_order(self):
        """The server sends an index; trusting position instead would mislabel."""
        def shuffled(body):
            rows = [{"index": i, "embedding": [float(i)]}
                    for i in range(len(body["input"]))]
            return {"data": list(reversed(rows))}

        stub = StubServer({"/models": _openai_models(["nomic-embed"]),
                           "/embeddings": shuffled})
        with mock.patch("urllib.request.urlopen", stub):
            vectors = LMStudioClient().embed(["a", "b", "c"])
        self.assertEqual(vectors, [[0.0], [1.0], [2.0]])

    def test_a_short_embedding_batch_is_an_error_not_a_silent_gap(self):
        stub = StubServer({"/models": _openai_models(["nomic-embed"]),
                           "/embeddings": {"data": [{"index": 0,
                                                     "embedding": [1.0]}]}})
        with mock.patch("urllib.request.urlopen", stub):
            with self.assertRaises(LMStudioUnavailable):
                LMStudioClient().embed(["a", "b"])

    def test_an_empty_embedding_request_costs_nothing(self):
        stub = StubServer({})
        with mock.patch("urllib.request.urlopen", stub):
            self.assertEqual(LMStudioClient().embed([]), [])
        self.assertEqual(stub.requests, [])

    def test_describe_never_leaks_prompts(self):
        stub = StubServer({"/models": _openai_models(["m1", "x-embed"])})
        with mock.patch("urllib.request.urlopen", stub):
            described = LMStudioClient().describe()
        self.assertEqual(described["embedding_models"], ["x-embed"])
        self.assertNotIn("prompt", json.dumps(described))


class LineageTests(unittest.TestCase):
    """Independence accounting - the part that decides what agreement is worth."""

    def test_same_publisher_different_arch_is_one_voice(self):
        """The Qwen pair: arch alone would call these independent."""
        a = _card("qwen/qwen3-coder-30b", "qwen3moe", "qwen")
        b = _card("qwen/qwen3-coder-next", "qwen3next", "qwen")
        self.assertNotEqual(a.arch, b.arch)
        self.assertTrue(_correlated(a, b))

    def test_same_arch_different_publisher_is_one_voice(self):
        """The gpt-oss pair: publisher alone would call these independent."""
        a = _card("openai/gpt-oss-20b", "gpt-oss", "openai")
        b = _card("openai-gpt-oss-20b-abliterated-neo", "gpt-oss", "DavidAU")
        self.assertNotEqual(a.publisher, b.publisher)
        self.assertTrue(_correlated(a, b))

    def test_genuinely_unrelated_models_stay_separate(self):
        """Over-grouping is the safe error, but it must not swallow everything."""
        a = _card("qwen/qwen3-coder-30b", "qwen3moe", "qwen")
        b = _card("google/gemma-4-31b", "gemma4", "google")
        self.assertFalse(_correlated(a, b))

    def test_correlation_is_transitive(self):
        """Base -> fine-tune -> requantization is one voice, not three."""
        base = _card("gpt-oss-20b", "gpt-oss", "openai")
        tune = _card("gpt-oss-20b-neo", "gpt-oss", "DavidAU")
        requant = _card("gpt-oss-20b-neo-q4", "gguf-x", "DavidAU")
        groups = independent_groups([base, tune, requant])
        self.assertEqual(len(groups), 1, "a lineage chain must collapse to one")

    def test_the_documented_over_grouping_is_real_and_deliberate(self):
        """DeepSeek-Coder and CodeLlama merge on arch=llama though separate.

        Asserted rather than hidden: this costs real corroboration, and the
        docstring claims it happens. If it ever stopped happening the claim
        would be stale.
        """
        a = _card("deepseek-coder-33b-instruct", "llama", "TheBloke")
        b = _card("codellama-34b", "llama", "TheBloke")
        self.assertTrue(_correlated(a, b))

    def test_an_empty_panel_has_no_voices(self):
        self.assertEqual(independent_groups([]), [])

    def test_lineage_key_is_lowercased(self):
        key = lineage_key(_card("Qwen/Qwen3-30B", "Qwen3MoE", "Qwen"))
        self.assertEqual(key[0], "qwen3moe")
        self.assertEqual(key[1], "qwen")


class PanelTests(unittest.TestCase):
    """The panel itself, against a stub catalogue."""

    def _panel(self, ids, answers=None):
        replies = {"/api/v0/models": _CATALOGUE,
                   "/v1/models": _openai_models([r["id"] for r in
                                                 _CATALOGUE["data"]])}
        if answers is not None:
            def responder(body):
                return _chat(answers[body["model"]], model=body["model"])
            replies["/chat/completions"] = responder
        stub = StubServer(replies)
        with mock.patch("urllib.request.urlopen", stub):
            panel = TeacherPanel(ids)
        panel.cli = None                # never shell out during tests
        return panel, stub

    def test_a_panel_of_two_qwens_is_one_voice(self):
        """Five models can be one source; this is the number that matters."""
        panel, _ = self._panel(["qwen/qwen3-coder-30b", "qwen/qwen3-coder-next"])
        self.assertEqual(len(panel.voices()), 1)

    def test_a_mixed_panel_counts_its_voices(self):
        panel, _ = self._panel(["qwen/qwen3-coder-30b", "openai/gpt-oss-20b",
                                "google/gemma-4-31b"])
        self.assertEqual(len(panel.voices()), 3)

    def test_an_unknown_model_is_refused_with_the_catalogue(self):
        with self.assertRaises(LMStudioUnavailable) as caught:
            self._panel(["not-a-real-model"])
        self.assertIn("Available:", str(caught.exception))

    def test_an_embedding_model_cannot_sit_on_a_panel(self):
        with self.assertRaises(LMStudioUnavailable) as caught:
            self._panel(["text-embedding-nomic-embed-text-v1.5"])
        self.assertIn("not chat models", str(caught.exception))

    def test_agreement_within_one_lineage_is_one_voice_not_two(self):
        """The central claim: a model and its sibling do not corroborate."""
        ids = ["qwen/qwen3-coder-30b", "qwen/qwen3-coder-next"]
        panel, stub = self._panel(ids, {i: "the sky is blue" for i in ids})
        with mock.patch("urllib.request.urlopen", stub):
            consensus = panel.ask("what colour is the sky?")
        self.assertTrue(consensus.agreed, "they did say the same thing")
        self.assertEqual(consensus.voices, 1)
        self.assertFalse(consensus.corroborated,
                         "one lineage agreeing with itself is not evidence")

    def test_agreement_across_lineages_does_corroborate(self):
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b",
               "google/gemma-4-31b"]
        panel, stub = self._panel(ids, {i: "the sky is blue" for i in ids})
        with mock.patch("urllib.request.urlopen", stub):
            consensus = panel.ask("what colour is the sky?")
        self.assertEqual(consensus.voices, 3)
        self.assertTrue(consensus.corroborated)

    def test_disagreement_is_reported_not_resolved(self):
        """A split is the correct output, not a failure to produce an answer."""
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        panel, stub = self._panel(ids, {ids[0]: "python is best",
                                        ids[1]: "scheme is best"})
        with mock.patch("urllib.request.urlopen", stub):
            consensus = panel.ask("which language is best?")
        self.assertEqual(len(consensus.split), 2)
        self.assertFalse(consensus.agreed)
        self.assertFalse(consensus.corroborated,
                         "a two-way split has one voice per position")

    def test_refusals_are_not_positions(self):
        """Several models 'agreeing' they don't know must not corroborate."""
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b",
               "google/gemma-4-31b"]
        panel, stub = self._panel(ids, {ids[0]: "I don't know.",
                                        ids[1]: "I'm not sure about that.",
                                        ids[2]: "Paris is the capital."})
        with mock.patch("urllib.request.urlopen", stub):
            consensus = panel.ask("what is the capital of France?")
        self.assertEqual(len(consensus.split), 1)
        self.assertFalse(consensus.corroborated)
        self.assertEqual(len(consensus.errors), 2)

    def test_one_model_failing_does_not_sink_the_panel(self):
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        replies = {"/api/v0/models": _CATALOGUE,
                   "/v1/models": _openai_models([r["id"] for r in
                                                 _CATALOGUE["data"]])}

        def flaky(body):
            if body["model"] == ids[1]:
                raise urllib.error.URLError("model crashed")
            return _chat("an answer", model=body["model"])

        replies["/chat/completions"] = flaky
        stub = StubServer(replies)
        with mock.patch("urllib.request.urlopen", stub):
            panel = TeacherPanel(ids)
            panel.cli = None
            consensus = panel.ask("anything?")
        self.assertEqual(len(consensus.answers), 1)
        self.assertIn(ids[1], consensus.errors)

    def test_the_caveat_is_always_present(self):
        """Independence is a lower bound, and the output must keep saying so."""
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        panel, stub = self._panel(ids, {i: "same" for i in ids})
        with mock.patch("urllib.request.urlopen", stub):
            consensus = panel.ask("q?")
        self.assertIn("LOWER BOUND", consensus.caveat)
        self.assertIn("never proof", consensus.caveat)

    def test_models_never_see_each_other_answers(self):
        """Agreement by contagion is indistinguishable from corroboration."""
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        panel, stub = self._panel(ids, {i: "answer" for i in ids})
        with mock.patch("urllib.request.urlopen", stub):
            panel.ask("the question")
        for url, body in stub.requests:
            if "chat" in url:
                text = json.dumps(body)
                self.assertNotIn("answer", text,
                                 "a prompt contained another model's reply")

    def test_the_independence_report_names_merged_models(self):
        panel, _ = self._panel(["qwen/qwen3-coder-30b", "qwen/qwen3-coder-next"])
        report = panel.independence_report()
        self.assertIn("ONE source", report)
        self.assertIn("2 model(s) -> 1 independent", report)


class PositionNormalisationTests(unittest.TestCase):
    """Orthography is normalised; meaning is never guessed at.

    The boundary is the point. Rewriting "two" to "2" is the same class of
    operation as lowercasing - two spellings of one token. Deciding that
    "fast" and "efficient" are one position would be a claim about meaning,
    and merging them would manufacture consensus.
    """

    def test_number_spellings_are_one_position(self):
        """Measured live: three models answered 'two moons', 'two' and '2'."""
        from ultraquant.interpreter.llmls import _position

        question = "How many moons does Mars have?"
        forms = ["two moons", "two", "2", "Two moons.", "The answer is 2."]
        positions = {_position(form, question) for form in forms}
        self.assertEqual(len(positions), 1, f"should be unanimous: {positions}")

    def test_different_answers_never_merge(self):
        from ultraquant.interpreter.llmls import _position

        question = "What is the capital of Australia?"
        answers = ["Canberra", "Sydney", "Melbourne"]
        positions = {_position(a, question) for a in answers}
        self.assertEqual(len(positions), 3)

    def test_synonyms_are_not_merged(self):
        """The refused capability, asserted so it cannot creep in later."""
        from ultraquant.interpreter.llmls import _position

        question = "How would you describe quicksort?"
        self.assertNotEqual(_position("fast", question),
                            _position("efficient", question))

    def test_question_echo_is_stripped(self):
        from ultraquant.interpreter.llmls import _position

        question = "What is the capital of Australia?"
        self.assertEqual(_position("The capital of Australia is Canberra",
                                   question),
                         _position("Canberra", question))

    def test_an_all_echo_answer_does_not_collapse_to_empty(self):
        """Otherwise every content-free reply would 'agree' with every other."""
        from ultraquant.interpreter.llmls import _position

        question = "What is a square?"
        first = _position("a square", question)
        second = _position("a circle", question)
        self.assertTrue(first)
        self.assertNotEqual(first, second)

    def test_prose_replies_are_flagged_not_silently_merged(self):
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        replies = {"/api/v0/models": _CATALOGUE,
                   "/v1/models": _openai_models([r["id"] for r in
                                                 _CATALOGUE["data"]])}
        long = ("The time complexity of binary search is O(log n) because the "
                "algorithm halves the remaining search space on every step of "
                "its iteration until it finds the target value")
        replies["/chat/completions"] = lambda b: _chat(long, model=b["model"])
        stub = StubServer(replies)
        with mock.patch("urllib.request.urlopen", stub):
            panel = TeacherPanel(ids)
            panel.cli = None
            consensus = panel.ask("complexity of binary search?")
        self.assertIn("FLOOR", consensus.prose_warning)


class QuarantineTests(unittest.TestCase):
    """Nothing a model says is believed on arrival."""

    def setUp(self):
        import tempfile, os
        from ultraquant.interpreter.stash import ContemporaryStash

        self.tmp = tempfile.mkdtemp()
        self.stash = ContemporaryStash(os.path.join(self.tmp, "stash.json"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_an_answer_enters_quarantined(self):
        answer = Answer(text="Water boils at 100 degrees Celsius at sea level.",
                        model="m1", prompt="boiling point?")
        ids = quarantine_answer(answer, self.stash)
        self.assertTrue(ids)
        for entry_id in ids:
            # "staged" is the stash's word for quarantined: recorded, not
            # believed. Only promote() moves a claim into memory.
            self.assertEqual(self.stash.get(entry_id)["status"], "staged")

    def test_provenance_marks_the_model(self):
        answer = Answer(text="A fact about something.", model="m1", prompt="q")
        ids = quarantine_answer(answer, self.stash)
        self.assertTrue(self.stash.get(ids[0])["url"].startswith("lmstudio://"))

    def test_one_model_answering_twice_is_still_one_source(self):
        """The single-source limit, which no amount of asking can escape."""
        first = Answer(text="The claim is that X holds.", model="m1", prompt="q")
        second = Answer(text="The claim is that X holds.", model="m1", prompt="q2")
        quarantine_answer(first, self.stash)
        quarantine_answer(second, self.stash)
        urls = {entry["url"] for entry in self.stash.entries()}
        self.assertEqual(len(urls), 1, "same model must collapse to one source")


if __name__ == "__main__":
    unittest.main()
