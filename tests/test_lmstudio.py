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

    def test_lineage_key_is_lowercased_and_reduced_to_a_family(self):
        key = lineage_key(_card("Qwen/Qwen3-30B", "Qwen3MoE", "Qwen"))
        self.assertEqual(key[0], "qwen", "arch is reduced to its family")
        self.assertEqual(key[1], "qwen")

    def test_renamed_qwen_derivatives_are_one_voice(self):
        """The dangerous direction, found on a real catalogue.

        Three Qwen derivatives defeated all three signals at once - different
        arch strings (qwen35moe / qwen3moe / qwen35), different publishers, and
        name stems mangled beyond recognition - and were counted as three
        independent voices. Under-grouping manufactures evidence, which is the
        one error this module exists to prevent.
        """
        cards = [
            _card("qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive",
                  "qwen35moe", "HauhauCS"),
            _card("qwen3-48b-a4b-savant-commander-distill-12x", "qwen3moe",
                  "DavidAU"),
            _card("katarau-9b-ru-rp-nsfw", "qwen35", "mradermacher"),
        ]
        for card in cards:
            self.assertNotEqual(card.publisher, cards[0].publisher
                                if card is not cards[0] else "x")
        self.assertEqual(len(independent_groups(cards)), 1,
                         "one base family must not pass as three voices")

    def test_the_arch_family_reduction_keeps_real_families_apart(self):
        """Blunter grouping must not collapse everything into one voice."""
        from ultraquant.interpreter.llmls import _arch_family

        self.assertEqual(_arch_family("qwen35moe"), "qwen")
        self.assertEqual(_arch_family("qwen3moe"), "qwen")
        self.assertEqual(_arch_family("gemma4"), "gemma")
        self.assertEqual(_arch_family("command-r"), "command")
        self.assertEqual(_arch_family("gpt-oss"), "gpt")
        self.assertNotEqual(_arch_family("qwen3moe"), _arch_family("gemma4"))
        self.assertNotEqual(_arch_family("llama"), _arch_family("command-r"))


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

    def test_an_already_loaded_model_is_not_loaded_again(self):
        """`lms load` on a resident model loads a SECOND copy, not a no-op.

        Caught by reading `lms ps` after a few panel runs: 61.5 GB resident
        across gpt-oss-20b, gpt-oss-20b:2, qwen3-coder-30b and
        qwen3-coder-30b:2 - half of it duplicates this code had created.
        """
        panel, _ = self._panel(["qwen/qwen3-coder-30b"])
        shelled = []
        panel.cli = "lms"
        panel.is_loaded = lambda _id: True
        with mock.patch("subprocess.run",
                        lambda *a, **k: shelled.append(a) or None):
            self.assertTrue(panel.load("qwen/qwen3-coder-30b"))
        self.assertEqual(shelled, [], "no second instance may be spawned")

    def test_a_model_that_is_not_loaded_is_loaded(self):
        panel, _ = self._panel(["qwen/qwen3-coder-30b"])
        panel.cli = "lms"
        panel.is_loaded = lambda _id: False
        calls = []

        class _Done:
            returncode = 0

        with mock.patch("subprocess.run",
                        lambda *a, **k: calls.append(a[0]) or _Done()):
            self.assertTrue(panel.load("qwen/qwen3-coder-30b"))
        self.assertTrue(calls and "load" in calls[0])

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


class GroundedQuestionTests(unittest.TestCase):
    """Under-specified questions look exactly like hard ones to a panel.

    Reported live: asked "what is code?" cold, gpt-oss answered "set of
    instructions for a computer" and Qwen answered "secret communication
    system". Both correct, about different senses of a polysemous word, and
    the panel recorded a split that read as unreliability. The system already
    held the sentences that settle which sense it meant.
    """

    def test_usages_are_attached_to_the_prompt(self):
        from ultraquant.interpreter.llmls import grounded_prompt

        prompt = grounded_prompt("What is code?",
                                 ["the code function runs one line of python"])
        self.assertIn("What is code?", prompt)
        self.assertIn("runs one line of python", prompt)

    def test_no_usages_leaves_the_question_untouched(self):
        """Nothing may depend on context existing."""
        from ultraquant.interpreter.llmls import grounded_prompt

        self.assertEqual(grounded_prompt("What is code?"), "What is code?")
        self.assertEqual(grounded_prompt("What is code?", []), "What is code?")
        self.assertEqual(grounded_prompt("What is code?", ["", "  "]),
                         "What is code?")

    def test_only_a_few_usages_are_sent(self):
        from ultraquant.interpreter.llmls import grounded_prompt

        prompt = grounded_prompt("q?", [f"usage {i}" for i in range(9)], limit=3)
        self.assertEqual(prompt.count("- usage"), 3)

    def test_the_panel_sends_context_but_records_the_plain_question(self):
        """A stored claim is about the question, not about the prompt."""
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        replies = {"/api/v0/models": _CATALOGUE,
                   "/v1/models": _openai_models([r["id"] for r in
                                                 _CATALOGUE["data"]]),
                   "/chat/completions": lambda b: _chat("instructions",
                                                        model=b["model"])}
        stub = StubServer(replies)
        with mock.patch("urllib.request.urlopen", stub):
            panel = TeacherPanel(ids)
            panel.cli = None
            consensus = panel.ask("What is code?",
                                  usages=["run this code and show the output"])
        sent = [b for url, b in stub.requests if "chat" in url]
        self.assertTrue(sent)
        for body in sent:
            self.assertIn("show the output",
                          body["messages"][-1]["content"],
                          "context should reach the model")
        self.assertEqual(consensus.question, "What is code?",
                         "the recorded question must stay clean")


class OverlapNoteTests(unittest.TestCase):
    """Reported, never credited - the entropy assessor's monobit rule, here.

    Grounding a question fixes which *sense* is asked about; it cannot make two
    models word a definition identically. The note says so; it must never move
    a verdict.
    """

    def _note(self, *positions):
        from ultraquant.interpreter.llmls import _overlap_note

        return _overlap_note(list(positions))

    def test_the_same_claim_worded_differently_is_flagged(self):
        note = self._note("sequence instructions executed computer",
                          "computer instructions written programming language")
        self.assertIn("worded differently", note)
        self.assertIn("NOT counted as agreement", note)

    def test_genuinely_different_answers_are_not_flagged(self):
        self.assertEqual(self._note("canberra", "sydney"), "")
        self.assertEqual(self._note("set instructions computer",
                                    "secret communication system"), "")

    def test_a_contested_question_is_not_flagged(self):
        """Two answers both naming vim are still two positions."""
        self.assertEqual(self._note("vim widely regarded as powerful",
                                    "vim emacs vs code configuration editing"),
                         "")

    def test_one_position_has_nothing_to_overlap_with(self):
        self.assertEqual(self._note("au"), "")
        self.assertEqual(self._note(), "")

    def test_the_note_never_changes_corroboration(self):
        """The whole point: an observation, not evidence."""
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        replies = {"/api/v0/models": _CATALOGUE,
                   "/v1/models": _openai_models([r["id"] for r in
                                                 _CATALOGUE["data"]])}
        answers = {ids[0]: "computer instructions written programming language",
                   ids[1]: "sequence instructions executed computer"}
        replies["/chat/completions"] = lambda b: _chat(answers[b["model"]],
                                                       model=b["model"])
        stub = StubServer(replies)
        with mock.patch("urllib.request.urlopen", stub):
            panel = TeacherPanel(ids)
            panel.cli = None
            consensus = panel.ask("What is code?")
        self.assertTrue(consensus.overlap_note, "the overlap should be noted")
        self.assertFalse(consensus.corroborated,
                         "a note must not buy corroboration")
        self.assertEqual(len(consensus.split), 2, "still two positions")


class StarvedReplyTests(unittest.TestCase):
    """Truncation and abstention need different fixes, so they read differently."""

    def _reason(self, text, completion=0, reasoning=0):
        from ultraquant.interpreter.llmls import _no_answer_reason

        usage = {"completion_tokens": completion,
                 "completion_tokens_details": {"reasoning_tokens": reasoning}}
        return _no_answer_reason(Answer(text=text, model="m", prompt="q",
                                        usage=usage))

    def test_a_reasoning_model_cut_off_says_so(self):
        """Measured: gpt-oss-20b, max_tokens=60, reasoning_tokens=51, empty."""
        reason = self._reason("", completion=60, reasoning=51)
        self.assertIn("reasoning", reason)
        self.assertIn("max_tokens", reason)

    def test_a_runaway_reasoner_is_told_apart_from_a_starved_one(self):
        """They need different fixes, so they must not read the same.

        Starvation: gpt-oss-20b spent 51 of 60 tokens thinking; raising the
        budget to 200 fixed it. Runaway: qwen3.6-35b-a3b spent 400 of 400 and
        then 1200 of 1200, empty both times - raising the budget only buys
        more thinking, and telling the reader to raise it sends them down a
        dead end. Only reasoning_effort="none" worked.
        """
        starved = self._reason("", completion=60, reasoning=51)
        runaway = self._reason("", completion=400, reasoning=400)
        self.assertIn("raise max_tokens", starved)
        self.assertIn("will not help", runaway)
        self.assertIn("reasoning_effort", runaway)
        self.assertNotEqual(starved, runaway)

    def test_an_empty_reply_without_reasoning_is_just_empty(self):
        self.assertEqual(self._reason("", completion=60, reasoning=0),
                         "empty reply")

    def test_a_refusal_is_reported_as_a_refusal(self):
        self.assertEqual(self._reason("I don't know."), "declined to answer")

    def test_the_default_budget_clears_the_measured_starvation(self):
        from ultraquant.interpreter.llmls import DEFAULT_MAX_TOKENS

        self.assertGreater(DEFAULT_MAX_TOKENS, 60,
                           "60 starved a reasoning model in a live run")


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

    def test_a_terse_answer_still_reaches_the_stash(self):
        """The interaction bug: TERSE made answers too short to store.

        Found by running the two together - the panel agreed 3-of-3 that gold
        is 'au' and nothing was quarantined, because add_page needs 20-420
        characters. Constraining replies for comparability had silently
        disabled the quarantine path.
        """
        from ultraquant.interpreter.llmls import _as_claim

        claim = _as_claim("What is the chemical symbol for gold?", "au")
        self.assertGreaterEqual(len(claim), 20)
        self.assertIn("au", claim)
        self.assertIn("gold", claim)
        self.assertNotIn("?", claim)

    def test_a_terse_panel_answer_is_filed_end_to_end(self):
        ids = ["qwen/qwen3-coder-30b", "openai/gpt-oss-20b"]
        replies = {"/api/v0/models": _CATALOGUE,
                   "/v1/models": _openai_models([r["id"] for r in
                                                 _CATALOGUE["data"]]),
                   "/chat/completions": lambda b: _chat("au", model=b["model"])}
        stub = StubServer(replies)
        with mock.patch("urllib.request.urlopen", stub):
            panel = TeacherPanel(ids)
            panel.cli = None
            result = panel.teach("What is the chemical symbol for gold?",
                                 self.stash)
        self.assertTrue(result["consensus"].corroborated)
        self.assertGreater(result["filed"], 0,
                           "a corroborated terse answer must be quarantined")

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
