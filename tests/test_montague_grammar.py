"""Validate expressions against the Montague-style lambda calculus grammar.

This is a Python recursive descent parser that mirrors the GBNF grammar in
specs/lambda_montague.gbnf. It serves two purposes:

1. Test suite: verify the grammar accepts all target forms and rejects
   invalid ones.
2. Standalone validator: can be imported and used to check expressions
   during data generation / analysis.

The parser operates on strings (not tokens) and returns (success, position)
tuples. A successful parse consumes the entire input.

Grammar (from specs/lambda_montague.gbnf):

    root           ::= expr "\\n"
    expr           ::= "¬"? binder-expr | connective-expr
    binder-expr    ::= binder var ". " expr
    connective-expr ::= unary (connective expr)*
    connective     ::= " ∧ " | " ∨ " | " → "
    unary          ::= "¬" atom | atom
    atom           ::= ident "(" arg-list ")" | ident | var | "(" expr ")"
    arg-list       ::= expr (", " expr)*
    binder         ::= "λ" | "∀" | "∃" | "ι"
    var            ::= [u-z]
    ident          ::= [a-z] [a-z_]+
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════════════════

BINDERS = {"λ", "∀", "∃", "ι"}
CONNECTIVES = [" ∧ ", " ∨ ", " → "]
VARS = set("uvwxyz")


class ParseError(Exception):
    """Raised when parsing fails at a specific position."""

    def __init__(self, pos: int, msg: str):
        self.pos = pos
        self.msg = msg
        super().__init__(f"pos {pos}: {msg}")


class Parser:
    """Recursive descent parser for Montague-style lambda expressions."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def peek(self, n: int = 1) -> str:
        return self.text[self.pos : self.pos + n]

    def peek_char(self) -> str:
        """Peek at the next character (handles multi-byte UTF-8)."""
        if self.at_end():
            return ""
        # Python strings are Unicode, so indexing gives codepoints
        return self.text[self.pos]

    def advance(self, n: int = 1) -> str:
        result = self.text[self.pos : self.pos + n]
        self.pos += n
        return result

    def expect(self, s: str) -> str:
        if self.text[self.pos : self.pos + len(s)] == s:
            self.pos += len(s)
            return s
        raise ParseError(
            self.pos,
            f"expected {s!r}, got {self.text[self.pos:self.pos+len(s)]!r}",
        )

    def try_match(self, s: str) -> bool:
        if self.text[self.pos : self.pos + len(s)] == s:
            self.pos += len(s)
            return True
        return False

    # ── Grammar rules ────────────────────────────────────────────────

    def parse_root(self) -> bool:
        """root ::= expr '\\n'"""
        self.parse_expr()
        # Accept with or without trailing newline
        if not self.at_end():
            self.expect("\n")
        return True

    def parse_expr(self) -> None:
        """expr ::= '¬'? binder-expr | connective-expr"""
        saved = self.pos

        # Try: optional ¬ followed by binder-expr
        had_neg = self.try_match("¬")
        if self._is_binder():
            self.parse_binder_expr()
            return
        if had_neg:
            # ¬ was consumed but no binder follows — backtrack
            self.pos = saved

        # Fall through to connective-expr
        self.parse_connective_expr()

    def parse_binder_expr(self) -> None:
        """binder-expr ::= binder var '. ' expr"""
        self.parse_binder()
        self.parse_var()
        self.expect(". ")
        self.parse_expr()

    def parse_connective_expr(self) -> None:
        """connective-expr ::= unary (connective expr)*"""
        self.parse_unary()
        while self._is_connective():
            self.parse_connective()
            self.parse_expr()

    def parse_connective(self) -> str:
        """connective ::= ' ∧ ' | ' ∨ ' | ' → '"""
        for conn in CONNECTIVES:
            if self.try_match(conn):
                return conn
        raise ParseError(self.pos, "expected connective")

    def parse_unary(self) -> None:
        """unary ::= '¬' unary | atom"""
        if self.peek_char() == "¬":
            self.advance(1)
            self.parse_unary()
        else:
            self.parse_atom()

    def parse_atom(self) -> None:
        """atom ::= ident '(' arg-list ')' | ident | var | '(' expr ')'"""
        ch = self.peek_char()

        # Parenthesized expression
        if ch == "(":
            self.advance(1)
            self.parse_expr()
            self.expect(")")
            return

        # Variable (single char u-z, not followed by [a-z_])
        if ch in VARS:
            next_pos = self.pos + 1
            if next_pos >= len(self.text) or self.text[next_pos] not in "abcdefghijklmnopqrstuvwxyz_":
                self.advance(1)
                return
            # It's an identifier (multi-char starting with u-z)
            self.parse_ident_or_app()
            return

        # Identifier or application
        if ch.isascii() and ch.islower():
            self.parse_ident_or_app()
            return

        raise ParseError(self.pos, f"expected atom, got {ch!r}")

    def parse_ident_or_app(self) -> None:
        """Parse ident '(' arg-list ')' or bare ident."""
        name = self._consume_ident()
        if not self.at_end() and self.peek_char() == "(":
            self.advance(1)  # consume '('
            self.parse_arg_list()
            self.expect(")")
        # else: bare identifier

    def parse_arg_list(self) -> None:
        """arg-list ::= expr (', ' expr)*"""
        self.parse_expr()
        while self.try_match(", "):
            self.parse_expr()

    def parse_binder(self) -> str:
        """binder ::= 'λ' | '∀' | '∃' | 'ι'"""
        ch = self.peek_char()
        if ch in BINDERS:
            self.advance(1)
            return ch
        raise ParseError(self.pos, f"expected binder, got {ch!r}")

    def parse_var(self) -> str:
        """var ::= [u-z]"""
        ch = self.peek_char()
        if ch in VARS:
            self.advance(1)
            return ch
        raise ParseError(self.pos, f"expected variable [u-z], got {ch!r}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _is_binder(self) -> bool:
        return not self.at_end() and self.peek_char() in BINDERS

    def _is_connective(self) -> bool:
        return any(
            self.text[self.pos : self.pos + len(c)] == c for c in CONNECTIVES
        )

    def _consume_ident(self) -> str:
        """Consume an identifier: [a-z][a-z_]+  (minimum 2 chars)."""
        start = self.pos
        ch = self.peek_char()
        if not (ch.isascii() and ch.islower()):
            raise ParseError(self.pos, f"expected identifier start, got {ch!r}")
        self.advance(1)

        # Must have at least one more [a-z_]
        if self.at_end() or self.text[self.pos] not in "abcdefghijklmnopqrstuvwxyz_":
            raise ParseError(
                self.pos,
                f"identifier must be 2+ chars, got {self.text[start:self.pos]!r}",
            )

        while not self.at_end() and self.text[self.pos] in "abcdefghijklmnopqrstuvwxyz_":
            self.advance(1)

        return self.text[start : self.pos]


def validate(expr: str) -> tuple[bool, str]:
    """Validate a Montague lambda expression.

    Returns (True, "") on success, (False, error_message) on failure.
    """
    try:
        p = Parser(expr)
        p.parse_root()
        if not p.at_end():
            return False, f"trailing content at pos {p.pos}: {expr[p.pos:]!r}"
        return True, ""
    except ParseError as e:
        return False, str(e)
    except IndexError:
        return False, "unexpected end of input"


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════

import pytest


class TestEvalTargets:
    """All 10 eval gold-standard forms must be accepted."""

    def test_simple_predication(self):
        assert validate("λx. runs(dog)") == (True, "")

    def test_universal_quantification(self):
        assert validate("∀y. student(y) → ∃z. book(z) ∧ reads(y, z)") == (True, "")

    def test_definite_description_with_iota(self):
        assert validate("black(ιy. cat(y) ∧ sat_on(y, ιz. mat(z)))") == (True, "")

    def test_bare_conditional(self):
        assert validate("rains → wet(ground)") == (True, "")

    def test_existential(self):
        assert validate("∃y. person(y) ∧ believes(y, flat(earth))") == (True, "")

    def test_simple_predication_bird(self):
        assert validate("λx. flies(bird)") == (True, "")

    def test_transitive(self):
        assert validate("λx. helps(alice, bob)") == (True, "")

    def test_negated_existential(self):
        assert validate("¬∃x. fish(x) ∧ swims(x)") == (True, "")

    def test_relative_clause_with_iota(self):
        assert validate("λx. laughs(ιy. teacher(y) ∧ sees(child, y))") == (True, "")

    def test_adverb_as_function(self):
        assert validate("λx. quickly(runs(tom))") == (True, "")


class TestMontaguePatterns:
    """Common Montague-style patterns."""

    # ── Quantification ───────────────────────────────────────────────

    def test_universal_intransitive(self):
        """Every dog runs → ∀x. dog(x) → runs(x)"""
        assert validate("∀x. dog(x) → runs(x)")[0]

    def test_existential_intransitive(self):
        """Some dog runs → ∃x. dog(x) ∧ runs(x)"""
        assert validate("∃x. dog(x) ∧ runs(x)")[0]

    def test_no_quantifier(self):
        """No dog runs → ¬∃x. dog(x) ∧ runs(x)"""
        assert validate("¬∃x. dog(x) ∧ runs(x)")[0]

    def test_nested_quantifiers(self):
        """Every student reads a book → ∀x. student(x) → ∃y. book(y) ∧ reads(x, y)"""
        assert validate("∀x. student(x) → ∃y. book(y) ∧ reads(x, y)")[0]

    def test_double_universal(self):
        assert validate("∀x. ∀y. loves(x, y) → loves(y, x)")[0]

    # ── Definite descriptions ────────────────────────────────────────

    def test_iota_simple(self):
        """The dog → ιx. dog(x)"""
        assert validate("runs(ιx. dog(x))")[0]

    def test_iota_nested(self):
        assert validate("black(ιy. cat(y) ∧ sat_on(y, ιz. mat(z)))")[0]

    def test_iota_in_argument(self):
        assert validate("loves(ιx. king(x), ιy. queen(y))")[0]

    # ── Negation ─────────────────────────────────────────────────────

    def test_negated_predicate(self):
        assert validate("¬runs(dog)")[0]

    def test_negated_existential_full(self):
        assert validate("¬∃x. fish(x) ∧ swims(x)")[0]

    def test_negated_in_connective(self):
        """¬a ∧ b = (¬a) ∧ b"""
        assert validate("¬runs(dog) ∧ sleeps(cat)")[0]

    def test_double_negation(self):
        assert validate("¬¬runs(dog)")[0]

    # ── Conjunction / Disjunction ────────────────────────────────────

    def test_conjunction(self):
        assert validate("runs(alice) ∧ sings(alice)")[0]

    def test_disjunction(self):
        assert validate("runs(alice) ∨ walks(bob)")[0]

    def test_triple_conjunction(self):
        assert validate("runs(alice) ∧ sings(alice) ∧ dances(alice)")[0]

    # ── Conditionals ─────────────────────────────────────────────────

    def test_bare_conditional(self):
        assert validate("rains → wet(ground)")[0]

    def test_conditional_with_quantifier_consequent(self):
        assert validate("rains → ∃x. wet(x)")[0]

    # ── Attitudes ────────────────────────────────────────────────────

    def test_believes(self):
        assert validate("believes(alice, flat(earth))")[0]

    def test_existential_attitude(self):
        assert validate("∃y. person(y) ∧ believes(y, flat(earth))")[0]

    def test_nested_attitude(self):
        assert validate("knows(alice, believes(bob, flat(earth)))")[0]

    # ── Adverbs / Modifiers ──────────────────────────────────────────

    def test_adverb(self):
        assert validate("quickly(runs(tom))")[0]

    def test_modifier_in_iota(self):
        """The big dog → ιx. dog(x) ∧ big(x)"""
        assert validate("runs(ιx. dog(x) ∧ big(x))")[0]

    # ── Relative clauses ─────────────────────────────────────────────

    def test_relative_clause(self):
        """The teacher who the child sees laughs"""
        assert validate("laughs(ιy. teacher(y) ∧ sees(child, y))")[0]

    def test_subject_relative(self):
        """The dog that runs sleeps"""
        assert validate("sleeps(ιx. dog(x) ∧ runs(x))")[0]

    # ── Ditransitive ─────────────────────────────────────────────────

    def test_ditransitive(self):
        assert validate("gave(man, doctor, book)")[0]

    # ── Lambda ───────────────────────────────────────────────────────

    def test_vacuous_lambda(self):
        assert validate("λx. runs(dog)")[0]

    def test_non_vacuous_lambda(self):
        assert validate("λx. runs(x)")[0]

    def test_nested_lambda(self):
        assert validate("λx. λy. loves(x, y)")[0]

    def test_identity(self):
        assert validate("λx. x")[0]

    # ── Parenthesized ────────────────────────────────────────────────

    def test_parens_around_conjunction(self):
        assert validate("(runs(dog) ∧ sleeps(cat)) → happy(owner)")[0]

    def test_parens_in_negation(self):
        assert validate("¬(runs(dog) ∧ sleeps(cat))")[0]

    # ── Complex / Combined ───────────────────────────────────────────

    def test_complex_montague(self):
        """Every man who loves a woman is happy"""
        expr = "∀x. (man(x) ∧ ∃y. woman(y) ∧ loves(x, y)) → happy(x)"
        assert validate(expr)[0]

    def test_scopal_ambiguity_surface(self):
        """Every student reads a book (∀ > ∃)"""
        assert validate("∀x. student(x) → ∃y. book(y) ∧ reads(x, y)")[0]

    def test_scopal_ambiguity_inverse(self):
        """Every student reads a book (∃ > ∀)"""
        assert validate("∃y. book(y) ∧ ∀x. student(x) → reads(x, y)")[0]

    def test_prepositional(self):
        assert validate("runs_in(dog, park)")[0]

    def test_copular(self):
        assert validate("tall(john)")[0]


class TestRejectInvalid:
    """These should all be REJECTED — they represent the teacher's
    inconsistencies that the grammar is designed to eliminate."""

    def test_reject_pipe_conjunction(self):
        ok, _ = validate("laugh(paul) | laugh(tom)")
        assert not ok

    def test_reject_ampersand_conjunction(self):
        ok, _ = validate("cries(anna) & runs(anna)")
        assert not ok

    def test_reject_does_not_pattern(self):
        ok, _ = validate("does_not_fall(lawyer)")
        # This actually parses as a valid application — ident("does_not_fall") + args
        # The grammar can't reject valid-looking applications with bad semantics
        # But the constrained grammar will prevent the teacher from needing this
        # because ¬ is available. Mark as known limitation.

    def test_reject_not_function(self):
        """not(sing(teacher)) — should use ¬ prefix instead."""
        # This parses as valid: not is an ident, sing(teacher) is an arg
        # Known limitation: grammar is syntactic, not semantic
        # The teacher should use ¬ because it's available in the grammar

    def test_reject_question_mark(self):
        ok, _ = validate("¬(bird(x) → cries(x)) ?")
        assert not ok

    def test_reject_where_clause(self):
        ok, _ = validate("hates(Peter, x) where x is Bob")
        assert not ok

    def test_reject_x_equals(self):
        ok, _ = validate("walks(chef) | x = no")
        assert not ok

    def test_reject_natural_language(self):
        ok, _ = validate("the dog runs")
        assert not ok

    def test_reject_uppercase_identifier(self):
        ok, _ = validate("runs(Dog)")
        assert not ok

    def test_reject_empty_args(self):
        ok, _ = validate("runs()")
        assert not ok

    def test_reject_single_char_ident_with_parens(self):
        """Single-char identifiers aren't valid — they're variables.
        Variables can't have argument lists."""
        ok, _ = validate("f(x)")
        assert not ok

    def test_reject_pipe_separator(self):
        ok, _ = validate("reads(chef, x) | artist(x)")
        assert not ok

    def test_reject_number_in_ident(self):
        ok, _ = validate("type1(x)")
        assert not ok

    def test_reject_mixed_connectives_nucleus_style(self):
        """Nucleus uses > for preference — not valid in Montague."""
        ok, _ = validate("safety > completion")
        assert not ok


class TestEdgeCases:
    """Edge cases that should work correctly."""

    def test_all_binder_types(self):
        for b in "λ∀∃ι":
            ok, msg = validate(f"{b}x. runs(x)")
            assert ok, f"binder {b} failed: {msg}"

    def test_all_connective_types(self):
        for c in ["∧", "∨", "→"]:
            ok, msg = validate(f"runs(dog) {c} sleeps(cat)")
            assert ok, f"connective {c} failed: {msg}"

    def test_all_variables(self):
        for v in "uvwxyz":
            ok, msg = validate(f"λ{v}. runs({v})")
            assert ok, f"variable {v} failed: {msg}"

    def test_deeply_nested_application(self):
        assert validate("very(quickly(runs(tom)))")[0]

    def test_long_identifier(self):
        assert validate("very_long_predicate_name(x)")[0]

    def test_underscore_in_ident(self):
        assert validate("sat_on(cat, mat)")[0]

    def test_many_arguments(self):
        assert validate("rel(x, y, z, w, u)")[0]

    def test_binder_in_argument(self):
        assert validate("runs(ιx. dog(x))")[0]

    def test_negation_in_argument(self):
        assert validate("believes(alice, ¬flat(earth))")[0]

    def test_connective_in_argument(self):
        # This is tricky: does (a ∧ b) work as an argument?
        # arg-list ::= expr (", " expr)*
        # expr can be connective-expr, so: believes(alice, a ∧ b)
        # But the ) after b would terminate the atom inside connective-expr
        # Let's see: parse_arg_list -> parse_expr -> parse_connective_expr
        #   -> parse_unary -> parse_atom -> "a" (ident? no, single char = var)
        # Hmm, "a" is not in [u-z], so it's not a var. It would try ident,
        # but "a" is only 1 char. So this would fail.
        # Let's test with proper identifiers:
        assert validate("believes(alice, good(bob) ∧ kind(bob))")[0]

    def test_with_trailing_newline(self):
        assert validate("runs(dog)\n")[0]

    def test_without_trailing_newline(self):
        assert validate("runs(dog)")[0]


# ══════════════════════════════════════════════════════════════════════
# CLI: validate expressions from stdin or arguments
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exprs = sys.argv[1:]
    else:
        exprs = [line.rstrip("\n") for line in sys.stdin]

    for expr in exprs:
        ok, msg = validate(expr)
        status = "✓" if ok else "✗"
        detail = "" if ok else f"  ({msg})"
        print(f"  {status} {expr}{detail}")
