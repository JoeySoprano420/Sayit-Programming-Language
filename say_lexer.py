# say_lexer.py
import re
from typing import Iterator, Tuple

# === Token definitions ===
TOKENS = [
    ("START",   r"Start"),
    ("RITUAL",  r"Ritual:"),
    ("MAKE",    r"Make:"),
    ("WHILE",   r"While"),
    ("IF",      r"If"),
    ("ELIF",    r"Elif"),
    ("ELSE",    r"Else:"),
    ("FINALLY", r"Finally:"),
    ("PRINT",   r"print"),
    ("STRING",  r'"[^"]*"'),
    ("NUMBER",  r"[0-9]+"),
    ("IDENT",   r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("OP",      r"[+\-*/<>=!]+"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("LBRACK",  r"\["),
    ("RBRACK",  r"\]"),
    ("DOTS",    r"\.\.\."),
    ("COLON",   r":"),
    ("NEWLINE", r"\n"),
    ("SKIP",    r"[ \t]+"),
]

# === Lexer ===
def _build_token_list():
    """
    Return a token list based on TOKENS with a COMMENT token inserted
    (keeps original TOKENS definition unchanged for compatibility).
    """
    t = list(TOKENS)
    names = [name for name, _ in t]
    if "COMMENT" not in names:
        try:
            idx = names.index("NEWLINE")
        except ValueError:
            idx = len(t)
        t.insert(idx, ("COMMENT", r"#.*"))
    return t

def tokenize(code: str, include_pos: bool = False) -> Iterator[Tuple]:
    """
    Convert source code into a sequence of tokens.

    Backward-compatible default: yields (TOKEN_TYPE, LEXEME).
    If include_pos=True yields (TOKEN_TYPE, LEXEME, LINE, COLUMN).

    This implementation:
    - Adds simple `# comment` support (skipped).
    - Performs explicit scanning to detect illegal characters and report location.
    """
    token_list = _build_token_list()
    regex = "|".join("(?P<%s>%s)" % pair for pair in token_list)
    token_re = re.compile(regex)
    pos = 0
    length = len(code)

    while pos < length:
        m = token_re.match(code, pos)
        if not m:
            # compute line/col for error message
            line = code.count("\n", 0, pos) + 1
            last_n = code.rfind("\n", 0, pos)
            col = pos - last_n
            raise SyntaxError(f"Illegal character {code[pos]!r} at line {line} column {col}")
        typ = m.lastgroup
        val = m.group()
        pos = m.end()

        if typ in ("SKIP", "COMMENT"):
            continue

        if include_pos:
            # compute line and column from m.start()
            start = m.start()
            line = code.count("\n", 0, start) + 1
            last_n = code.rfind("\n", 0, start)
            col = start - last_n
            yield (typ, val, line, col)
        else:
            yield (typ, val)


if __name__ == "__main__":
    # Lightweight CLI: print tokens for a given file. Use --pos to include line/column.
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="say_lexer", description="Tokenize a .say source file")
    parser.add_argument("file", nargs="?", help="Source file to tokenize")
    parser.add_argument("--pos", action="store_true", help="Include token line and column")
    args = parser.parse_args()

    if not args.file:
        print("Usage: say_lexer.py <file.say> [--pos]")
        sys.exit(1)

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"Error reading {args.file}: {e}")
        sys.exit(1)

    try:
        for tok in tokenize(src, include_pos=args.pos):
            print(tok)
    except SyntaxError as e:
        print(f"Lexing error: {e}")
        sys.exit(1)
