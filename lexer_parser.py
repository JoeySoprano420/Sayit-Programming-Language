# say_lexer.py
import re

TOKENS = [
    ("START", r"Start"),
    ("RITUAL", r"Ritual:"),
    ("MAKE", r"Make:"),
    ("WHILE", r"While"),
    ("IF", r"If"),
    ("ELIF", r"Elif"),
    ("ELSE", r"Else:"),
    ("FINALLY", r"Finally:"),
    ("PRINT", r"print"),
    ("STRING", r'"[^"]*"'),
    ("NUMBER", r"[0-9]+"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("OP", r"[+\-*/<>=!]+"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("DOTS", r"\.\.\."),
    ("COLON", r":"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t]+"),
]

def tokenize(code):
    regex = "|".join("(?P<%s>%s)" % pair for pair in TOKENS)
    scanner = re.compile(regex).scanner(code)
    for m in iter(scanner.match, None):
        if m.lastgroup == "SKIP": continue
        yield (m.lastgroup, m.group())
