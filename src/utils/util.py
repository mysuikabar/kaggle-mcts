def parse_prefix_tree(s: str) -> None:
    """
    https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/532795
    """
    stack = []
    indent = 0
    current_line = ""

    def flush_line() -> None:
        nonlocal current_line
        if current_line.strip():
            print(f"Level {indent:02} - " + ". " * indent, current_line.strip())
        current_line = ""

    i = 0
    while i < len(s):
        char = s[i]
        if char == "(" or char == "{":
            flush_line()
            stack.append(char)
            indent += 1
        elif char == ")" or char == "}":
            flush_line()
            indent -= 1
        elif char not in "(){}":
            j = i
            while j < len(s) and s[j] not in "(){}":
                j += 1
            current_line += s[i:j].strip()
            i = j - 1
        if current_line and (char == ")" or char == "}"):
            flush_line()
        i += 1
    flush_line()
