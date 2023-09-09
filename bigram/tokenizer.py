def tokenize(text: str):
    tokens = []
    current = ""
    for c in f"{text}$":
        if c.isalpha():
            current = f"{current}{c}"
        else:
            if current:
                tokens.append(current)
            current = ""
            tokens.append(str(c))
    return tokens[:-1]
