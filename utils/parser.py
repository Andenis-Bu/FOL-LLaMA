import re

def parse_fol_response(text: str) -> str | None:
    candidates = []

    candidates.extend(
        block.strip()
        for block in re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    )

    candidates.extend(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    scored = []
    for c in candidates:
        score = 0

        score += 3 * len(re.findall(r"[∀∃]", c))
        score += 2 * len(re.findall(r"[→∧∨¬↔⇒⊕]", c))
        score += 3 * len(re.findall(r"\w+\([a-z],?\s*[a-z]?\)", c))

        if re.match(r"^\s*-\s+", c):
            score -= 10

        score -= len(re.findall(r"[A-Za-z]{4,}", c))
        score -= max(0, len(c) - 120) / 20

        if score > 0:
            scored.append((score, c))

    if not scored:
        return None

    best = max(scored, key=lambda x: x[0])[1]
    if best.startswith("FOL:"):
        best = best[len("FOL:"):].strip()
    return best