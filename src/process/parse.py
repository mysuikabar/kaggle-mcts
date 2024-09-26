import re


def parse_lud_rules(text: str) -> dict[str, str]:
    pattern = r"^\(game\s+\"(.*?)\"\s+\(players\s+(.*?)\)\s+\(equipment\s+(.*?)\)\s+\(rules\s+(.*?)\)\s+\)$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return {
            "LudRules_game": match.group(1),
            "LudRules_players": match.group(2),
            "LudRules_equipment": match.group(3),
            "LudRules_rules": match.group(4),
        }
    else:
        raise ValueError(f"Invalid input text: {text}")
