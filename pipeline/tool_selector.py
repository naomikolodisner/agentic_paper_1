import random


class ToolSelector:
    TOOLS = [
        "VirSorter2", "DeepVirFinder", "GeNomad", "MARVEL",
        "VirFinder", "VIBRANT", "viralVerify", "ViraMiner",
        "MetaPhinder", "Seeker", "VirSorter",
    ]

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.best_tool = None
        self.best_score = -1.0
        self.tool_scores = {}

    def score_from_checkv_blast(self, quality_ratio: float, match_ratio: float) -> float:
        """Kept for future use: combine CheckV quality and BLAST match into one signal."""
        return (quality_ratio + match_ratio) / 2

    def update_tool_score(self, tool_name: str, f1_score: float) -> None:
        """Update the running score for a tool using its F1 score."""
        self.tool_scores[tool_name] = f1_score
        if f1_score > self.best_score:
            self.best_score = f1_score
            self.best_tool = tool_name

    def choose_tool(self, available_tools: list[str] | None = None) -> str:
        """Alpha-greedy selection: exploit best tool with probability alpha, else explore."""
        tools = available_tools if available_tools is not None else self.TOOLS
        if not self.best_tool or self.best_tool not in tools:
            return random.choice(tools)
        if random.random() < self.alpha:
            return self.best_tool
        return random.choice(tools)
