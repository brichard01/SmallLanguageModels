import pandas as pd

search_tools_df = pd.read_csv("data/harmonized_system_by_parent.csv")
sections_df = pd.read_csv("data/sections_prepared.csv")

section_candidates = '\n'.join(sections_df.section_letter + ' : ' + sections_df.name)

SYSTEM_PROMPT = f"""Determine the correct 6-digit code for a product description using only the provided tools.

The hierarchy is:
Section (capital letter) > 2-digit chapter > 4-digit heading > 6-digit subheading

General approach:
- Use the tools to navigate the hierarchy step by step.
- Consider only the codes returned by the tools at each level.
- Never anticipate, infer, invent, or guess codes that have not been explicitly returned by a tool.
- Do not rely on prior knowledge of the classification system; the tools are the sole source of truth.
- When moving to a deeper level, choose only among the candidates provided by the previous tool call.
- If the correct code cannot be determined from the available tool outputs, continue exploring with the tools rather than guessing.
- Keep reasoning concise and focused on selecting the next tool action.
- Use only one tool at a time.
- If no suitable code is found at the current level, always go back to the most relevant higher level and explore other branches. Never force a classification into an unsuitable code.

Available sections:
{section_candidates}
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_section_children",
            "description": "Search 2-digit codes by section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_letter": {"type": "string", "description": "Capital letter of a section (A-U)."}
                },
                "required": ["section_letter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code_children",
            "description": "Search children codes of a 2-digit or 4-digit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "String code (2 or 4 digits)."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_code",
            "description": "Submit the final correct 6-digit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "String code (6 digits)."}
                },
                "required": ["code"],
            },
        },
    },
]


class HSCodeEnv:
    search_tools_dataframe = search_tools_df

    def reset(self, **kwargs):
        self.available_codes = set()
        self.searched = set()
        self.submitted = None
        self.data = kwargs
        self.reward = 0
        self.total_calls = 0
        return None

    def search_section_children(self, section_letter: str) -> str:
        """Search the 2-digit chapter codes contained in a section.

        Args:
            section_letter: Capital letter of a section (A-U).
        """
        if self.submitted is not None:
            return "Submission already made. Classification complete, no further tool calls needed."
        self.total_calls += 1
        self.reward -= 3
        if section_letter not in [chr(i) for i in range(ord("A"), ord("U") + 1)]:
            return "section_letter must be a capital letter between A and U."
        result = self.search_tools_dataframe[self.search_tools_dataframe.group == section_letter]
        if result.empty:
            return f"No data found for section {section_letter}."
        self.available_codes.update(result.hscode_concat.item().split(', '))
        self.searched.add(section_letter)
        self.reward += 3.5
        if section_letter == self.data.get('section'):
            self.reward += 0.5
        return f'The child codes under section {section_letter} are :\n' + result.aggregate_concat.item()

    def search_code_children(self, code: str) -> str:
        """Search the child codes of a 2-digit or 4-digit code.

        Args:
            code: String code (2 or 4 digits).
        """
        if self.submitted is not None:
            return "Submission already made. Classification complete, no further tool calls needed."
        self.total_calls += 1
        self.reward -= 3
        if not isinstance(code, str):
            return "Code must be provided as a string."
        code = code.strip()
        if not (code.isdigit() and len(code) in [2, 4]):
            return "Code must be a 2 or 4 digit string."
        if code not in self.available_codes:
            return "You can't search for a code that was not returned by a tool."
        result = self.search_tools_dataframe[self.search_tools_dataframe.group == code]
        if result.empty:
            return f"No child codes found for code {code}."
        self.available_codes.update(result.hscode_concat.item().split(', '))
        self.searched.add(code)
        self.reward += 3.5
        if code == self.data.get('hs_2'):
            self.reward += 1.5
        elif code == self.data.get('hs_4'):
            self.reward += 2.5
        return f"The child codes under code {code} are:\n" + result.aggregate_concat.item()

    def submit_final_code(self, code: str) -> str:
        """Submit the final correct 6-digit code.

        Args:
            code: String code (6 digits).
        """
        if self.submitted is not None:
            return "Code already submitted. Classification complete, do not call any more tools."
        self.total_calls += 1
        if not isinstance(code, str):
            return "Code must be provided as a string."
        code = code.strip()
        if not (code.isdigit() and len(code) == 6):
            return "Final code must be a 6 digit string."
        if code not in self.available_codes:
            return "You can't submit a code that was not returned by a tool."
        self.submitted = code
        if code == self.data.get('answer'):
            self.reward += 10
        return "Your final code was submitted. Classification complete, do not call any more tools."
