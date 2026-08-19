"""Environment variant for the submit-then-verify technique.

Two differences from `HSCodeEnv`: codes can be searched in batches, and
`submit_final_code` is only *provisional* — it triggers a mandatory re-inspection
of sibling branches before a terminal `finish`. This gives the model an explicit
chance to overrule itself.
"""

import hscode_env

_DF = hscode_env.search_tools_df

SYSTEM_PROMPT = f"""Determine the correct 6-digit code for a product description using only the tools.

The hierarchy is: Section (capital letter) > 2-digit chapter > 4-digit heading > 6-digit subheading.

- Navigate step by step with the tools. Consider only codes returned by the tools.
- Never invent or guess codes that a tool has not returned.
- You can search several codes in a single search_code_children call by passing a list.
- When you call submit_final_code, it is only provisional: you must then inspect the
  sibling branches and any alternative you skipped to be sure nothing fits better.
- You may submit_final_code again to overrule a previous submission.
- Call finish only once you are confident no better path was left unexplored.

Available sections:
{hscode_env.section_candidates}
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_section_children",
        "description": "List the 2-digit chapters under a section.",
        "parameters": {"type": "object", "properties": {
            "section_letter": {"type": "string", "description": "Capital letter A-U."}},
            "required": ["section_letter"]}}},
    {"type": "function", "function": {
        "name": "search_code_children",
        "description": "List children of one or more 2- or 4-digit codes.",
        "parameters": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"},
                      "description": "One or more 2- or 4-digit codes."}},
            "required": ["codes"]}}},
    {"type": "function", "function": {
        "name": "submit_final_code",
        "description": "Provisionally submit a 6-digit code (can be overruled later).",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "6-digit code."}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Finalize once confident nothing better was left unexplored.",
        "parameters": {"type": "object", "properties": {}}}},
]


class InspectEnv:
    df = _DF

    def reset(self, **kwargs):
        self.available_codes = set()
        self.submitted = None
        self.done = False
        self.total_calls = 0
        self.reward = 0

    def search_section_children(self, section_letter):
        self.total_calls += 1
        result = self.df[self.df.group == section_letter]
        if result.empty:
            return f"No data for section {section_letter}."
        self.available_codes.update(result.hscode_concat.item().split(", "))
        return f"Chapters under section {section_letter}:\n" + result.aggregate_concat.item()

    def search_code_children(self, codes):
        self.total_calls += 1
        if isinstance(codes, str):
            codes = [codes]
        outputs = []
        for code in codes:
            code = code.strip()
            if code not in self.available_codes:
                outputs.append(f"{code}: not available - must come from a prior tool result.")
                continue
            result = self.df[self.df.group == code]
            if result.empty:
                outputs.append(f"{code}: no children.")
                continue
            self.available_codes.update(result.hscode_concat.item().split(", "))
            outputs.append(f"Children of {code}:\n" + result.aggregate_concat.item())
        return "\n\n".join(outputs)

    def submit_final_code(self, code):
        self.total_calls += 1
        code = code.strip()
        if not (code.isdigit() and len(code) == 6):
            return "Final code must be a 6-digit string."
        if code not in self.available_codes:
            return "You can't submit a code that was not returned by a tool."
        self.submitted = code
        return (f"Recorded {code} provisionally. Now inspect everything: re-check the sibling "
                "headings/subheadings and any branch you skipped to make sure no better code "
                "exists. Submit again to overrule, or call finish when you are confident.")

    def finish(self):
        self.total_calls += 1
        if self.submitted is None:
            return "Nothing submitted yet - call submit_final_code first."
        self.done = True
        return f"Finished. Final code: {self.submitted}."
