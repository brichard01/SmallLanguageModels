"""Method with a rebuilt env (no rewards): batch code search, a submit that
triggers a re-inspection pass (and can be overruled), and a terminal `finish`."""

import json
import pandas as pd
from openai import OpenAI

search_tools_df = pd.read_csv("data/harmonized_system_by_parent.csv")
sections_df = pd.read_csv("data/sections_prepared.csv")
section_candidates = "\n".join(sections_df.section_letter + " : " + sections_df.name)

client = OpenAI()

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
{section_candidates}
"""

# Responses API tool format (flat, no nested "function" key).
TOOLS = [
    {"type": "function", "name": "search_section_children",
     "description": "List the 2-digit chapters under a section.",
     "parameters": {"type": "object", "properties": {
         "section_letter": {"type": "string", "description": "Capital letter A-U."}},
         "required": ["section_letter"]}},
    {"type": "function", "name": "search_code_children",
     "description": "List children of one or more 2- or 4-digit codes.",
     "parameters": {"type": "object", "properties": {
         "codes": {"type": "array", "items": {"type": "string"},
                   "description": "One or more 2- or 4-digit codes."}},
         "required": ["codes"]}},
    {"type": "function", "name": "submit_final_code",
     "description": "Provisionally submit a 6-digit code (can be overruled later).",
     "parameters": {"type": "object", "properties": {
         "code": {"type": "string", "description": "6-digit code."}},
         "required": ["code"]}},
    {"type": "function", "name": "finish",
     "description": "Finalize once confident nothing better was left unexplored.",
     "parameters": {"type": "object", "properties": {}}},
]


class InspectEnv:
    df = search_tools_df

    def reset(self):
        self.available_codes = set()
        self.submitted = None
        self.done = False

    def search_section_children(self, section_letter):
        result = self.df[self.df.group == section_letter]
        if result.empty:
            return f"No data for section {section_letter}."
        self.available_codes.update(result.hscode_concat.item().split(", "))
        return f"Chapters under section {section_letter}:\n" + result.aggregate_concat.item()

    def search_code_children(self, codes):
        if isinstance(codes, str):
            codes = [codes]
        outputs = []
        for code in codes:
            code = code.strip()
            if code not in self.available_codes:
                outputs.append(f"{code}: not available — must come from a prior tool result.")
                continue
            result = self.df[self.df.group == code]
            if result.empty:
                outputs.append(f"{code}: no children.")
                continue
            self.available_codes.update(result.hscode_concat.item().split(", "))
            outputs.append(f"Children of {code}:\n" + result.aggregate_concat.item())
        return "\n\n".join(outputs)

    def submit_final_code(self, code):
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
        if self.submitted is None:
            return "Nothing submitted yet — call submit_final_code first."
        self.done = True
        return f"Finished. Final code: {self.submitted}."


def run(row, model="gpt-5-nano", max_steps=30, verbal=False):
    env = InspectEnv()
    env.reset()

    # First turn: seed with the product description. Reasoning items are kept on
    # OpenAI's side and carried into each following turn via previous_response_id.
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=row["product_description"],
        tools=TOOLS,
    )

    steps = 0
    for _ in range(max_steps):
        if verbal:
            print(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break

        outputs = []
        for tc in calls:
            result = getattr(env, tc.name)(**json.loads(tc.arguments or "{}"))
            steps += 1
            outputs.append({"type": "function_call_output", "call_id": tc.call_id, "output": result})
        if env.done:
            break

        # Pass only the new tool outputs; previous_response_id retains the rest.
        response = client.responses.create(
            model=model,
            previous_response_id=response.id,
            instructions=SYSTEM_PROMPT,
            input=outputs,
            tools=TOOLS,
        )

    return {
        "submitted": env.submitted,
        "reward": 0,
        "correct": env.submitted == row["answer"],
        "steps": steps,
    }


if __name__ == "__main__":
    row = pd.read_csv("data/benchmark_dataset.csv", dtype=str).to_dict("records")[0]
    print("Product:", row["product_description"])
    print("Answer:", row["answer"])
    results = run(row, verbal=True)
    print("Result:", results)
