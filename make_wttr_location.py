"""This script evaluates an LLM prompt for processing text so that it can be used for the wttr.in API"""

from ollama import Client
from pydantic import BaseModel
from typing import Optional




# TODO: define  llm_parse_for_wttr()
def llm_parse_for_wttr(raw_input: str) -> str:
    """Parse the input using the LLM and return the result."""
    
    try:
        LLM_MODEL: str = "gemma3:27b"  #  this is running on the AI server
        client: Client = Client(host="http://ai.dfec.xyz:11434")  # this is the AI server

        response = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You are a weather assistant. Your job is to parse the input text and extract the location for the wttr.in API.
                        The location should be formatted as follows:
                        - All spaces should be replaced with plus signs (+).
                        - For cities, use the city name (e.g., "New+York").
                        - For regions, use the region name (e.g., "California").
                        - For airports, use the IATA code (e.g., "LAX").
                        - For specific locations or landmarks, use a tilde (~) before the location (e.g., "~Grand+Canyon").
                        """,
                },
                {"role": "user", "content": raw_input},
            ],
            model="gemma3:27b",
        )

        return response["message"]["content"]

    except Exception as e:
        print(f"Error occured parsing the location for wttr.in: {e}")
        return None

# # Test cases
# test_cases = [  # TODO: Replace these test cases with ones for wttr.in
#     {
#         "input": "What is the weather at the United States Air Force Academy?",
#         "expected": "~United+States+Air+Force+Academy",
#     },
#     {
#         "input": "What is the weather in Colorado Springs?",
#         "expected": "Colorado+Springs",
#     },
#     {"input": "What is the weather at the Grand Canyon?", "expected": "~Grand+Canyon"},
#     {"input": "What is the weather at the Denver Airport?", "expected": "DEN"},
#     {"input": "Give me the weather in New York", "expected": "New+York"},
#     {"input": "What is the weather in California?", "expected": "California"},
#     {"input": "What is the weather at the Los Angeles Airport?", "expected": "LAX"},
#     {"input": "What is the weather in Chicago?", "expected": "Chicago"},
#     {
#         "input": "What is the weather at the Statue of Liberty?",
#         "expected": "~Statue+of+Liberty",
#     },
#     {"input": "What is the weather in San Francisco?", "expected": "San+Francisco"},
#     {"input": "What is the weather at the Eiffel Tower?", "expected": "~Eiffel+Tower"},
#     {"input": "What is the weather in Paris?", "expected": "Paris"},
#     {
#         "input": "What is the weather at the Sydney Opera House?",
#         "expected": "~Sydney+Opera+House",
#     },
#     {"input": "What is the weather in Tokyo?", "expected": "Tokyo"},
# ]


# # Function to iterate through test cases
# def run_tests():
#     num_passed = 0

#     for i, test in enumerate(test_cases, 1):
#         raw_input = test["input"]
#         expected_output = test["expected"]

#         print(f"\nTest {i}: {raw_input}")
#         try:
#             result = llm_parse_for_wttr(raw_input).strip()
#             expected = expected_output.strip()

#             print("LLM Output  :", result)
#             print("Expected    :", expected)

#             if result == expected:
#                 print("✅ PASS")
#                 num_passed += 1
#             else:
#                 print("❌ FAIL")

#         except Exception as e:
#             print("💥 ERROR:", e)

#     print(f"\nSummary: {num_passed} / {len(test_cases)} tests passed.")


# # Run the test cases
# run_tests()
