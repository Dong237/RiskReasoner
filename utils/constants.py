from enum import Enum

# NOTE the instructions written here are model-specific and sensitive to model changes
# the three instructiosn are tested specifically on Qwen2.5-7B-Instruct and shown to be able to elicit
# proper CoT rationales. In case of model changes, please test the instructions again and adapt.

STEP_TAG = "\n\n"
SPLIT_TOKEN = "Final assessment"
# NOTE change the search pattern accordingly when the SPLIT_TOKEN changes
# also this search pattern is not pefect, it does not capture "\n\n**Final Assessment:** good"
# namely when ** is between "good" and "assessment", but this is a flaw I am not smart enough to balance
SEARCH_PATTERN = r"\s*\*?\*?Final\s*[Aa]ssessment\*?\*?\s*:\s*(good|bad)\*?\*?"

# possible end tokens when inferencing with Qwen2.5-7B
POSSIBLE_END_TOKENS = ['\n   \n', '\n  \n', '\n \n', '\n\n', '.\n\n', ':\n\n', ' \n\n']

class Prompts(Enum):
    # System Prompts
    SYSTEM_PROMPT_CREDIT_SCORING = (
        "You are a risk management assistant who is good at credit scoring."
    )

    # Instructions
    INSTRUCTION_FREESTYLE = (
        "You are given a text as the description of a customer's credit report about "
        "his/her loan status, and also the predicted loan status (probability) by a "
        "trained machine learning system as reference (and reference only).\n\n"
        "Please analyse the given information and give your reasoning steps, also use your prior knowledge. "
        "Give your final answer in the end in this format: "
        f"{SPLIT_TOKEN} [choose from \"good\" or \"bad\"]' "
        "(Make sure you use lower case and do NOT add any other texts or symbols (e.g., **) after the word \"good\" or \"bad\").\n"
    )

    INSTRUCTION_REGULARIZED = (
        "You are given a text as the description of a customer's credit report about "
        "his/her loan status, and also the predicted loan status (probability) by a "
        "trained machine learning system.\n\n"
        "Please analyse the given information and give your reasoning steps, also use your prior knowledge. "
        "The following steps below can be used as an example, but I encourage you to be flexible and analyse "
        "using different logical steps depending on the specific problem, "
        f"and give your final answer in the end by saying {SPLIT_TOKEN}.\n\n"
        "### Steps for Assessing Loan Status\n\n"
        "1. **Understand the Objective**: Approach the task methodically, ensuring all relevant factors are considered "
        "before making a decision.\n"
        "2. **Evaluate Financial Stability**: Analyze indicators that reflect the client’s capacity to manage financial obligations.\n"
        "3. **Analyze Credit History**: Examine patterns and metrics that signify creditworthiness and potential risks.\n"
        "4. **Examine Loan Details**: Assess the context and characteristics of the loan to identify alignment with the client’s profile.\n"
        "5. **Review Supporting Factors**: Consider additional evidence that provides further confidence or raises concerns.\n"
        "6. **Integrate External Insights**: Leverage external predictions or tools to complement your evaluation.\n"
        "7. **Make a Decision**: Combine all insights to arrive at a clear and reasoned classification of \"good\" or \"bad\". "
        f"Give your decision in this format: {SPLIT_TOKEN}: [choose from \"good\" or \"bad\"] "
        "(Make sure you use lower case and do NOT add any other texts or symbols after the word \"good\" or \"bad\").\n"
    )

    INSTRUCTION_STEP_BY_STEP = (
        "You are given a text describing a customer's credit report and predicted loan status (probabilities) "
        "from a trained machine learning system as reference (and reference only).\n\n"
        "Please analyze the provided information STEP BY STEP following these instructions:\n"
        "1. **During reasoning process**: For each piece of data, evaluate its significance "
        "to the customer's creditworthiness. Then use your prior knowledge to interpret the data "
        "and consider how each factor impacts the final credit risk assessment.\n"
        "2. **When presenting your reasoning steps**:\n"
        "    - Analyse each attribute from the report, discuss its importance, and determine whether it positively "
        "or negatively affects the loan status.\n"
        "    - Combine all factors to form a holistic assessment, taking into account the prediction probabilities "
        "provided by the machine learning model.\n"
        "3. **In the end of your output**: after presenting the reasoning steps, give your final answer in the following "
        f"format: '{SPLIT_TOKEN}: [choose from \"good\" or \"bad\"]'. "
        "End you output with one of the assessment token, this means that your last output token can only be either "
        "\"good\" or \"bad\", not any other texts or symbols. "
        f"For example: don't use **{SPLIT_TOKEN}: good**, instead just say {SPLIT_TOKEN}: good (without '**').\n\n"
        "Here is the customer's credit report:\n"
    )