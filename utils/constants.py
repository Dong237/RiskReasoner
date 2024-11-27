from enum import Enum


class Prompts(Enum):
    # System Prompts
    SYSTEM_PROMPT_CREDIT_SCORING = (
        "You are a risk management assistant who is good at credit scoring."
    )

    # Instructions
    INSTRUCTION_FREESTYLE = (
        "You are given a text as the description of a customer's credit report about "
        "his/her loan status, and also the predicted loan status (probability) by a "
        "trained machine learning system.\n\n"
        "Please analyse the given information and give your reasoning steps, also use your prior knowledge. "
        "Give your final answer in the end in this format: "
        "'My final assessment: [choose from \"good\" or \"bad\"]' "
        "(Make sure you use lower case).\n"
    )

    INSTRUCTION_REGULARIZED = (
        "You are given a text as the description of a customer's credit report about "
        "his/her loan status, and also the predicted loan status (probability) by a "
        "trained machine learning system.\n\n"
        "Please analyse the given information and give your reasoning steps, also use your prior knowledge. "
        "You may follow these steps below and give your final answer in the end by saying 'My final assessment:'.\n\n"
        "### Steps for Assessing Loan Status\n\n"
        "1. **Understand the Objective**: Approach the task methodically, ensuring all relevant factors are considered "
        "before making a decision.\n"
        "2. **Evaluate Financial Stability**: Analyze indicators that reflect the client’s capacity to manage financial obligations.\n"
        "3. **Analyze Credit History**: Examine patterns and metrics that signify creditworthiness and potential risks.\n"
        "4. **Examine Loan Details**: Assess the context and characteristics of the loan to identify alignment with the client’s profile.\n"
        "5. **Review Supporting Factors**: Consider additional evidence that provides further confidence or raises concerns.\n"
        "6. **Integrate External Insights**: Leverage external predictions or tools to complement your evaluation.\n"
        "7. **Make a Decision**: Combine all insights to arrive at a clear and reasoned classification of \"good\" or \"bad\". "
        "Give your decision in this format: 'My final assessment: [choose from \"good\" or \"bad\"]' (Make sure you use lower case).\n"
    )
