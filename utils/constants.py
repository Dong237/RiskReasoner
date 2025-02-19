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
    
    SYSTEM_PROMPT_R1_FORMAT = (	
        "You are a risk management assistant who is good at credit scoring and about to do risk assessment."
        "You should answer the query by first thinking about the reasoning process in the mind and then provides "
        "the answer. The reasoning process is enclosed within <think>\n </think>"
        "i.e., <think>\n reasoning process here </think> answer here"
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
    
    EXPLANATION_FEATURES = (
        "Here are the explanations of all features in a customer's redit report you need to consider:\n\n"
        "Loan Information:\n"
        "- Installment: Monthly payment required for the loan, ranging from $30.74 to $1466.04 in the population we study.\n"
        "- Loan Purpose: The reason for taking the loan, with options such as debt consolidation, home improvement, small business, medical, moving, and more.\n"
        "- Loan Application Type: Whether the loan is individual or joint. Possible values: 'Individual', 'Joint App'.\n"
        "- Interest Rate: The annual interest rate on the loan, ranging from 5.31% to 30.99% in the population we study.\n"
        "- Last Payment Amount: The most recent payment made, ranging from $0.0 to $41434.0 in the population we study.\n"
        "- Loan Amount: The total loan amount requested, ranging from $1000.0 to $40000.0 in the population we study.\n"
        "- Revolving Balance: The total revolving credit balance (e.g., credit cards), ranging from $0.0 to $669257.0 in the population we study.\n\n"
        "History Information:\n"
        "- Delinquency in 2 Years: Number of missed payments in the past 2 years, ranging from 0 to 20 in the population we study.\n"
        "- Inquiries in 6 Months: The number of credit inquiries in the last 6 months, ranging from 0 to 7 in the population we study.\n"
        "- Mortgage Accounts: The number of mortgage accounts held, ranging from 0 to 18 in the population we study.\n"
        "- Grade: The loan grade based on creditworthiness, ranging from 'A' (best) to 'G' (worst).\n"
        "- Open Accounts: The number of open credit accounts, ranging from 1 to 48 in the population we study.\n"
        "- Revolving Utilization Rate: The percentage of available credit being used, ranging from 0% to 177.7% in the population we study.\n"
        "- Total Accounts: The total number of credit accounts held, ranging from 2 to 95 in the population we study.\n"
        "- Fico Range Low: The lower bound of the borrower’s FICO score, ranging from 660 to 845 in the population we study.\n"
        "- Fico Range High: The upper bound of the borrower’s FICO score, ranging from 664 to 850 in the population we study.\n\n"
        "Soft Information:\n"
        "- Address State: The borrower’s state of residence. Possible values include all 50 US states (e.g., 'CA', 'NY', 'TX').\n"
        "- Employment Length: The number of years the borrower has been employed. Options range from '< 1 year' to '10+ years' in the population we study.\n"
        "- Home Ownership: The borrower’s homeownership status. Possible values: 'OWN', 'RENT', 'MORTGAGE', 'ANY', 'NONE', 'OTHER'.\n"
        "- Verification Status: Whether the borrower’s income is verified. Possible values: 'Source Verified', 'Not Verified', 'Verified'.\n"
        "- Annual Income: The borrower’s annual income, ranging from $0 to $2,000,000 in the population we study.\n\n"
    )
