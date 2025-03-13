from enum import Enum

# NOTE the instructions written here are model-specific and sensitive to model changes
# the three instructiosn are tested specifically on Qwen2.5-7B-Instruct and shown to be able to elicit
# proper CoT rationales. In case of model changes, please test the instructions again and adapt.

STEP_TAG = "\n\n"
SPLIT_TOKEN = "Final assessment"
KS_TOKEN = "Default risk"

# NOTE change the search pattern accordingly when the SPLIT_TOKEN changes
# also this search pattern is not pefect, it does not capture "\n\n**Final Assessment:** good"
# namely when ** is between "good" and "assessment", but this is a flaw I am not smart enough to balance
SEARCH_PATTERN = r"\s*\*?\*?Final\s*[Aa]ssessment\*?\*?\s*:\s*(good|bad)\*?\*?"
SEARCH_PATTERN_KS = r"\s*\*?\*?Default\s*[Rr]isk\*?\*?\s*:\s*((?:[1-9]|[1-9][0-9]|100))\b\*?\*?"
# Search for the combined pattern
SEARCH_PATTERN_RL_FORMAT = r"Final assessment: (good|bad)\s*\nDefault risk: ([1-9]|[1-9][0-9]|100)\s*$"

# possible end tokens when inferencing with Qwen2.5-7B
POSSIBLE_END_TOKENS = ['\n   \n', '\n  \n', '\n \n', '\n\n', '.\n\n', ':\n\n', ' \n\n']

class Prompts(Enum):
    # System Prompts
    SYSTEM_PROMPT_CREDIT_SCORING = (
        "You are a risk management assistant who is good at credit scoring."
    )
    
    SYSTEM_PROMPT_R1_FORMAT = (	
        "You are a risk management assistant who is good at credit scoring and are doing risk assessment."
        "You should answer the query by first breaking down the problem and analyze it in your mind and then provides "
        "the stuctured reasoning process and the final answer. The internal thinking process in your mind is enclosed within <think>\n </think>"
        "i.e., <think>\n internal thinking process here </think> structured reasoning steps, holistic assessment and final answer here"
    )
    
    INTRO_CUSTOMER_CREDIT_REPORT = "Here is the customer's credit report:\n"

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
    )
    
    INSTRUCTION_STEP_BY_STEP_R1 = (
        "You are given a text describing a customer's credit report and predicted loan status (probabilities) "
        "from a trained machine learning system as reference (reference only, could be wrong). Explanations for "
        "features of customers mentioned in the credit report are provided beforehand\n\n"
        "Please analyze all information STEP BY STEP following these instructions:\n"
        "1. **During your internal thinking process**: \n"
        "    - Go through each feature in the credit report, evaluate its significance to the customer's creditworthiness. "
        "Then use BOTH of your prior knowledge and explanations of feature meanings given to you to interpret the data.\n"
        "    - After understanding all features, do a thorough cross-referencing, i.e., cross-refer to all related features, "
        "think thoroughly about how these interrelated features can make combined impact.\n" 
        "    - During cross-referencing, you are encourage to interpret combined feature using sophisticated measure, such as calculating "
        "some percentage or rate, for example the Debt-to-Income (DTI) Ratio. Also keep in mind that some features are more relevant than others e.g., soft information is "
        "probably less significant compared to the loan or history information in most cases.\n"
        "    - When you have doubts about your conclusion, repeat above process, jump back and forth through provided information "
        "to seek evidence to validate your answer, give your educated fianl answer util you are confident.\n"
        "2. **During your external reasoning step**: give structed reasoning steps based on your internal thinking, and then, "
        "give a holistic assessment stating the most significant evidences that lead you to your conclusion. In the end, give your final answer. \n"
        "3. **In the beginning of your output during internal thinking**: state how you would analyse this problem to show that you have fully "
        "understand the requirements given above for how to reason.\n"
        "4. **In the end of your output during external reasoning**: after finishing the reasoning steps, first give a combined holistic assessment resulted "
        f"from your reasoning, then provided the final answer in the following format: '{SPLIT_TOKEN}: [choose from \"good\" or \"bad\"]'. "
        "End your output with one of the assessment token, this means that your last output token can only be either "
        "\"good\" or \"bad\", not any other texts or symbols. "
        f"For example: don't use **{SPLIT_TOKEN}: good**, instead just say {SPLIT_TOKEN}: good (without '**').\n\n"
    )
    
    INSTRUCTION_STEP_BY_STEP_R1_KS = (
        "You are given a text describing a customer's credit report and predicted loan status (probabilities) "
        "from a trained machine learning system as reference (reference only, could be wrong). Explanations for "
        "features of customers mentioned in the credit report are provided beforehand\n\n"
        "Please analyze all information STEP BY STEP following these instructions:\n"
        "1. **During your internal thinking process**: \n"
        "    - Go through each feature in the credit report, evaluate its significance to the customer's creditworthiness. "
        "Then use BOTH of your prior knowledge and explanations of feature meanings given to you to interpret the data.\n"
        "    - After understanding all features, do a thorough cross-referencing, i.e., cross-refer to all related features, "
        "think thoroughly about how these interrelated features can make combined impact.\n" 
        "    - During cross-referencing, you are encourage to interpret combined feature using sophisticated measure, such as calculating "
        "some percentage or rate, for example the Debt-to-Income (DTI) Ratio. Also keep in mind that some features are more relevant than others e.g., soft information is "
        "probably less significant compared to the loan or history information in most cases.\n"
        "    - When you have doubts about your conclusion, repeat above process, jump back and forth through provided information "
        "to seek evidence to validate your answer, give your educated final answer until you are confident.\n"
        "2. **During your external reasoning step**: give structured reasoning steps based on your internal thinking, and then, "
        "give a holistic assessment stating the most significant evidences that lead you to your conclusion. In the end, give your final answer. \n"
        "3. **In the beginning of your output during internal thinking**: state how you would analyse this problem to show that you have fully "
        "understand the requirements given above for how to reason.\n"
        "4. **In the end of your output during external reasoning**: after finishing the reasoning steps, first give a combined holistic assessment resulted "
        "from your reasoning as your argument of how you arrived at your final answer. Then give your final answer and end your output. \n"
        "5. **For your final answer**: please provide TWO pieces of information in the following format as your final answer:\n"
        f"'{SPLIT_TOKEN}: [choose from \"good\" or \"bad\"]'\n '{KS_TOKEN}: [number between 1-100]'\n"
        "The default risk score should reflect your confidence in the loan outcome, where 1 means virtually no chance of default (definitely good) "
        "and 100 means certain default (definitely bad). Scores between 30-70 indicate moderate uncertainty.\n"
        "Ensure your risk score aligns with your binary classification - 'good' ratings should generally have scores below 50, "
        "while 'bad' ratings should have scores above 50.\n\n"
    )
    
    EXPLANATION_FEATURES = (
        "Here are the explanations of all features in a customer's credit report you need to consider:\n"
        
        "Loan Information:\n"
        "- Installment: Monthly payment required for the loan, ranging from $30.74 to $1466.04 in the population we study. "
        "For a given customer, higher installments relative to income may strain repayment capacity\n"
        "- Loan Purpose: The reason for taking the loan, with options such as debt consolidation (refers to taking out a new loan "
        "or credit card to pay off other existing loans or credit cards, effectively combining multiple debts into a single one, typically "
        "with a lower interest rate, but may also indicate prior financial stress), home improvement, small business, medical, moving, and more.\n"
        "- Loan Application Type: Whether the loan is individual or joint (applied for by two or more individuals together). Possible values: 'Individual', 'Joint App'.\n"
        "- Interest Rate: The annual interest rate on the loan, ranging from 5.31% to 30.99% in the population we study.\n"
        "- Last Payment Amount: The most recent payment made, ranging from $0.0 to $41434.0 in the population we study. "
        "For a given customer, a $0 payment could indicate delinquency, while very high payments might signal one-time settlements\n"
        "- Loan Amount: The total loan amount requested, ranging from $1000.0 to $40000.0 in the population we study.\n"
        "- Revolving Balance: The amount of money that remains unpaid from the previous billing period, which gets carried over to the next "
        "billing cycle, ranging from $0.0 to $669257.0 in the population we study.\n"
        
        "History Information:\n"
        "- Delinquency in 2 Years: Number of missed payments in the past 2 years, ranging from 0 to 20 in the population we study. "
        "Note that in practice, even one delinquency can have significant impact on creditworthiness\n"
        "- Inquiries in 6 Months: The number of credit inquiries in the last 6 months, refecting how actively the customer is seeking "
        "credit, ranging from 0 to 7 in the population we study.\n"
        "- Mortgage Accounts: The number of mortgage accounts held, reflecting how many active mortgage loans the customer is handling. "
        "Mortgage accounts in good standing may improve credit mix, but having too many mortgage loans with balances can negatively affect "
        "creditworthiness. The feature ranges from 0 to 18 in the population we study.\n"
        "- Grade: The loan grade based on creditworthiness, refecting risk associated with the loan. It raneges from 'A' (best) to 'G' (worst) with "
        "risk being low to high.\n"
        "- Open Accounts: The number of open credit accounts, managing too many open accounts may indicate potential overextension or difficulty "
        "in managing debt. But similar as Mortgage Accounts, please differentiate between 'too many' (overextension) and 'diverse mix' (positive) "
        "This feature ranges from 1 to 48 in the population we study.\n"
        "- Revolving Utilization Rate: The percentage of available credit being used, ranging from 0% to 177.7% in the population we study. "
        "Keeping this rate low can positively impact the credit score, ususally utilization >30% harms credit scores, and >100% implies over-limit borrowing\n"
        "- Total Accounts: The total number of credit accounts held, having a diverse mix of credit accounts can positively impact the credit score "
        "by showing responsible credit management, but having too many accounts—especially if they carry high balances—can be seen as a potential risk, "
        "indicating overextension or difficulty in managing debt effectively. The feature ranges from 2 to 95 in the population we study.\n"
        "- Fico Range Low: The lower bound of the borrower’s FICO score, ranging from 660 to 845 in the population we study.\n"
        "- Fico Range High: The upper bound of the borrower’s FICO score, ranging from 664 to 850 in the population we study.\n"
        
        "Soft Information:\n"
        "- Address State: The borrower’s state of residence. Possible values include all 50 US states (e.g., 'CA', 'NY', 'TX').\n"
        "- Employment Length: The number of years the borrower has been employed. Options range from '< 1 year' to '10+ years' in the population we study.\n"
        "- Home Ownership: The borrower’s homeownership status. Possible values: 'OWN', 'RENT', 'MORTGAGE', 'ANY', 'NONE', 'OTHER'.\n"
        "- Verification Status: Whether the borrower’s income is verified. Possible values: 'Source Verified', 'Not Verified', 'Verified'.\n"
        "- Annual Income: The borrower’s annual income, ranging from $0 to $2,000,000 in the population we study.\n\n"
    )
