import os
import numpy as np
import json

class TopicEvaluator():
    def __init__(self, client):
        self.client = client

    def extract_topics(self, document):

        tools = [
            {
                "type": "function",
                "name": "extract_topics",
                "description": "Extracts the topics from a company press release.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "The company press release document to extract topics from."
                        }
                    },
                    "required": ["document"]
                }
            }
        ]

        system_prompt = f"""
        # Instruction
        ## Goal
        ### You are an expert in extracting important topics from company press releases. Your goal will involve answering the questions below using the information provided.
        - **Definition**: You are given a definition of the trait that is being evaluated to help guide your extraction.
        - **Data**: Your input data includes a document.
        - **Tasks**: To complete your extraction you will be asked to evaluate the Data in different ways.
        """

        user_prompt = f"""
        user:
        # Definition
        **Materiality**: The degree to which a topic impacts the company's operations, financial performance, or strategic direction. A material topic is one that has a significant influence on the company's current or future state either positively or negatively.


        # Ratings
        ## [Materiality: 1] (Not Material)

        **Example:**
            **input**: "The company hosted a charity event to support local communities and raise funds for a nonprofit organization."
            **output**: {{"topic": "The company hosted a charity event to support local communities and raise funds for a nonprofit organization.", "materiality": 1, "reasoning": "While the charity event is a positive initiative, it does not have a significant impact on the company's operations, financial performance, or strategic direction. Therefore, it is considered not material."}}

        
        ## [Materiality: 2] (Low Materiality)

        **Example:**
            **input**: "The company announced a new employee wellness program to promote work-life balance and improve employee satisfaction."
            **output**: {{"topic": "The company announced a new employee wellness program to promote work-life balance and improve employee satisfaction.", "materiality": 2, "reasoning": "The employee wellness program is a positive initiative that may have a limited impact on the company's operations, financial performance, or strategic direction. Therefore, it is considered low materiality."}}

        ## [Materiality: 3] (Moderate Materiality)

        **Example:**
            **input**: "The company reported a 10 percent increase in quarterly revenue due to strong sales performance in its core product line."
            **output**: {{"topic": "The company reported a 10 percent increase in quarterly revenue due to strong sales performance in its core product line.", "materiality": 3, "reasoning": "The increase in quarterly revenue is a positive development that has a moderate impact on the company's operations and financial performance. Therefore, it is considered moderate materiality."}}

        ## [Materiality: 4] (High Materiality)

        **Example:**
            **input**: "The company announced a major strategic partnership with a leading technology firm to develop innovative solutions for the healthcare industry."
            **output**: {{"topic": "The company announced a major strategic partnership with a leading technology firm to develop innovative solutions for the healthcare industry.", "materiality": 4, "reasoning": "The strategic partnership has a significant impact on the company's operations and strategic direction, as it opens up new opportunities for growth and innovation in the healthcare industry. Therefore, it is considered high materiality."}}

        ## [Materiality: 5] (Very High Materiality)

        **Example:**
            **input**: "The company announced a major acquisition of a competitor, which is expected to significantly increase its market share and revenue."
            **output**: {{"topic": "The company announced a major acquisition of a competitor, which is expected to significantly increase its market share and revenue.", "materiality": 5, "reasoning": "The acquisition of a competitor has a very high impact on the company's operations, financial performance, and strategic direction, as it significantly increases market share and revenue potential. Therefore, it is considered very high materiality."}}
 
        # Data
        input: {document}

        # Tasks
        ## Please provide your assessment for the previous input based on the Definition above.
        Your output must be a valid JSON object with the following structure:
        {{
            "topics": [
                {{
                    "topic": "string",
                    "materiality": 1 | 2 | 3 | 4 | 5,
                    "reasoning": "string"
                }}
            ]
        }}

        # Output
        """

        input_list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response = self.client.responses.create(
            model="gpt-4o-mini",
            tools=tools,
            input=input_list,
        )

        print("Response:", response.output_text)