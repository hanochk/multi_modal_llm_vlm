prompt_obj_cls = '''I need you to help me classify expressions. I will give you an expression, and your task is to classify whether the expression represents an object category. Here are a few more specific rules:
    
1. If the expression describes an object with an attribute, it is invalid. For example, whereas "A bear" is valid, "A black bear" isn't.
2. If the expression describes a unique object rather than a category of objects in the abstract, it is invalid. For example, "street" is valid, but "Stanfield street" isn't.
3. Only concrete, visible objects are valid. "A beautiful day" or "Love" are invalid.
4. Given an expression, you first have to reason about whether it is valid or invalid, and only then give your final answer.

Expression: 2013
Reasoning: The year 2013 is not a concrete object that can be seen, therefor it is invalid.
Decision: INVALID.

Expression: airplane
Reasoning: An airplane is a concrete category of objects, and is therefor valid.
Decision: VALID

Expression: tall
Reasoning: tall is an adjective, and by itself it is not an object category, therefor invalid.
Decision: INVALID

Expression: yellow fruit
Reasoning: A fruit is an object, but yellow is a possible attribute of a fruit, so the expression yellow fruit is invalid.
Decision: INVALID

Expression: {} '''.format(expression)