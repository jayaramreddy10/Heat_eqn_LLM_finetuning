

import transformers
#Checking the version of transformers package
print(transformers.__version__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def load_model():
    device = "cuda:0"
    model_path = "modified_granite_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()
    return model, tokenizer



def get_model_response(model, tokenizer, prompt, max_tokens=300):
    input_text = f"Question: {prompt}\n\nAnswer:"
    input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **input_tokens,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean up the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(input_text):].strip()
    return response



def model_inference(cases):
    model, tokenizer = load_model()
    
    results = []
    
    for case, prompt in cases.items():
        print(f"\nTesting {case}...")
        try:
            response = get_model_response(model, tokenizer, prompt)
            results.append(response)
            print(f"\nResponse for {case}:")
            print(response)
        
        except Exception as e:
            print(f"Error in {case}: {str(e)}")
            results.append(f"Error: {str(e)}")
    
    return results

device = "cuda:0"
model_path = "ibm-granite/granite-3.1-2b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

# model.save_pretrained("modified_granite_model")
# tokenizer.save_pretrained("modified_granite_model")

folder_path ='finetuned_granite_model'
# Load the model
model = AutoModelForCausalLM.from_pretrained(folder_path)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(folder_path)
    
# cases = {
#     "Case1Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     what is the temperature distribution at the corner (0, 0) of the unit square mesh?""",
#     "Case1Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     how does the temperature change with respect to the position along the x-axis at y = 0.5?""",
#     "Case1Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     if we increase the coeeficient of pi in the force function what will happen?""",
#     "Case2Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     explain why the temperature is zero at both x=0 and x=1, and what this means physically.""",
#     "Case2Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     at what coordinates does the maximum temperature occur, and what determines this location?""",
#     "Case2Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     how does the temperature profile change along the vertical line x=0.5 compared to x=0.25?""",
#     "Case3Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     what is the temperature at the corner (0, 0) of the unit square mesh?""",
#     "Case3Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     what physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?""",
#     "Case3Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     what does the boundary condition u(1,y)=y(1−y) represent physically in this heat diffusion problem?""",
#     "Case4Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     what can you infer about the decay rate of temperature!""",
#     "Case4Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time@""",
#     "Case4Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
#     What is the effect of alpha on the deacy rate of heat dissipation#"""
# }

cases = {
    "Case1Q1": """Analyze this steady-state heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 8π²sin(2πx)sin(2πy)
                        Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = 0
                            T(x,0) = 0
                            T(x,1) = 0
            What is the temperature distribution at the corner (0, 0) of the unit square mesh?""",
    "Case1Q2": """Analyze this steady-state heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 8π²sin(2πx)sin(2πy)
                        Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = 0
                            T(x,0) = 0
                            T(x,1) = 0

                        How does the temperature change with respect to the position along the x-axis at y = 0.5?""",
    "Case1Q3": """Analyze this steady-state heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 8π²sin(2πx)sin(2πy)
                        Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = 0
                            T(x,0) = 0
                            T(x,1) = 0

                        If we increase the coefficient of π in the force function, what will happen to the temperature distribution?""",
    "Case2Q1": """Analyze this steady-state heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 0
                        Initial and Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = 0
                            T(x,0) = 0
                            T(x,1) = sin(πx)

                        Explain why the temperature is zero at both x=0 and x=1, and what this means physically.""",
    "Case2Q2": """Analyze this steady-state heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 0
                        Initial and Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = 0
                            T(x,0) = 0
                            T(x,1) = sin(πx)

                    At what coordinates does the maximum temperature occur, and what determines this location?""",
    "Case2Q3": """Analyze this steady-state heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 0
                        Initial and Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = 0
                            T(x,0) = 0
                            T(x,1) = sin(πx)

                How does the temperature profile change along the vertical line x=0.5 compared to x=0.25?""",
    "Case3Q1": """Analyze this transient heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 0
                        Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = y(1-y)
                            T(x,0) = 0
                            T(x,1) = 0

                        What is the temperature at the corner (0, 0) of the unit square mesh?""",
    "Case3Q2": """Analyze this transient heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 0
                        Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = y(1-y)
                            T(x,0) = 0
                            T(x,1) = 0

                        What physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?""",
    "Case3Q3": """Analyze this transient heat equation solution
                        ∂²T/∂x² + ∂²T/∂y² = 0
                        Boundary Conditions:
                            T(0,y) = 0
                            T(1,y) = y(1-y)
                            T(x,0) = 0
                            T(x,1) = 0

                        What does the boundary condition u(1,y)=y(1-y) represent physically in this heat diffusion problem?""",
    "Case4Q1": """Analyze this transient heat equation solution
                        ∂T/∂t = ∂²T/∂x² + ∂²T/∂y²
                        Boundary Conditions:
                            T(x,y,0) = sin(πx)sin(πy)
                            T(0,y,t) = 0
                            T(1,y,t) = 0
                            T(x,0,t) = 0
                            T(x,1,t) = 0

                        What can you infer about the decay rate of temperature?""",
    "Case4Q2": """Analyze this transient heat equation solution
                        ∂T/∂t = ∂²T/∂x² + ∂²T/∂y²
                        Boundary Conditions:
                            T(x,y,0) = sin(πx)sin(πy)
                            T(0,y,t) = 0
                            T(1,y,t) = 0
                            T(x,0,t) = 0
                            T(x,1,t) = 0

                        Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time?""",
    "Case4Q3": """Analyze this transient heat equation solution
                        ∂T/∂t = ∂²T/∂x² + ∂²T/∂y²
                        Boundary Conditions:
                            T(x,y,0) = sin(πx)sin(πy)
                            T(0,y,t) = 0
                            T(1,y,t) = 0
                            T(x,0,t) = 0
                            T(x,1,t) = 0

                        What is the effect of alpha on the deacy rate of heat dissipation?"""
}

results = model_inference(cases)
df = pd.DataFrame({"Id" : list(range(1,13)), "Answer": results})
df.to_csv('submission2.csv',index = False)










