import numpy as np
# Import necessary classes
from heat_eqn_handling_boundary_conditions import TwoDHeatEquationSolver

# def generate_question_data_for_llm(solution_array, case_number, question_id):
#     """
#     Generate prompt and response pairs for fine-tuning an LLM based on heat equation solutions.
    
#     Parameters:
#     - solution_array: 2D numpy array of temperature distribution
#     - case_number: Case number of the heat equation problem
#     - question_id: Specific question identifier
    
#     Returns:
#     - Dictionary containing 'prompt' and 'response' for the given question
#     """
#     if case_number == 1:
#         if question_id == 1:
#             # Question 1: Temperature at corner (0, 0)
#             prompt = "What is the temperature distribution at the corner (0, 0) of the unit square mesh?"
#             response = (
#                 f"The temperature at the corner (0, 0) is {solution_array[0, 0]:.2f}, based on the computed solution. "
#                 "This value is influenced by the boundary conditions, which ensure no heat flux at the edges of the unit square."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 2:
#             # Question 2: Temperature profile along x-axis at y = 0.5
#             prompt = "How does the temperature change with respect to the position along the x-axis at y = 0.5?"
#             response = (
#                 f"The temperature values along the x-axis at y = 0.5 are: "
#                 f"{', '.join(f'{solution_array[x, solution_array.shape[1] // 2]:.2f}' for x in range(solution_array.shape[0]))}. "
#                 "This demonstrates how the temperature varies horizontally at mid-height, as determined by the heat source and boundary conditions."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 3:
#             # Question 3: Effect of increasing coefficient of pi in the force function
#             prompt = "If we increase the coefficient of pi in the force function, what will happen?"
#             response = (
#                 "Increasing the coefficient of π in the force function amplifies the heat source intensity. "
#                 f"This leads to a proportional increase in temperature across the solution domain. For example, "
#                 f"the temperature at (0, 0) would increase from {solution_array[0, 0]:.2f} to "
#                 f"{solution_array[0, 0] * 2:.2f} if the coefficient is doubled, showing a direct scaling effect."
#             )
#             return {"prompt": prompt, "response": response}
            
#     elif case_number == 2:
#         if question_id == 4:
#             # Question 4: Why is temperature zero at x=0 and x=1?
#             prompt = "Explain why the temperature is zero at both x=0 and x=1, and what this means physically."
#             response = (
#                 "The temperature is zero at both x=0 and x=1 because the boundary conditions T(0,y) = 0 and T(1,y) = 0 "
#                 "are applied. This enforces that along the vertical edges of the domain, the temperature remains zero, "
#                 "indicating insulated or perfectly conducting boundaries. Physically, this represents a situation where "
#                 "no heat flows in or out at these boundaries, maintaining zero temperature throughout."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 5:
#             # Question 5: Coordinates of maximum temperature
#             prompt = "At what coordinates does the maximum temperature occur, and what determines this location?"
#             response = (
#                 f"The maximum temperature in the solution array is {solution_array.max():.2f}, and it occurs at the coordinate "
#                 f"{divmod(solution_array.argmax(), solution_array.shape[1])}. This location is determined by the boundary "
#                 "condition T(x,1) = sin(πx), which causes the temperature to peak near the top boundary where the sine function "
#                 "has its highest values, specifically at x=0.5."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 6:
#             # Question 6: Temperature profile along vertical lines x=0.5 and x=0.25
#             prompt = "How does the temperature profile change along the vertical line x=0.5 compared to x=0.25?"
#             response = (
#                 "The temperature profile along the vertical line x=0.5 is: "
#                 f"{', '.join(f'{solution_array[solution_array.shape[0] // 2, y]:.2f}' for y in range(solution_array.shape[1]))}. "
#                 "The profile along the vertical line x=0.25 is: "
#                 f"{', '.join(f'{solution_array[solution_array.shape[0] // 4, y]:.2f}' for y in range(solution_array.shape[1]))}. "
#                 "The temperature is generally higher along the line x=0.5 than along x=0.25 because the sine boundary condition "
#                 "T(x,1) = sin(πx) leads to a larger contribution to the temperature near x=0.5, where the sine function peaks."
#             )
#             return {"prompt": prompt, "response": response}

#     elif case_number == 3: 
#         if question_id == 7:
#             # Question 7: Temperature at the corner (0, 0)
#             prompt = "What is the temperature at the corner (0, 0) of the unit square mesh?"
#             response = (
#                 f"The temperature at the corner (0, 0) is {solution_array[0, 0]:.2f}, as computed from the solution array. "
#                 "This value is influenced by the boundary conditions applied at the edges of the domain, where the corner "
#                 "point has minimal exposure to the heat source and is constrained by the surrounding temperature distribution."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 8:
#             # Question 8: Physical significance of u(0,y) = 0
#             prompt = "What physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?"
#             response = (
#                 "The boundary condition u(0, y) = 0 at x = 0 implies that the temperature along the left edge of the unit square is fixed to zero. "
#                 "This condition represents an insulated or perfectly conducting boundary where no heat flows in or out. "
#                 "Physically, this suggests that the system maintains a constant temperature (zero in this case) along the left boundary, "
#                 "which influences the overall temperature distribution throughout the domain."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 9:
#             # Question 9: Physical significance of u(1,y) = y(1-y)
#             prompt = "What does the boundary condition u(1,y)=y(1-y) represent physically in this heat diffusion problem?"
#             response = (
#                 "The boundary condition u(1, y) = y(1 - y) at x = 1 defines a parabolic temperature distribution along the right edge of the domain. "
#                 "This condition represents a situation where the temperature varies quadratically with y, starting from 0 at y = 0, "
#                 "peaking at y = 0.5, and returning to 0 at y = 1. Physically, this can represent a heat source that varies with position along "
#                 "the right boundary, with the highest temperature in the middle and decreasing temperatures towards the top and bottom edges."
#             )
#             return {"prompt": prompt, "response": response}
#     elif case_number == 4: 
#         if question_id == 10:
#             # Question 10: Decay rate of temperature
#             prompt = "What can you infer about the decay rate of temperature?"
#             response = (
#                 "The decay rate of temperature is governed by the parameter α and the initial temperature distribution. "
#                 "In this case, the temperature decays exponentially with time due to the diffusion process described by the heat equation. "
#                 "Physically, the exponential decay ensures that the temperature distribution flattens out as the system approaches thermal equilibrium, "
#                 "with no heat flux at the boundaries."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 11:
#             # Question 11: Spatial pattern unchanged, amplitude decreases
#             prompt = "Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time."
#             response = (
#                 "The spatial pattern remains unchanged because the initial condition T(x,y,0) = sin(πx)sin(πy) represents the fundamental mode of the heat equation, "
#                 "and higher-order modes are absent in this setup. The amplitude decreases with time because the heat dissipates uniformly throughout the domain, "
#                 "causing the temperature values to reduce exponentially while preserving the shape of the initial temperature distribution."
#             )
#             return {"prompt": prompt, "response": response}
        
#         elif question_id == 12:
#             # Question 12: Effect of alpha on decay rate
#             prompt = "What is the effect of alpha on the decay rate of heat dissipation?"
#             response = (
#                 "The parameter α directly affects the decay rate of heat dissipation. A larger α results in faster diffusion, "
#                 "causing the temperature to decay more rapidly. Conversely, a smaller α slows down the diffusion process, "
#                 "leading to a slower decay of temperature over time. In this setup, with α = 0.01, the diffusion is relatively slow, "
#                 "resulting in gradual dissipation of heat as time progresses."
#             )
#             return {"prompt": prompt, "response": response}
    
#     return {"error": "Invalid case number or question id"}

def generate_question_data_for_llm(solution_array, case_number, question_id):
    """
    Generate prompt and response pairs for fine-tuning an LLM based on heat equation solutions.
    
    Parameters:
    - solution_array: 2D numpy array of temperature distribution
    - case_number: Case number of the heat equation problem
    - question_id: Specific question identifier
    
    Returns:
    - Dictionary containing 'prompt' and 'response' for the given question
    """
    if case_number == 1:
        # Case 1: Steady-state heat equation with specific force term
        expression = "∂²T/∂x² + ∂²T/∂y² = 8π²sin(2πx)sin(2πy)"
        boundary_conditions = (
            "Boundary Conditions:\n"
            "T(0,y) = 0\n"
            "T(1,y) = 0\n"
            "T(x,0) = 0\n"
            "T(x,1) = 0\n"
        )
        if question_id == 1:
            prompt = f"""Case1Q1: Analyze this steady-state heat equation solution
                        {expression}
                        {boundary_conditions}
                        What is the temperature distribution at the corner (0, 0) of the unit square mesh?"""
            # response = f"The temperature at the corner (0, 0) is {solution_array[0, 0]:.2f}, influenced by the boundary conditions."
            response = f"The temperature at the corner (0, 0) is solution_array[0, 0], influenced by the boundary conditions."
            return {"prompt": prompt, "response": response}
        
        elif question_id == 2:
            prompt = f"""Case1Q2: Analyze this steady-state heat equation solution
                        {expression}
                        {boundary_conditions}
                        How does the temperature change with respect to the position along the x-axis at y = 0.5?"""
            response = (
                "The temperature values along the x-axis at y = 0.5 are: "
                # f"{', '.join(f'{solution_array[x, solution_array.shape[1] // 2]:.2f}' for x in range(solution_array.shape[0]))}."
                "{solution_array[x, solution_array.shape[1] // 2]' for x in range(solution_array.shape[0]))}."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 3:
            prompt = f"""Case1Q3: Analyze this steady-state heat equation solution
                        {expression}
                        {boundary_conditions}
                        If we increase the coefficient of π in the force function, what will happen to the temperature distribution?"""
            response = (
                f"Increasing the coefficient of π will amplify the source term, proportionally raising the temperature. "
                # f"At (0, 0), the temperature might increase from {solution_array[0, 0]:.2f} to "
                # f"{solution_array[0, 0] * 2:.2f} if the coefficient doubles."
                "At (0, 0), the temperature might increase from solution_array[0, 0] to "
                "solution_array[0, 0] * 2 if the coefficient doubles."
            )
            return {"prompt": prompt, "response": response}

    elif case_number == 2:
        # Case 2: Steady-state heat equation with no source term
        expression = "∂²T/∂x² + ∂²T/∂y² = 0"
        boundary_conditions = (
            "Initial and Boundary Conditions:\n"
            "T(0,y) = 0\n"
            "T(1,y) = 0\n"
            "T(x,0) = 0\n"
            "T(x,1) = sin(πx)\n"
        )
        if question_id == 4:
            prompt = f"""Case2Q1: Analyze this steady-state heat equation solution
                        {expression}
                        {boundary_conditions}
                        Explain why the temperature is zero at both x=0 and x=1, and what this means physically."""
            response = (
                "The temperature is zero at both x=0 and x=1 because the boundary conditions T(0,y) = 0 and T(1,y) = 0 "
                "are applied. This enforces that along the vertical edges of the domain, the temperature remains zero, "
                "indicating insulated or perfectly conducting boundaries. Physically, this represents a situation where "
                "no heat flows in or out at these boundaries, maintaining zero temperature throughout."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 5:
            prompt = f"""Case2Q2: Analyze this steady-state heat equation solution
                        {expression}
                        {boundary_conditions}
                        At what coordinates does the maximum temperature occur, and what determines this location?"""
            response = (
                # f"The maximum temperature in the solution array is {solution_array.max():.2f}, and it occurs at the coordinate "
                # f"{divmod(solution_array.argmax(), solution_array.shape[1])}. This location is determined by the boundary "
                "The maximum temperature in the solution array is solution_array.max(), and it occurs at the coordinate "
                "{divmod(solution_array.argmax(), solution_array.shape[1])}. This location is determined by the boundary "
                "condition T(x,1) = sin(πx), which causes the temperature to peak near the top boundary where the sine function "
                "has its highest values, specifically at x=0.5."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 6:
            prompt = f"""Case2Q3: Analyze this steady-state heat equation solution
                        {expression}
                        {boundary_conditions}
                        How does the temperature profile change along the vertical line x=0.5 compared to x=0.25?"""
            response = (
                "The temperature profile along the vertical line x=0.5 is: "
                # f"{', '.join(f'{solution_array[solution_array.shape[0] // 2, y]:.2f}' for y in range(solution_array.shape[1]))}. "
                # "The profile along the vertical line x=0.25 is: "
                # f"{', '.join(f'{solution_array[solution_array.shape[0] // 4, y]:.2f}' for y in range(solution_array.shape[1]))}. "
                "{solution_array[solution_array.shape[0] // 2, y]' for y in range(solution_array.shape[1]))}. "
                "The profile along the vertical line x=0.25 is: "
                "{solution_array[solution_array.shape[0] // 4, y]' for y in range(solution_array.shape[1]))}. "
                "The temperature is generally higher along the line x=0.5 than along x=0.25 because the sine boundary condition "
                "T(x,1) = sin(πx) leads to a larger contribution to the temperature near x=0.5, where the sine function peaks."
            )
            return {"prompt": prompt, "response": response}

    elif case_number == 3:
        # Case 3: Transient heat equation with a sinusoidal source term
        expression = "∂²T/∂x² + ∂²T/∂y² = 0"
        boundary_conditions = (
            "Boundary Conditions:\n"
            "T(0,y) = 0\n"
            "T(1,y) = y(1-y)\n"
            "T(x,0) = 0\n"
            "T(x,1) = 0\n"
        )
        if question_id == 7:
            prompt = f"""Case3Q1: Analyze this transient heat equation solution
                        {expression}
                        {boundary_conditions}
                        What is the temperature at the corner (0, 0) of the unit square mesh?"""
            response = (
                # f"The temperature at the corner (0, 0) is {solution_array[0, 0]:.2f}, as computed from the solution array. "
                "The temperature at the corner (0, 0) is solution_array[0, 0], as computed from the solution array. "
                "This value is influenced by the boundary conditions applied at the edges of the domain, where the corner "
                "point has minimal exposure to the heat source and is constrained by the surrounding temperature distribution."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 8:
            prompt = f"""Case3Q2: Analyze this transient heat equation solution
                        {expression}
                        {boundary_conditions}
                        What physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?"""
            response = (
                "The boundary condition u(0, y) = 0 at x = 0 implies that the temperature along the left edge of the unit square is fixed to zero. "
                "This condition represents an insulated or perfectly conducting boundary where no heat flows in or out. "
                "Physically, this suggests that the system maintains a constant temperature (zero in this case) along the left boundary, "
                "which influences the overall temperature distribution throughout the domain."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 9:
            prompt = f"""Case3Q3: Analyze this transient heat equation solution
                        {expression}
                        {boundary_conditions}
                        What does the boundary condition u(1,y)=y(1-y) represent physically in this heat diffusion problem?"""
            response = (
                "The boundary condition u(1, y) = y(1 - y) at x = 1 defines a parabolic temperature distribution along the right edge of the domain. "
                "This condition represents a situation where the temperature varies quadratically with y, starting from 0 at y = 0, "
                "peaking at y = 0.5, and returning to 0 at y = 1. Physically, this can represent a heat source that varies with position along "
                "the right boundary, with the highest temperature in the middle and decreasing temperatures towards the top and bottom edges."
            )
            return {"prompt": prompt, "response": response}

    elif case_number == 4:
        # Case 4: Transient heat equation without a source term
        expression = "∂T/∂t = ∂²T/∂x² + ∂²T/∂y²"
        boundary_conditions = (
            "Boundary Conditions:\n"
            "T(x,y,0) = sin(πx)sin(πy)\n"
            "T(0,y,t) = 0\n"
            "T(1,y,t) = 0\n"
            "T(x,0,t) = 0\n"
            "T(x,1,t) = 0\n"
        )
        if question_id == 10:
            prompt = f"""Case4Q1: Analyze this transient heat equation solution
                        {expression}
                        {boundary_conditions}
                        What can you infer about the decay rate of temperature?"""
            response = (
                "The decay rate of temperature is governed by the parameter α and the initial temperature distribution. "
                "In this case, the temperature decays exponentially with time due to the diffusion process described by the heat equation. "
                "Physically, the exponential decay ensures that the temperature distribution flattens out as the system approaches thermal equilibrium, "
                "with no heat flux at the boundaries."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 11:
            prompt = f"""Case4Q2: Analyze this transient heat equation solution
                        {expression}
                        {boundary_conditions}
                        Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time?"""
            response = (
                "The spatial pattern remains unchanged because the initial condition T(x,y,0) = sin(πx)sin(πy) represents the fundamental mode of the heat equation, "
                "and higher-order modes are absent in this setup. The amplitude decreases with time because the heat dissipates uniformly throughout the domain, "
                "causing the temperature values to reduce exponentially while preserving the shape of the initial temperature distribution."
            )
            return {"prompt": prompt, "response": response}

        elif question_id == 12:
            prompt = f"""Case4Q3: Analyze this transient heat equation solution
                        {expression}
                        {boundary_conditions}
                        What is the effect of alpha on the deacy rate of heat dissipation?"""
            response = (
                "The parameter α directly affects the decay rate of heat dissipation. A larger α results in faster diffusion, "
                "causing the temperature to decay more rapidly. Conversely, a smaller α slows down the diffusion process, "
                "leading to a slower decay of temperature over time. In this setup, with α = 0.01, the diffusion is relatively slow, "
                "resulting in gradual dissipation of heat as time progresses."
            )
            return {"prompt": prompt, "response": response}

    return {"error": "Invalid case number or question id"}

def process_heat_equation_questions(solver, case):
    """
    Process questions for a specific heat equation case
    """
    if case == 1:
        solution = solver.solve_steady_state()
    elif case == 2:
        solution = solver.solve_steady_state()
    elif case == 3:
        solution = solver.solve_steady_state()
    elif case == 4:
        solution, _ = solver.solve_time_dependent(t_end=0.1, dt=0.01)
    
    results = {}
    questions_by_case = {
        1: [
            (1, "What is the temperature distribution at the corner (0, 0) of the unit square mesh?"),
            (2, "How does the temperature change with respect to the position along the x-axis at y = 0.5?"),
            (3, "If we increase the coefficient of pi in the force function what will happen?")
        ],
        2: [
            (4, "Explain why the temperature is zero at both x=0 and x=1, and what this means physically."),
            (5, "At what coordinates does the maximum temperature occur, and what determines this location?"),
            (6, "How does the temperature profile change along the vertical line x=0.5 compared to x=0.25?")
        ],
        3: [
            (7, "What is the temperature at the corner (0, 0) of the unit square mesh?"),
            (8, "What physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?"),
            (9, "What does the boundary condition u(1,y)=y(1-y) represent physically in this heat diffusion problem?")
        ],
        4: [
            (10, "What can you infer about the decay rate of temperature?"),
            (11, "Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time."),
            (12, "What is the effect of alpha on the decay rate of heat dissipation?")
        ]
    }


    for q_id, question_text in questions_by_case[case]:  # Iterate through question IDs and texts
        case_number = (q_id - 1) // 3 + 1  # Calculate the case number (1, 2, 3, or 4)
        if is_question_applicable(case_number, q_id):
            data = generate_question_data_for_llm(solution, case_number, q_id)
            if data:
                results[q_id] = {
                    "Case": case_number,
                    "Question": question_text,
                    "Data": data    #dict of keys: {prompt, response}
                }

    
    return results

def is_question_applicable(case, question_id):
    """
    Determine if a question is applicable for a specific case
    """
    # Define applicability rules based on case and question_id
    # This is a placeholder and should be filled with actual logic
    return True




# Create solver instances for different cases
solver_case1 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=1)
solver_case2 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=2)
solver_case3 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=3)
solver_case4 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=4, alpha=0.01)

# Process questions for each case
results_case1 = process_heat_equation_questions(solver_case1, case=1)
results_case2 = process_heat_equation_questions(solver_case2, case=2)
results_case3 = process_heat_equation_questions(solver_case3, case=3)
results_case4 = process_heat_equation_questions(solver_case4, case=4)
print('jai')

# results_case1.keys()
# dict_keys([1, 2, 3])
# results_case2.keys()
# dict_keys([4, 5, 6])
# results_case3.keys()
# dict_keys([7, 8, 9])
# results_case4.keys()
# dict_keys([10, 11, 12])

# results_case4[10]: {'Case': 4, 'Question': 'What can you infer about the decay rate of temperature?', 'Data': {'prompt': 'Case4Q1: Analyze this transient heat equation solution\n                        ∂T/∂t ...infer about the decay rate of temperature?', 'response': 'The decay rate of temperature is governed by the parameter α and the initial temperat...rium, with no heat flux at the boundaries.'}}

# Print or further process results
# Sample structure for all results
results = {
    "case_1": results_case1,
    "case_2": results_case2,
    "case_3": results_case3,
    "case_4": results_case4
}

# Prepare fine-tuning data
fine_tuning_data = []

for case, case_results in results.items():
    for q_id, content in case_results.items():
        entry = {
            "prompt": content['Data']['prompt'],
            "response": content['Data']['response']
        }
        fine_tuning_data.append(entry)

# Save the fine-tuning data to a text file
with open("fine_tuning_data.txt", "w") as f:
    for entry in fine_tuning_data:
        f.write(f"Prompt: {entry['prompt']}\n")
        f.write(f"Response: {entry['response']}\n\n")


print(f"Fine-tuning data prepared with {len(fine_tuning_data)} samples.")
