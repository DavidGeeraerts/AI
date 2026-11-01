# Execution Methodology: Plan-First Specification

## Core Directive
NEVER begin writing implementation code or modifying system files until a formal specification and task breakdown have been generated, reviewed, and explicitly approved by the user. 

## Phase 1: Discovery & Specification
When given a new feature request or complex problem, your first output must be a project specification. Create or update a `SPEC.md` file containing:
1. **Objective:** A concise statement of what is being built or fixed.
2. **Architecture & Approach:** The specific technologies, design patterns, data models, and file structures you plan to use or modify.
3. **Edge Cases & Constraints:** Potential pitfalls, security considerations, error handling strategies, and explicitly out-of-scope items.
4. **Dependencies:** Any new libraries, API keys, or environmental changes required.

## Phase 2: Task Breakdown (Tickets)
Once the specification is drafted, generate a sequential task list in a `TODO.md` file. 
* Break down the implementation into discrete, atomic steps (e.g., "1. Update database schema", "2. Create API endpoint", "3. Connect frontend UI").
* Each step must be independently testable or verifiable.
* Include the specific file paths that will be touched in each step.

## Phase 3: The Approval Gate (Grill-Me)
After outputting the Spec and the Task Breakdown, you must STOP.
* Present the proposed plan to the user.
* Ask the user: "Please review the proposed specification and task breakdown. Do you approve this plan, or are there edge cases I missed that we should adjust before coding?"
* Do not execute any build commands or write any application code until the user provides affirmative consent.

## Phase 4: Implementation & Verification
Once approved, execute the task list strictly in sequence:
1. **Implement:** Write the code for the current ticket only.
2. **Verify:** Run relevant tests, linters, or manual verification steps to ensure the step works exactly as specified before moving on.
3. **Update State:** Mark the completed step as `[x]` in `TODO.md`.
4. **Iterate:** Proceed to the next ticket until all tasks are complete.