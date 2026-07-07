# ML vs Heuristic Router Comparison

The Decision Engine labels in `training_dataset.csv` are treated as the offline ground truth for this comparison.

## Summary Metrics

- Heuristic Router accuracy vs Decision Engine: 0.3203
- ML Router accuracy vs Decision Engine: 0.8516
- Heuristic and ML agreement: 0.2219
- Heuristic average prediction latency: 1.219870 ms
- ML average prediction latency: 6.794597 ms
- Heuristic average confidence: 0.5813
- ML average confidence: 0.7975

## Strengths

- Heuristic Router: transparent, deterministic, and independent of training data drift.
- Decision Engine: uses offline benchmark outcomes to create utility-aware labels.
- ML Router: learns the Decision Engine policy from pre-routing features and can capture nonlinear feature interactions.

## Weaknesses

- Heuristic Router: hand-tuned thresholds may not match the validated Decision Engine labels.
- Decision Engine: depends on post-inference benchmark signals and cannot run before routing in production.
- ML Router: only generalizes as well as the Phase 2 dataset covers real traffic.

## Heuristic Misclassifications

- `coding_001`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a Python function that takes a list of integers and returns the sum of all even numbers.
- `coding_002`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a SQL query to select all rows from a table named 'employees' where the department is 'Engineering'.
- `coding_003`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a Python function that reverses a string without using the built-in reverse method.
- `coding_004`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a JavaScript function that checks whether a given number is prime.
- `coding_005`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a Python function that counts the number of vowels in a given string.
- `coding_006`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a SQL query that returns the total number of orders placed by each customer, ordered by order count descending.
- `coding_007`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a Python function that returns the factorial of a non-negative integer using a loop.
- `coding_009`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a Python class that implements a stack data structure using a list, with push, pop, peek, and is_empty methods.
- `coding_010`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write a SQL query that finds all customers who placed more than three orders in the last 90 days, returning their name and order count.
- `coding_011`: actual `REMOTE`, predicted `LOCAL`. Prompt: Debug the following Python function that is meant to flatten a nested list but raises a RecursionError for deeply nested inputs. Provide a fix using an iterative approach.

## ML Misclassifications

- `coding_021`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a Python function to check if a dictionary contains a specific key and return its value, defaulting to None if missing.
- `coding_027`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a TypeScript interface definition for a User object containing id, username, email, active status, and optional profile fields.
- `coding_036`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a Python function to check if a word is a palindrome, ignoring capitalization and non-alphanumeric characters.
- `coding_079`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a C++ class implementing a lock-free Compare-and-Swap (CAS) based queue. Define structure for node pointers, atomic operations, enqueue/dequeue methods, and handle memory allocation safety under
- `coding_080`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a Python script that parses a huge JSON dump (10GB) using stream processing techniques (like ijson or generator pipelines) to count frequency of specific nested key values, minimizing memory foo
- `creative_writing_043`: actual `LOCAL`, predicted `REMOTE`. Prompt: Compose a sonnet (14 lines, ABAB CDCD EFEF GG rhyme scheme) exploring the theme of memory decay and digital archiving. The language should be rich and atmospheric.
- `creative_writing_047`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a story chapter describing a character's journey through a dense forest that changes its layout based on the traveler's emotional state. Focus on psychological depth and nature metaphors.
- `creative_writing_050`: actual `LOCAL`, predicted `REMOTE`. Prompt: Draft a creative exchange of letters between two rival astronomers in the 18th century, each claiming to have discovered the same comet first. Maintain historical syntax.
- `creative_writing_055`: actual `LOCAL`, predicted `REMOTE`. Prompt: Compose a poem exploring the physical structures and architectural shapes of an abandoned industrial steel mill being reclaimed by nature.
- `creative_writing_056`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write a story from the perspective of a cartographer mapping the shoreline of a coastline that shifts with each high tide, making measurement impossible.

## Recommendation

Use the ML router as a shadow-mode replacement candidate. It is trained only on pre-routing features, while the Decision Engine remains the offline labeling and audit mechanism.
