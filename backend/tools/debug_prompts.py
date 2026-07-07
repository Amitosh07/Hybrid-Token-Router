import sys
sys.path.insert(0, '.')
from app.services.feature_extractor import extract_features
from app.services.router import route

prompts = [
    ('summarization_001', 'easy', 'Summarise the following sentence in 5 words or fewer: The quantum mechanics of subatomic particles is an extremely complex field of physics that challenges our intuitions about reality.'),
    ('summarization_003', 'easy', 'Write a TL;DR (one to two sentences) for the following: Retrieval-augmented generation (RAG) is a technique that combines large language models with external knowledge bases to improve factual accuracy.'),
    ('planning_001', 'easy', 'Create a simple daily schedule for a remote worker who wants to exercise in the morning, work from 9am to 5pm, and have a dedicated reading hour in the evening.'),
    ('coding_015', 'hard', 'Design a Python class that implements a thread-safe producer-consumer queue using threading.Condition, supporting multiple producers and consumers without deadlock.'),
    ('reasoning_020', 'hard', 'Three suspects A, B, and C are questioned about a robbery. A says: I did not do it. B says: A is lying. C says: B is lying. If exactly one of the three is telling the truth, determine who committed the robbery and verify your answer is consistent with all three statements.'),
]

for pid, diff, prompt in prompts:
    f = extract_features(prompt, debug=True)
    r = route(f)
    dbg = f['_debug']
    cs = f['category_scores']
    print(f'=== {pid} [{diff}] ===')
    print(f'  complexity_score : {f["complexity_score"]}')
    print(f'  technical        : {cs["technical"]}')
    print(f'  reasoning        : {cs["reasoning"]}')
    print(f'  task             : {cs["task"]}')
    print(f'  constraint       : {cs["constraint"]}')
    print(f'  context          : {cs["context"]}')
    print(f'  routing_score    : {r["routing_score"]}')
    print(f'  provider         : {r["provider"]}')
    tsk = dbg['stage_4c_task']
    print(f'  task_group       : {tsk["matched_group"]} base={tsk["base_score"]} mods={tsk["modifier_bonus"]}')
    print(f'  task_modifiers   : {tsk["fired_modifiers"]}')
    print(f'  tech_domains     : {dbg["stage_4a_technical"]["domain_contributions"]}')
    print(f'  reasoning_groups : {dbg["stage_4b_reasoning"]["group_contributions"]}')
    print(f'  constraints      : {dbg["stage_4d_constraint"]["fired_patterns"]}')
    print()
