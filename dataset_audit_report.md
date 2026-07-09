# Merged Dataset Audit Report
**Role**: Senior ML Engineer  
**Date**: July 9, 2026  
**Conversation ID**: 18afc6e4-41f8-422c-a989-13476094e741

---

## Executive Summary

This report presents a comprehensive dataset audit of the merged routing dataset containing **15,000 prompts** (LOCAL: **5,850** (39.0%), REMOTE: **9,150** (61.0%)).

This audit analyzes the category distribution, parent category representation, prompt length buckets, predictors of routing, category imbalance, dataset quality, and training readiness.

> [!WARNING]
> **Audit Conclusion**: The dataset is **NOT suitable for production training** in its current state. The 61% REMOTE distribution is heavily distorted by artificial template suffixes in the imported dataset, domain keyword bias in the project data, a narrow prompt length distribution (all prompts are between 24 and 80 words), and rule-based false positives in the feature extractor.

---

## 1. Category Distribution

The dataset contains a mix of existing project data and an imported dataset. Below is the detailed category-by-category breakdown of the 15,000 prompts.

### Mathematics
Total: 708
LOCAL: 2
REMOTE: 706
REMOTE Rate: 99.72%
Average Length: 58.10
Average Complexity: 0.4799
Average Reasoning Score: 4.71

### Coding
Total: 698
LOCAL: 132
REMOTE: 566
REMOTE Rate: 81.09%
Average Length: 53.26
Average Complexity: 0.3806
Average Reasoning Score: 3.77

### Creative Writing
Total: 698
LOCAL: 185
REMOTE: 513
REMOTE Rate: 73.50%
Average Length: 54.09
Average Complexity: 0.4002
Average Reasoning Score: 4.07

### Planning
Total: 698
LOCAL: 7
REMOTE: 691
REMOTE Rate: 99.00%
Average Length: 54.16
Average Complexity: 0.5449
Average Reasoning Score: 5.57

### Translation
Total: 698
LOCAL: 2
REMOTE: 696
REMOTE Rate: 99.71%
Average Length: 55.13
Average Complexity: 0.4181
Average Reasoning Score: 4.22

### Reasoning
Total: 697
LOCAL: 0
REMOTE: 697
REMOTE Rate: 100.00%
Average Length: 50.37
Average Complexity: 0.4790
Average Reasoning Score: 4.76

### Summarization
Total: 688
LOCAL: 128
REMOTE: 560
REMOTE Rate: 81.40%
Average Length: 53.35
Average Complexity: 0.3949
Average Reasoning Score: 3.97

### General
Total: 135
LOCAL: 6
REMOTE: 129
REMOTE Rate: 95.56%
Average Length: 50.27
Average Complexity: 0.4178
Average Reasoning Score: 4.21

### C Programming
Total: 20
LOCAL: 1
REMOTE: 19
REMOTE Rate: 95.00%
Average Length: 33.45
Average Complexity: 0.4659
Average Reasoning Score: 4.80

### Economic History
Total: 20
LOCAL: 16
REMOTE: 4
REMOTE Rate: 20.00%
Average Length: 38.35
Average Complexity: 0.3246
Average Reasoning Score: 3.10

### Urban Sociology
Total: 20
LOCAL: 6
REMOTE: 14
REMOTE Rate: 70.00%
Average Length: 38.40
Average Complexity: 0.3393
Average Reasoning Score: 3.20

### 2D Animation
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.30
Average Complexity: 0.3588
Average Reasoning Score: 3.60

### 3D Animation
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.90
Average Complexity: 0.3407
Average Reasoning Score: 3.30

### 3D Bioprinting
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3427
Average Reasoning Score: 3.30

### 5G Communication
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.60
Average Complexity: 0.4158
Average Reasoning Score: 4.20

### Academic Integrity
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.90
Average Complexity: 0.3302
Average Reasoning Score: 3.30

### Accounting
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 31.90
Average Complexity: 0.3715
Average Reasoning Score: 3.70

### Acting For Film
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.40
Average Complexity: 0.3568
Average Reasoning Score: 3.50

### Action Figures
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.90
Average Complexity: 0.3450
Average Reasoning Score: 3.20

### Active Listening
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.80
Average Complexity: 0.3334
Average Reasoning Score: 3.20

### Adaptation Strategies
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.30
Average Complexity: 0.3602
Average Reasoning Score: 3.60

### Adaptive Reuse
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.10
Average Complexity: 0.3400
Average Reasoning Score: 3.20

### Adoption General
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 37.90
Average Complexity: 0.3781
Average Reasoning Score: 3.90

### Adult Education
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.80
Average Complexity: 0.3410
Average Reasoning Score: 3.20

### Adventure Tourism
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 37.30
Average Complexity: 0.3194
Average Reasoning Score: 3.00

### Aero Design Competitions
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 39.40
Average Complexity: 0.3630
Average Reasoning Score: 3.60

### Aerospace Engineering
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.50
Average Complexity: 0.3324
Average Reasoning Score: 3.20

### Aerospace Technology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3323
Average Reasoning Score: 3.20

### Aesthetics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 32.70
Average Complexity: 0.3558
Average Reasoning Score: 3.50

### Affiliate Marketing
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.90
Average Complexity: 0.3460
Average Reasoning Score: 3.40

### Agile Methodologies
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.30
Average Complexity: 0.3494
Average Reasoning Score: 3.40

### Agriculture
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.30
Average Complexity: 0.3512
Average Reasoning Score: 3.60

### Air Traffic Control
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 39.50
Average Complexity: 0.3444
Average Reasoning Score: 3.50

### Aircraft Maintenance
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.00
Average Complexity: 0.3455
Average Reasoning Score: 3.40

### Airport Management
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 38.80
Average Complexity: 0.3942
Average Reasoning Score: 4.10

### Algebra
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.20
Average Complexity: 0.3176
Average Reasoning Score: 3.00

### Algorithms
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.80
Average Complexity: 0.5341
Average Reasoning Score: 5.50

### Alumni Networks
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 38.00
Average Complexity: 0.3193
Average Reasoning Score: 3.00

### Ancient Civilizations
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.70
Average Complexity: 0.3211
Average Reasoning Score: 3.00

### Animal Behavior
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.20
Average Complexity: 0.3263
Average Reasoning Score: 3.10

### Animal Ethics Secular
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 39.20
Average Complexity: 0.3452
Average Reasoning Score: 3.30

### Animal Nutrition
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.70
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Animal Physiology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.60
Average Complexity: 0.3314
Average Reasoning Score: 3.20

### Animal Rescue
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.60
Average Complexity: 0.3412
Average Reasoning Score: 3.40

### Animal Shelters
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.70
Average Complexity: 0.3576
Average Reasoning Score: 3.70

### Animals General
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 30.00
Average Complexity: 0.3421
Average Reasoning Score: 3.30

### Animation
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.80
Average Complexity: 0.3401
Average Reasoning Score: 3.30

### Anime Fandom
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.30
Average Complexity: 0.3414
Average Reasoning Score: 3.20

### Anthropology General
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.60
Average Complexity: 0.3171
Average Reasoning Score: 3.10

### Anti Inflammatory Diets
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.60
Average Complexity: 0.3225
Average Reasoning Score: 3.10

### Antique Furniture
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.40
Average Complexity: 0.3321
Average Reasoning Score: 3.20

### Api Design
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.80
Average Complexity: 0.5183
Average Reasoning Score: 5.10

### Aquaculture
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.40
Average Complexity: 0.3844
Average Reasoning Score: 3.90

### Arboriculture
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.40
Average Complexity: 0.3625
Average Reasoning Score: 3.50

### Archaeobotany
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.30
Average Complexity: 0.3164
Average Reasoning Score: 3.00

### Archaeological Field Techniques
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.60
Average Complexity: 0.3222
Average Reasoning Score: 3.10

### Archaeological Preservation
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.80
Average Complexity: 0.3417
Average Reasoning Score: 3.30

### Archaeology General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.60
Average Complexity: 0.3379
Average Reasoning Score: 3.30

### Archaeology Methods
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.80
Average Complexity: 0.3266
Average Reasoning Score: 3.10

### Archaeozoology
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.60
Average Complexity: 0.3206
Average Reasoning Score: 3.00

### Architectural Engineering
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 38.00
Average Complexity: 0.3541
Average Reasoning Score: 3.50

### Architecture General
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.80
Average Complexity: 0.3322
Average Reasoning Score: 3.10

### Archival Science
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.90
Average Complexity: 0.3158
Average Reasoning Score: 3.00

### Aromatherapy
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 33.50
Average Complexity: 0.3187
Average Reasoning Score: 3.00

### Art History General
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.80
Average Complexity: 0.3328
Average Reasoning Score: 3.20

### Artifact Curation
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.70
Average Complexity: 0.3367
Average Reasoning Score: 3.30

### Artificial Intelligence
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.60
Average Complexity: 0.4685
Average Reasoning Score: 4.60

### Artificial Languages
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 34.00
Average Complexity: 0.3520
Average Reasoning Score: 3.60

### Artisanal Bread
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 32.50
Average Complexity: 0.3248
Average Reasoning Score: 3.10

### Arts And Crafts For Relaxation
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 38.40
Average Complexity: 0.3635
Average Reasoning Score: 3.70

### Arts General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.30
Average Complexity: 0.3471
Average Reasoning Score: 3.40

### Assertiveness
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.60
Average Complexity: 0.3351
Average Reasoning Score: 3.20

### Assessment Methods
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.30
Average Complexity: 0.3985
Average Reasoning Score: 3.90

### Astrobiology
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.20
Average Complexity: 0.3406
Average Reasoning Score: 3.30

### Astronaut Training
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 39.10
Average Complexity: 0.3363
Average Reasoning Score: 3.30

### Astronomy
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.10
Average Complexity: 0.3189
Average Reasoning Score: 3.00

### Atmospheric Sciences
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.90
Average Complexity: 0.3342
Average Reasoning Score: 3.30

### Audiobook Production
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.90
Average Complexity: 0.3447
Average Reasoning Score: 3.30

### Augmented Reality
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.50
Average Complexity: 0.3481
Average Reasoning Score: 3.40

### Author Branding
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.60
Average Complexity: 0.3235
Average Reasoning Score: 3.00

### Autobiographies
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.00
Average Complexity: 0.3535
Average Reasoning Score: 3.50

### Automation
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 31.00
Average Complexity: 0.3925
Average Reasoning Score: 4.00

### Automl
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.50
Average Complexity: 0.4464
Average Reasoning Score: 4.60

### Automotive Design
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 36.20
Average Complexity: 0.3694
Average Reasoning Score: 3.60

### Aviation Civil
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.70
Average Complexity: 0.3676
Average Reasoning Score: 3.60

### Aviation General
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.00
Average Complexity: 0.3516
Average Reasoning Score: 3.50

### Aviation History
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.40
Average Complexity: 0.3429
Average Reasoning Score: 3.50

### Ayurvedic Principles Non Religious
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.70
Average Complexity: 0.3264
Average Reasoning Score: 3.10

### Bakery And Pastry Arts
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.70
Average Complexity: 0.3330
Average Reasoning Score: 3.40

### Baking Arts
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.50
Average Complexity: 0.3223
Average Reasoning Score: 3.10

### Ballet
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.00
Average Complexity: 0.3705
Average Reasoning Score: 3.80

### Baroque Art
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.40
Average Complexity: 0.3420
Average Reasoning Score: 3.40

### Bartending Mixology
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 32.70
Average Complexity: 0.3185
Average Reasoning Score: 3.00

### Behavioral Economics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 35.60
Average Complexity: 0.3462
Average Reasoning Score: 3.40

### Behavioral Psychology
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.60
Average Complexity: 0.3499
Average Reasoning Score: 3.50

### Beverage Crafting
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.30
Average Complexity: 0.3204
Average Reasoning Score: 3.00

### Bibliometrics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.50
Average Complexity: 0.3547
Average Reasoning Score: 3.70

### Big Data
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.20
Average Complexity: 0.3907
Average Reasoning Score: 4.00

### Big Data Tools
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 38.00
Average Complexity: 0.4009
Average Reasoning Score: 4.00

### Bilingualism
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.80
Average Complexity: 0.3184
Average Reasoning Score: 3.10

### Bioarchaeology
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.20
Average Complexity: 0.3146
Average Reasoning Score: 3.10

### Biodiversity
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.10
Average Complexity: 0.3740
Average Reasoning Score: 3.60

### Bioethics Secular
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.00
Average Complexity: 0.4026
Average Reasoning Score: 4.00

### Biofeedback
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.80
Average Complexity: 0.3450
Average Reasoning Score: 3.40

### Bioinformatics
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 32.60
Average Complexity: 0.3742
Average Reasoning Score: 3.60

### Biological Anthropology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.50
Average Complexity: 0.3308
Average Reasoning Score: 3.20

### Biology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 30.20
Average Complexity: 0.3381
Average Reasoning Score: 3.30

### Biomedical Engineering
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.50
Average Complexity: 0.3390
Average Reasoning Score: 3.50

### Birdwatching
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.60
Average Complexity: 0.3214
Average Reasoning Score: 3.10

### Black Holes
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.10
Average Complexity: 0.3361
Average Reasoning Score: 3.30

### Blockchain
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 31.70
Average Complexity: 0.3761
Average Reasoning Score: 3.80

### Blues
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.70
Average Complexity: 0.3377
Average Reasoning Score: 3.30

### Board Games
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3220
Average Reasoning Score: 3.10

### Board Governance
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.20
Average Complexity: 0.3774
Average Reasoning Score: 3.80

### Body Language
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.20
Average Complexity: 0.3348
Average Reasoning Score: 3.10

### Book Cover Design
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 39.30
Average Complexity: 0.3494
Average Reasoning Score: 3.60

### Book Marketing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.70
Average Complexity: 0.3204
Average Reasoning Score: 3.00

### Botany
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.30
Average Complexity: 0.3331
Average Reasoning Score: 3.20

### Brain Boosting Foods
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 32.50
Average Complexity: 0.3172
Average Reasoning Score: 3.00

### Brain Training Apps
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.30
Average Complexity: 0.3293
Average Reasoning Score: 3.20

### Braising
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.00
Average Complexity: 0.3271
Average Reasoning Score: 3.30

### Brand Strategy
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.90
Average Complexity: 0.3277
Average Reasoning Score: 3.10

### Branding
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.50
Average Complexity: 0.3394
Average Reasoning Score: 3.30

### Breathing Techniques
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.80
Average Complexity: 0.3233
Average Reasoning Score: 3.10

### Broadcasting
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.50
Average Complexity: 0.3352
Average Reasoning Score: 3.30

### Budgeting And Saving
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.10
Average Complexity: 0.3468
Average Reasoning Score: 3.40

### Bullet Journaling
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.90
Average Complexity: 0.3047
Average Reasoning Score: 3.00

### Business Continuity
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 39.50
Average Complexity: 0.4901
Average Reasoning Score: 4.90

### Business Etiquette
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.60
Average Complexity: 0.3344
Average Reasoning Score: 3.20

### Business Growth
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.70
Average Complexity: 0.3377
Average Reasoning Score: 3.30

### Business Incubation
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.80
Average Complexity: 0.3331
Average Reasoning Score: 3.20

### Business Innovation
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.40
Average Complexity: 0.3375
Average Reasoning Score: 3.30

### Business Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.90
Average Complexity: 0.3659
Average Reasoning Score: 3.70

### Business Negotiation
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.00
Average Complexity: 0.3379
Average Reasoning Score: 3.30

### Business Strategy
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.80
Average Complexity: 0.3503
Average Reasoning Score: 3.40

### Cad Design
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.70
Average Complexity: 0.3234
Average Reasoning Score: 3.00

### Cake Decorating
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.60
Average Complexity: 0.3206
Average Reasoning Score: 3.00

### Cake Sculpting
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.60
Average Complexity: 0.3205
Average Reasoning Score: 3.00

### Calculus
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.30
Average Complexity: 0.3356
Average Reasoning Score: 3.10

### Calligraphy
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 31.00
Average Complexity: 0.3181
Average Reasoning Score: 3.00

### Calligraphy Advanced
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.00
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Camping
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.60
Average Complexity: 0.3220
Average Reasoning Score: 3.10

### Canning And Preserving
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.30
Average Complexity: 0.3223
Average Reasoning Score: 3.10

### Carbon Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.70
Average Complexity: 0.3590
Average Reasoning Score: 3.50

### Carbon Sequestration
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.20
Average Complexity: 0.3401
Average Reasoning Score: 3.30

### Career Development
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.90
Average Complexity: 0.3514
Average Reasoning Score: 3.40

### Cartography
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 33.00
Average Complexity: 0.3185
Average Reasoning Score: 3.00

### Cat Behavior
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.70
Average Complexity: 0.3356
Average Reasoning Score: 3.30

### Cataloging And Classification
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.30
Average Complexity: 0.3270
Average Reasoning Score: 3.10

### Ceramics
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 31.00
Average Complexity: 0.3460
Average Reasoning Score: 3.40

### Change Management
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 38.50
Average Complexity: 0.3956
Average Reasoning Score: 3.90

### Character Design
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.70
Average Complexity: 0.3573
Average Reasoning Score: 3.60

### Chemical Engineering
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.30
Average Complexity: 0.3680
Average Reasoning Score: 3.70

### Chemistry
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.00
Average Complexity: 0.3388
Average Reasoning Score: 3.20

### Child Development Stages
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.80
Average Complexity: 0.3643
Average Reasoning Score: 3.80

### Child Nutrition
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.70
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Children S Activities
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.90
Average Complexity: 0.3440
Average Reasoning Score: 3.40

### Children S Literature
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 38.20
Average Complexity: 0.3676
Average Reasoning Score: 3.80

### Children S Music
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.20
Average Complexity: 0.3468
Average Reasoning Score: 3.50

### Chinese Cuisine
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.10
Average Complexity: 0.3225
Average Reasoning Score: 3.10

### Chocolate Making
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.70
Average Complexity: 0.3166
Average Reasoning Score: 3.00

### Chronic Illness Management
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.10
Average Complexity: 0.3510
Average Reasoning Score: 3.50

### Circular Economy
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.90
Average Complexity: 0.4073
Average Reasoning Score: 4.10

### Circus Arts
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 33.70
Average Complexity: 0.3670
Average Reasoning Score: 3.80

### Circus Performances
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.10
Average Complexity: 0.3847
Average Reasoning Score: 3.90

### City Infrastructure
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.70
Average Complexity: 0.3457
Average Reasoning Score: 3.40

### Civil Engineering
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.80
Average Complexity: 0.3508
Average Reasoning Score: 3.40

### Classic Cars
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.80
Average Complexity: 0.3268
Average Reasoning Score: 3.20

### Classical Music
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.50
Average Complexity: 0.3616
Average Reasoning Score: 3.70

### Classroom Management
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.80
Average Complexity: 0.3459
Average Reasoning Score: 3.40

### Clean Energy Advocacy
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 39.20
Average Complexity: 0.3456
Average Reasoning Score: 3.40

### Climate Change Science
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.50
Average Complexity: 0.3593
Average Reasoning Score: 3.50

### Climate Modeling
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 39.70
Average Complexity: 0.3586
Average Reasoning Score: 3.50

### Climate Science Advanced
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 40.40
Average Complexity: 0.3607
Average Reasoning Score: 3.70

### Climatology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.20
Average Complexity: 0.3426
Average Reasoning Score: 3.30

### Clinical Psychology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.40
Average Complexity: 0.3291
Average Reasoning Score: 3.30

### Cloud Computing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.10
Average Complexity: 0.4947
Average Reasoning Score: 5.00

### Cloud Computing Advanced
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.00
Average Complexity: 0.5457
Average Reasoning Score: 5.30

### Cloud Cost Optimization
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 39.00
Average Complexity: 0.5266
Average Reasoning Score: 5.20

### Cloud Data Warehousing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.80
Average Complexity: 0.4410
Average Reasoning Score: 4.50

### Cloud Migration
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.60
Average Complexity: 0.5952
Average Reasoning Score: 5.90

### Cloud Security
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.30
Average Complexity: 0.5276
Average Reasoning Score: 5.30

### Coastal Geography
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 37.60
Average Complexity: 0.3405
Average Reasoning Score: 3.30

### Cognitive Exercises
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.80
Average Complexity: 0.3300
Average Reasoning Score: 3.20

### Cognitive Psychology
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.60
Average Complexity: 0.3895
Average Reasoning Score: 3.80

### Cognitive Science
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 34.30
Average Complexity: 0.4000
Average Reasoning Score: 4.10

### Coin Collecting
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.90
Average Complexity: 0.3224
Average Reasoning Score: 3.20

### Collecting
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.20
Average Complexity: 0.3327
Average Reasoning Score: 3.30

### Collecting Advanced
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.40
Average Complexity: 0.3547
Average Reasoning Score: 3.50

### Coloring Therapy
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3311
Average Reasoning Score: 3.30

### Combinatorics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.20
Average Complexity: 0.4012
Average Reasoning Score: 4.00

### Comedy
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 30.30
Average Complexity: 0.3807
Average Reasoning Score: 3.80

### Comic Book Collecting
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 40.60
Average Complexity: 0.3355
Average Reasoning Score: 3.20

### Comic Book Collecting Detailed
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 41.00
Average Complexity: 0.3405
Average Reasoning Score: 3.20

### Comic Book Writing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.10
Average Complexity: 0.3815
Average Reasoning Score: 4.00

### Comic Conventions
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 34.50
Average Complexity: 0.3384
Average Reasoning Score: 3.50

### Comic Illustration
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.80
Average Complexity: 0.3660
Average Reasoning Score: 3.70

### Comics And Manga
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 36.30
Average Complexity: 0.3717
Average Reasoning Score: 3.80

### Communication General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.60
Average Complexity: 0.3399
Average Reasoning Score: 3.20

### Communication Skills
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.20
Average Complexity: 0.3308
Average Reasoning Score: 3.00

### Community Building
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.40
Average Complexity: 0.3288
Average Reasoning Score: 3.20

### Community Engagement
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.40
Average Complexity: 0.3253
Average Reasoning Score: 3.10

### Community Gardens
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.60
Average Complexity: 0.3297
Average Reasoning Score: 3.20

### Community Outreach
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.50
Average Complexity: 0.3442
Average Reasoning Score: 3.40

### Compensation Benefits
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.40
Average Complexity: 0.3733
Average Reasoning Score: 3.80

### Competitive Analysis
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.40
Average Complexity: 0.3557
Average Reasoning Score: 3.50

### Competitive Memorizing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.70
Average Complexity: 0.3295
Average Reasoning Score: 3.20

### Compiler Design
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.90
Average Complexity: 0.5302
Average Reasoning Score: 5.40

### Composting
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.50
Average Complexity: 0.3361
Average Reasoning Score: 3.20

### Computational Linguistics
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.50
Average Complexity: 0.4127
Average Reasoning Score: 4.20

### Computer Networks
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.80
Average Complexity: 0.4019
Average Reasoning Score: 4.10

### Computer Science General
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.70
Average Complexity: 0.4675
Average Reasoning Score: 4.80

### Computer Vision
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.10
Average Complexity: 0.4162
Average Reasoning Score: 4.20

### Concert Organization
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.40
Average Complexity: 0.3399
Average Reasoning Score: 3.40

### Confectionery
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.60
Average Complexity: 0.3221
Average Reasoning Score: 3.10

### Conference Planning
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.30
Average Complexity: 0.3349
Average Reasoning Score: 3.30

### Confidence Boosting
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.50
Average Complexity: 0.3288
Average Reasoning Score: 3.10

### Confidence Building
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.90
Average Complexity: 0.3248
Average Reasoning Score: 3.00

### Conflict Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.70
Average Complexity: 0.3389
Average Reasoning Score: 3.30

### Conflict Resolution
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.80
Average Complexity: 0.3327
Average Reasoning Score: 3.10

### Conflict Resolution Advanced
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.40
Average Complexity: 0.3292
Average Reasoning Score: 3.20

### Conservation Biology
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.00
Average Complexity: 0.3583
Average Reasoning Score: 3.50

### Conservation Techniques
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 36.70
Average Complexity: 0.3421
Average Reasoning Score: 3.40

### Consulting
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.50
Average Complexity: 0.3782
Average Reasoning Score: 3.90

### Consulting Freelancing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.90
Average Complexity: 0.3893
Average Reasoning Score: 4.00

### Consumer Electronics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.70
Average Complexity: 0.3572
Average Reasoning Score: 3.50

### Containerization
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 29.70
Average Complexity: 0.4959
Average Reasoning Score: 5.10

### Contemporary Art Movements
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.60
Average Complexity: 0.3302
Average Reasoning Score: 3.20

### Contemporary Dance
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.90
Average Complexity: 0.3620
Average Reasoning Score: 3.60

### Content Marketing
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.60
Average Complexity: 0.3378
Average Reasoning Score: 3.30

### Cooking Techniques
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.80
Average Complexity: 0.3181
Average Reasoning Score: 3.00

### Corporate Ethics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.00
Average Complexity: 0.3445
Average Reasoning Score: 3.30

### Corporate Events
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.90
Average Complexity: 0.3225
Average Reasoning Score: 3.10

### Corporate Social Responsibility
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 41.30
Average Complexity: 0.3598
Average Reasoning Score: 3.50

### Cosmic Microwave Background
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 37.70
Average Complexity: 0.3156
Average Reasoning Score: 3.00

### Cosmology Non Religious
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.60
Average Complexity: 0.3185
Average Reasoning Score: 3.00

### Costume Design
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.50
Average Complexity: 0.3276
Average Reasoning Score: 3.00

### Crafts General
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 31.30
Average Complexity: 0.3250
Average Reasoning Score: 3.10

### Criminology
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.50
Average Complexity: 0.3908
Average Reasoning Score: 3.90

### Critical Thinking
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.50
Average Complexity: 0.3400
Average Reasoning Score: 3.40

### Cross Cultural Communication
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.70
Average Complexity: 0.3365
Average Reasoning Score: 3.20

### Cross Cultural Etiquette
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.30
Average Complexity: 0.3216
Average Reasoning Score: 3.00

### Crowdfunding
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.70
Average Complexity: 0.3490
Average Reasoning Score: 3.50

### Cryptocurrency Non Political
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.40
Average Complexity: 0.3575
Average Reasoning Score: 3.60

### Cryptography
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.10
Average Complexity: 0.4960
Average Reasoning Score: 5.20

### Cuisine Styles
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.70
Average Complexity: 0.3183
Average Reasoning Score: 3.00

### Culinary Arts
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3197
Average Reasoning Score: 3.00

### Cultural Anthropology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.70
Average Complexity: 0.3212
Average Reasoning Score: 3.20

### Cultural Anthropology Non Religious
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.20
Average Complexity: 0.3313
Average Reasoning Score: 3.20

### Cultural Clothing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.20
Average Complexity: 0.3189
Average Reasoning Score: 3.00

### Cultural Cuisine
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.20
Average Complexity: 0.3185
Average Reasoning Score: 3.00

### Cultural Etiquette
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.30
Average Complexity: 0.3265
Average Reasoning Score: 3.00

### Cultural History
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.00
Average Complexity: 0.3200
Average Reasoning Score: 3.10

### Cultural Tourism
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.90
Average Complexity: 0.3217
Average Reasoning Score: 3.20

### Culture And Society
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.70
Average Complexity: 0.3345
Average Reasoning Score: 3.30

### Curriculum Development
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.30
Average Complexity: 0.3703
Average Reasoning Score: 3.60

### Customer Service
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.40
Average Complexity: 0.3353
Average Reasoning Score: 3.30

### Cyber Physical Systems
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.20
Average Complexity: 0.3543
Average Reasoning Score: 3.60

### Cybersecurity
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.70
Average Complexity: 0.4750
Average Reasoning Score: 4.90

### Cybersecurity Advanced
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.10
Average Complexity: 0.4785
Average Reasoning Score: 4.90

### Dark Energy
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.40
Average Complexity: 0.3193
Average Reasoning Score: 3.00

### Dark Matter
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.80
Average Complexity: 0.3200
Average Reasoning Score: 3.00

### Data Analysis
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.50
Average Complexity: 0.3824
Average Reasoning Score: 4.00

### Data Governance
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.10
Average Complexity: 0.3586
Average Reasoning Score: 3.60

### Data Science General
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.50
Average Complexity: 0.3897
Average Reasoning Score: 3.90

### Data Structures
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.30
Average Complexity: 0.4266
Average Reasoning Score: 4.40

### Data Visualization
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.60
Average Complexity: 0.3448
Average Reasoning Score: 3.60

### Data Wrangling
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.70
Average Complexity: 0.3677
Average Reasoning Score: 3.60

### Database Systems
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 31.90
Average Complexity: 0.4800
Average Reasoning Score: 4.70

### Debt Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.30
Average Complexity: 0.3367
Average Reasoning Score: 3.30

### Decision Making
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.40
Average Complexity: 0.3648
Average Reasoning Score: 3.50

### Decluttering
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.50
Average Complexity: 0.3294
Average Reasoning Score: 3.10

### Decoupage
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 32.90
Average Complexity: 0.3186
Average Reasoning Score: 3.00

### Deep Learning
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.00
Average Complexity: 0.5026
Average Reasoning Score: 5.10

### Demography
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.20
Average Complexity: 0.3229
Average Reasoning Score: 3.10

### Descriptive Analytics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.20
Average Complexity: 0.3394
Average Reasoning Score: 3.30

### Design General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.20
Average Complexity: 0.3248
Average Reasoning Score: 3.10

### Design Thinking
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 34.90
Average Complexity: 0.3491
Average Reasoning Score: 3.40

### Dessert Plating
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.40
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Desserts And Pastries
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.20
Average Complexity: 0.3181
Average Reasoning Score: 3.00

### Development Economics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.30
Average Complexity: 0.3434
Average Reasoning Score: 3.40

### Developmental Psychology
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.00
Average Complexity: 0.3502
Average Reasoning Score: 3.70

### Devops
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 30.50
Average Complexity: 0.5277
Average Reasoning Score: 5.20

### Differentiated Instruction
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 39.50
Average Complexity: 0.3928
Average Reasoning Score: 3.80

### Digital Branding
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.90
Average Complexity: 0.3588
Average Reasoning Score: 3.50

### Digital Communication
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.00
Average Complexity: 0.3347
Average Reasoning Score: 3.20

### Digital Detox
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.10
Average Complexity: 0.3249
Average Reasoning Score: 3.20

### Digital Entertainment Platforms
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.80
Average Complexity: 0.3280
Average Reasoning Score: 3.10

### Digital Ethics
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.00
Average Complexity: 0.3777
Average Reasoning Score: 3.70

### Digital Exhibits
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.50
Average Complexity: 0.3470
Average Reasoning Score: 3.70

### Digital Libraries
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.00
Average Complexity: 0.3424
Average Reasoning Score: 3.30

### Digital Marketing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.00
Average Complexity: 0.3329
Average Reasoning Score: 3.20

### Digital Nomad Lifestyle
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.60
Average Complexity: 0.3229
Average Reasoning Score: 3.10

### Digital Planning
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.20
Average Complexity: 0.3282
Average Reasoning Score: 3.20

### Digital Twins
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.00
Average Complexity: 0.3550
Average Reasoning Score: 3.40

### Dining Etiquette
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.90
Average Complexity: 0.3191
Average Reasoning Score: 3.00

### Disaster Recovery
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.00
Average Complexity: 0.5698
Average Reasoning Score: 5.80

### Disaster Resilience
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 38.70
Average Complexity: 0.4345
Average Reasoning Score: 4.50

### Discord Communities
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.50
Average Complexity: 0.3430
Average Reasoning Score: 3.40

### Distributed Systems
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.70
Average Complexity: 0.5240
Average Reasoning Score: 5.30

### Diy Projects
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.70
Average Complexity: 0.3211
Average Reasoning Score: 3.10

### Documentary Filmmaking
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.80
Average Complexity: 0.3616
Average Reasoning Score: 3.60

### Dog Training
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.60
Average Complexity: 0.3245
Average Reasoning Score: 3.10

### Dollmaking
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 31.60
Average Complexity: 0.3181
Average Reasoning Score: 3.00

### Donor Relations
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.20
Average Complexity: 0.3385
Average Reasoning Score: 3.30

### Doujinshi Non Erotic
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.00
Average Complexity: 0.3680
Average Reasoning Score: 4.00

### Dramatic Literature
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.50
Average Complexity: 0.3573
Average Reasoning Score: 3.30

### Drones
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.40
Average Complexity: 0.3266
Average Reasoning Score: 3.00

### E Commerce
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.60
Average Complexity: 0.3224
Average Reasoning Score: 3.00

### Early Childhood Education
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 41.60
Average Complexity: 0.3458
Average Reasoning Score: 3.50

### Early Literacy
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.90
Average Complexity: 0.3342
Average Reasoning Score: 3.40

### Earth Science
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.60
Average Complexity: 0.3232
Average Reasoning Score: 3.10

### Earthquake Engineering
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 39.10
Average Complexity: 0.4030
Average Reasoning Score: 4.00

### Ebook Formatting
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.30
Average Complexity: 0.3247
Average Reasoning Score: 3.10

### Eco Fashion
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.30
Average Complexity: 0.3329
Average Reasoning Score: 3.20

### Eco Friendly Living
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.00
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Eco Friendly Products
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.30
Average Complexity: 0.3367
Average Reasoning Score: 3.20

### Eco Tourism
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.00
Average Complexity: 0.3166
Average Reasoning Score: 3.10

### Ecology
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 32.80
Average Complexity: 0.3424
Average Reasoning Score: 3.30

### Ecology Advanced
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.70
Average Complexity: 0.3468
Average Reasoning Score: 3.40

### Economics General
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.10
Average Complexity: 0.3267
Average Reasoning Score: 3.20

### Ecosystem Services
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.10
Average Complexity: 0.3498
Average Reasoning Score: 3.40

### Edge Ai
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.80
Average Complexity: 0.3889
Average Reasoning Score: 4.00

### Editing And Proofreading
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.20
Average Complexity: 0.3306
Average Reasoning Score: 3.20

### Edtech
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.50
Average Complexity: 0.3509
Average Reasoning Score: 3.40

### Education General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.40
Average Complexity: 0.3490
Average Reasoning Score: 3.40

### Educational Games
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.50
Average Complexity: 0.3309
Average Reasoning Score: 3.30

### Educational Psychology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.70
Average Complexity: 0.3486
Average Reasoning Score: 3.40

### Elder Care
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.90
Average Complexity: 0.3573
Average Reasoning Score: 3.60

### Electric Vehicles
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.10
Average Complexity: 0.3716
Average Reasoning Score: 3.70

### Electrical Engineering
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.50
Average Complexity: 0.3583
Average Reasoning Score: 3.60

### Electronic Music
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 32.60
Average Complexity: 0.3393
Average Reasoning Score: 3.60

### Email Marketing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.80
Average Complexity: 0.3264
Average Reasoning Score: 3.10

### Embroidery
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 32.00
Average Complexity: 0.3687
Average Reasoning Score: 3.70

### Emerging Tech
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.00
Average Complexity: 0.4226
Average Reasoning Score: 4.30

### Emotional Intelligence
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.10
Average Complexity: 0.3261
Average Reasoning Score: 3.00

### Emotional Resilience
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.90
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Employee Engagement
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.10
Average Complexity: 0.3190
Average Reasoning Score: 3.00

### Employee Retention
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.50
Average Complexity: 0.3379
Average Reasoning Score: 3.20

### English Language
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.30
Average Complexity: 0.3386
Average Reasoning Score: 3.40

### Entertainment General
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.20
Average Complexity: 0.3506
Average Reasoning Score: 3.50

### Entomology
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.50
Average Complexity: 0.3435
Average Reasoning Score: 3.40

### Entrepreneurship
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.00
Average Complexity: 0.3455
Average Reasoning Score: 3.30

### Environmental Economics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 39.00
Average Complexity: 0.3574
Average Reasoning Score: 3.60

### Environmental Engineering
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.20
Average Complexity: 0.3992
Average Reasoning Score: 3.90

### Environmental Ethics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.20
Average Complexity: 0.3613
Average Reasoning Score: 3.60

### Environmental Initiatives
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.10
Average Complexity: 0.3367
Average Reasoning Score: 3.30

### Epistemology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.50
Average Complexity: 0.3283
Average Reasoning Score: 3.20

### Esports
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.90
Average Complexity: 0.3475
Average Reasoning Score: 3.30

### Estate Planning
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.10
Average Complexity: 0.3502
Average Reasoning Score: 3.40

### Ethical Dilemmas Secular
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.00
Average Complexity: 0.4363
Average Reasoning Score: 4.30

### Ethical Hacking
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.60
Average Complexity: 0.5429
Average Reasoning Score: 5.30

### Ethics Non Religious
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.70
Average Complexity: 0.3612
Average Reasoning Score: 3.50

### Ethnography
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.10
Average Complexity: 0.3119
Average Reasoning Score: 3.00

### Ethnomusicology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3383
Average Reasoning Score: 3.30

### Etiquette For Children
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.10
Average Complexity: 0.3236
Average Reasoning Score: 3.10

### Etiquette For Seniors
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.50
Average Complexity: 0.3373
Average Reasoning Score: 3.20

### Etiquette General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.40
Average Complexity: 0.3328
Average Reasoning Score: 3.20

### Etl Processes
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.30
Average Complexity: 0.4083
Average Reasoning Score: 4.10

### Event Management General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.30
Average Complexity: 0.3290
Average Reasoning Score: 3.30

### Event Planning
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.10
Average Complexity: 0.3273
Average Reasoning Score: 3.20

### Exam Techniques
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.80
Average Complexity: 0.3398
Average Reasoning Score: 3.30

### Exhibit Design
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.90
Average Complexity: 0.3291
Average Reasoning Score: 3.30

### Exhibition Design
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.10
Average Complexity: 0.3414
Average Reasoning Score: 3.40

### Existentialism Secular
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.60
Average Complexity: 0.4041
Average Reasoning Score: 4.20

### Exoplanets
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.70
Average Complexity: 0.3189
Average Reasoning Score: 3.00

### Exotic Pets
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.60
Average Complexity: 0.3212
Average Reasoning Score: 3.10

### Experiential Learning
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.10
Average Complexity: 0.3439
Average Reasoning Score: 3.30

### Experimental Archaeology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.00
Average Complexity: 0.3410
Average Reasoning Score: 3.30

### Explainable Ai
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.90
Average Complexity: 0.3397
Average Reasoning Score: 3.30

### Expo Management
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.80
Average Complexity: 0.3269
Average Reasoning Score: 3.30

### Extended Reality
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.10
Average Complexity: 0.3203
Average Reasoning Score: 3.00

### Extreme Sports
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.60
Average Complexity: 0.3204
Average Reasoning Score: 3.00

### Facebook Groups
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 37.50
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Family Budgeting
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.40
Average Complexity: 0.3202
Average Reasoning Score: 3.10

### Family Dynamics
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 38.00
Average Complexity: 0.3206
Average Reasoning Score: 3.10

### Fantasy
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 31.20
Average Complexity: 0.3612
Average Reasoning Score: 3.70

### Fashion Design
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.70
Average Complexity: 0.3445
Average Reasoning Score: 3.40

### Fashion Entrepreneurship
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.30
Average Complexity: 0.3313
Average Reasoning Score: 3.20

### Fashion General
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.30
Average Complexity: 0.3212
Average Reasoning Score: 3.10

### Fashion History
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 39.50
Average Complexity: 0.3203
Average Reasoning Score: 3.10

### Fashion Journalism
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.40
Average Complexity: 0.3333
Average Reasoning Score: 3.30

### Fashion Marketing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.70
Average Complexity: 0.3334
Average Reasoning Score: 3.20

### Fashion Styling
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.60
Average Complexity: 0.3238
Average Reasoning Score: 3.00

### Fermentation
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.60
Average Complexity: 0.3320
Average Reasoning Score: 3.20

### Festival Planning
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.20
Average Complexity: 0.3525
Average Reasoning Score: 3.50

### Festivals And Celebrations Non Religious
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.10
Average Complexity: 0.3442
Average Reasoning Score: 3.30

### Film Production
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3374
Average Reasoning Score: 3.40

### Finance
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.60
Average Complexity: 0.4152
Average Reasoning Score: 4.20

### Financial Analysis
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 38.20
Average Complexity: 0.3757
Average Reasoning Score: 3.70

### Financial Literacy For Teens
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.00
Average Complexity: 0.3295
Average Reasoning Score: 3.20

### Fishing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.10
Average Complexity: 0.3255
Average Reasoning Score: 3.20

### Fitness Competitions
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.80
Average Complexity: 0.3433
Average Reasoning Score: 3.50

### Fiverr Strategies
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.50
Average Complexity: 0.3387
Average Reasoning Score: 3.20

### Flight Safety
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 39.60
Average Complexity: 0.3766
Average Reasoning Score: 3.70

### Flipped Classroom
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.40
Average Complexity: 0.3638
Average Reasoning Score: 3.50

### Folk Music
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.70
Average Complexity: 0.3282
Average Reasoning Score: 3.20

### Folk Tales
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.20
Average Complexity: 0.3576
Average Reasoning Score: 3.70

### Folklore Studies
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.50
Average Complexity: 0.3365
Average Reasoning Score: 3.50

### Food And Beverage Service
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.40
Average Complexity: 0.3238
Average Reasoning Score: 3.10

### Food General
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 31.30
Average Complexity: 0.3178
Average Reasoning Score: 3.00

### Food Science
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.80
Average Complexity: 0.3610
Average Reasoning Score: 3.60

### Forensic Psychology
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 36.00
Average Complexity: 0.3784
Average Reasoning Score: 3.80

### Forest Bathing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.50
Average Complexity: 0.3243
Average Reasoning Score: 3.10

### Forestry
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.00
Average Complexity: 0.3668
Average Reasoning Score: 3.60

### Forum Moderation
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.70
Average Complexity: 0.3559
Average Reasoning Score: 3.50

### Franchising
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.10
Average Complexity: 0.3349
Average Reasoning Score: 3.40

### Freelance Writing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.70
Average Complexity: 0.3417
Average Reasoning Score: 3.20

### Freelancing
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.30
Average Complexity: 0.3360
Average Reasoning Score: 3.30

### Freelancing General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.40
Average Complexity: 0.3365
Average Reasoning Score: 3.30

### French Cuisine
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.00
Average Complexity: 0.3175
Average Reasoning Score: 3.00

### French Language
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.90
Average Complexity: 0.3396
Average Reasoning Score: 3.30

### Fundraising
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.60
Average Complexity: 0.3310
Average Reasoning Score: 3.30

### Fusion Cuisine
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.40
Average Complexity: 0.3230
Average Reasoning Score: 3.10

### Galaxy Formation
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.80
Average Complexity: 0.3357
Average Reasoning Score: 3.40

### Game Design
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.00
Average Complexity: 0.3503
Average Reasoning Score: 3.40

### Game Development
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.40
Average Complexity: 0.3601
Average Reasoning Score: 3.60

### Game Streaming
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.50
Average Complexity: 0.3330
Average Reasoning Score: 3.20

### Game Theory
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.70
Average Complexity: 0.3488
Average Reasoning Score: 3.40

### Game Theory In Economics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 40.30
Average Complexity: 0.3213
Average Reasoning Score: 3.10

### Gamification
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.80
Average Complexity: 0.3511
Average Reasoning Score: 3.50

### Gaming General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.00
Average Complexity: 0.3412
Average Reasoning Score: 3.30

### Gardening
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.60
Average Complexity: 0.3259
Average Reasoning Score: 3.20

### General Knowledge
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.00
Average Complexity: 0.3346
Average Reasoning Score: 3.20

### Genetics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.60
Average Complexity: 0.3310
Average Reasoning Score: 3.30

### Gentrification Socioeconomic
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.00
Average Complexity: 0.3169
Average Reasoning Score: 3.10

### Geoarchaeology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.90
Average Complexity: 0.3451
Average Reasoning Score: 3.30

### Geochemistry
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 31.60
Average Complexity: 0.3456
Average Reasoning Score: 3.40

### Geographical Trivia
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3226
Average Reasoning Score: 3.10

### Geography General
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.70
Average Complexity: 0.3158
Average Reasoning Score: 3.00

### Geology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.20
Average Complexity: 0.3389
Average Reasoning Score: 3.30

### Geometry
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 30.70
Average Complexity: 0.3288
Average Reasoning Score: 3.10

### Geomorphology
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 32.40
Average Complexity: 0.3673
Average Reasoning Score: 3.70

### Geospatial Analysis
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.30
Average Complexity: 0.3447
Average Reasoning Score: 3.40

### German Language
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.70
Average Complexity: 0.3268
Average Reasoning Score: 3.10

### Gig Economy
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.70
Average Complexity: 0.3363
Average Reasoning Score: 3.30

### Glaciology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.00
Average Complexity: 0.3322
Average Reasoning Score: 3.20

### Glass Painting
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.40
Average Complexity: 0.3222
Average Reasoning Score: 3.10

### Global Customs
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.60
Average Complexity: 0.3632
Average Reasoning Score: 3.70

### Global Environmental Agreements Non Political
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 41.10
Average Complexity: 0.3319
Average Reasoning Score: 3.30

### Gluten Free Cooking
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.40
Average Complexity: 0.3194
Average Reasoning Score: 3.00

### Go Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.20
Average Complexity: 0.4601
Average Reasoning Score: 4.60

### Go To Market Strategies
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.90
Average Complexity: 0.4014
Average Reasoning Score: 4.00

### Goal Reviews
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.70
Average Complexity: 0.4360
Average Reasoning Score: 4.20

### Goal Setting
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.70
Average Complexity: 0.3539
Average Reasoning Score: 3.50

### Grant Writing
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.80
Average Complexity: 0.3510
Average Reasoning Score: 3.50

### Graphic Design
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.40
Average Complexity: 0.3273
Average Reasoning Score: 3.10

### Graphic Design Freelancing
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3343
Average Reasoning Score: 3.30

### Graphic Novels
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.00
Average Complexity: 0.3902
Average Reasoning Score: 4.00

### Graphic Novels Advanced
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.00
Average Complexity: 0.4184
Average Reasoning Score: 4.20

### Gratitude Practice
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.20
Average Complexity: 0.3112
Average Reasoning Score: 3.00

### Gravitational Waves
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.90
Average Complexity: 0.3539
Average Reasoning Score: 3.60

### Green Living Hacks
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.70
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Green Policies Non Political
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.00
Average Complexity: 0.3414
Average Reasoning Score: 3.30

### Green Technology
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.50
Average Complexity: 0.3699
Average Reasoning Score: 3.60

### Grilling And Smoking
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.60
Average Complexity: 0.3226
Average Reasoning Score: 3.10

### Habit Tracking
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3193
Average Reasoning Score: 3.00

### Habitat Restoration
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.20
Average Complexity: 0.3616
Average Reasoning Score: 3.60

### Habits And Routines
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.20
Average Complexity: 0.3368
Average Reasoning Score: 3.30

### Health And Wellness
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.00
Average Complexity: 0.3437
Average Reasoning Score: 3.50

### Healthy Aging
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.50
Average Complexity: 0.3204
Average Reasoning Score: 3.10

### Help Desk Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.50
Average Complexity: 0.3687
Average Reasoning Score: 3.70

### Herbal Supplements General
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.50
Average Complexity: 0.3215
Average Reasoning Score: 3.10

### Heritage Management
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.30
Average Complexity: 0.3323
Average Reasoning Score: 3.20

### Heritage Tourism
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.00
Average Complexity: 0.3157
Average Reasoning Score: 3.00

### Herpetology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.60
Average Complexity: 0.3329
Average Reasoning Score: 3.30

### High Performance Computing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.80
Average Complexity: 0.4271
Average Reasoning Score: 4.20

### Hiking
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.40
Average Complexity: 0.3344
Average Reasoning Score: 3.20

### Hip Hop Non Political
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.80
Average Complexity: 0.3317
Average Reasoning Score: 3.20

### Historic Preservation
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 38.10
Average Complexity: 0.3488
Average Reasoning Score: 3.50

### Historical Facts
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.20
Average Complexity: 0.3274
Average Reasoning Score: 3.20

### Historical Fiction
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.40
Average Complexity: 0.4069
Average Reasoning Score: 4.10

### History General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3210
Average Reasoning Score: 3.10

### History Of Science
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 39.90
Average Complexity: 0.3278
Average Reasoning Score: 3.20

### Hobbies General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.50
Average Complexity: 0.3349
Average Reasoning Score: 3.30

### Holistic Health
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.50
Average Complexity: 0.3383
Average Reasoning Score: 3.20

### Holistic Nutrition
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.60
Average Complexity: 0.3325
Average Reasoning Score: 3.20

### Holography
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.10
Average Complexity: 0.3703
Average Reasoning Score: 3.60

### Home D Cor
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.00
Average Complexity: 0.3171
Average Reasoning Score: 3.00

### Home Improvement
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.50
Average Complexity: 0.3274
Average Reasoning Score: 3.40

### Homebrewing Non Erotic
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.20
Average Complexity: 0.3274
Average Reasoning Score: 3.10

### Homeschool Activities
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.00
Average Complexity: 0.3221
Average Reasoning Score: 3.10

### Homeschooling
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.40
Average Complexity: 0.3730
Average Reasoning Score: 3.70

### Horror Non Erotic
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.20
Average Complexity: 0.3525
Average Reasoning Score: 3.50

### Horticulture
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 34.10
Average Complexity: 0.3269
Average Reasoning Score: 3.20

### Hospitality
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.40
Average Complexity: 0.3214
Average Reasoning Score: 3.10

### Hospitality Etiquette
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 40.10
Average Complexity: 0.3362
Average Reasoning Score: 3.30

### Hospitality Marketing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.80
Average Complexity: 0.3285
Average Reasoning Score: 3.10

### Hotel Management
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.10
Average Complexity: 0.3259
Average Reasoning Score: 3.30

### Housing Policy Non Political
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 40.00
Average Complexity: 0.3409
Average Reasoning Score: 3.30

### Human Geography
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.10
Average Complexity: 0.3165
Average Reasoning Score: 3.00

### Human Resources
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.80
Average Complexity: 0.3566
Average Reasoning Score: 3.70

### Human Resources Advanced
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 39.90
Average Complexity: 0.3736
Average Reasoning Score: 3.70

### Hunting Conservation Centric
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.40
Average Complexity: 0.3501
Average Reasoning Score: 3.40

### Hydrogeology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.80
Average Complexity: 0.3598
Average Reasoning Score: 3.40

### Hydrology
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 33.80
Average Complexity: 0.3669
Average Reasoning Score: 3.70

### Hydrotherapy
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.10
Average Complexity: 0.3350
Average Reasoning Score: 3.30

### Ichthyology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.00
Average Complexity: 0.3357
Average Reasoning Score: 3.20

### Immunology Basics
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 33.70
Average Complexity: 0.3587
Average Reasoning Score: 3.80

### Impact Assessment
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.00
Average Complexity: 0.3613
Average Reasoning Score: 3.60

### Impressionism
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.70
Average Complexity: 0.3188
Average Reasoning Score: 3.00

### Improvisational Theatre
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.00
Average Complexity: 0.4113
Average Reasoning Score: 4.00

### Incident Response
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.90
Average Complexity: 0.5102
Average Reasoning Score: 5.20

### Indian Cuisine
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.00
Average Complexity: 0.3182
Average Reasoning Score: 3.00

### Individual Sports
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.90
Average Complexity: 0.3232
Average Reasoning Score: 3.10

### Industrial Design
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.40
Average Complexity: 0.3373
Average Reasoning Score: 3.30

### Industrial Engineering
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.80
Average Complexity: 0.3580
Average Reasoning Score: 3.60

### Industrial Organizational Psychology
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 39.40
Average Complexity: 0.3589
Average Reasoning Score: 3.60

### Influencer Marketing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.60
Average Complexity: 0.3551
Average Reasoning Score: 3.60

### Influencer Outreach
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.90
Average Complexity: 0.3567
Average Reasoning Score: 3.50

### Information Literacy
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3355
Average Reasoning Score: 3.30

### Information Technology
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 31.60
Average Complexity: 0.4245
Average Reasoning Score: 4.40

### Infrastructure As Code
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.80
Average Complexity: 0.5152
Average Reasoning Score: 5.20

### Infrastructure Automation
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.80
Average Complexity: 0.5345
Average Reasoning Score: 5.30

### Innovation Management
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.70
Average Complexity: 0.3419
Average Reasoning Score: 3.40

### Instructional Design
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3474
Average Reasoning Score: 3.30

### Insurance Basics
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.80
Average Complexity: 0.3790
Average Reasoning Score: 3.80

### Interior Design
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.10
Average Complexity: 0.3294
Average Reasoning Score: 3.10

### Interior Styling
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.50
Average Complexity: 0.3191
Average Reasoning Score: 3.00

### International Trade
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.60
Average Complexity: 0.3643
Average Reasoning Score: 3.70

### Internet Of Things
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.00
Average Complexity: 0.4077
Average Reasoning Score: 4.20

### Interpersonal Communication
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.60
Average Complexity: 0.3425
Average Reasoning Score: 3.20

### Interview Skills
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.80
Average Complexity: 0.3482
Average Reasoning Score: 3.60

### Invasive Species
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.90
Average Complexity: 0.4048
Average Reasoning Score: 4.10

### Investing
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.80
Average Complexity: 0.4021
Average Reasoning Score: 4.00

### It Infrastructure
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 34.50
Average Complexity: 0.5059
Average Reasoning Score: 5.20

### It Service Management
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.70
Average Complexity: 0.3401
Average Reasoning Score: 3.40

### Italian Cuisine
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.00
Average Complexity: 0.3188
Average Reasoning Score: 3.00

### Italian Language
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.40
Average Complexity: 0.3229
Average Reasoning Score: 3.10

### Japanese Cuisine
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 32.90
Average Complexity: 0.3181
Average Reasoning Score: 3.00

### Japanese Language
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.60
Average Complexity: 0.3261
Average Reasoning Score: 3.10

### Java Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.30
Average Complexity: 0.4404
Average Reasoning Score: 4.50

### Javascript Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 31.60
Average Complexity: 0.4065
Average Reasoning Score: 4.00

### Jazz
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 32.60
Average Complexity: 0.3179
Average Reasoning Score: 3.00

### Jewelry Design
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.50
Average Complexity: 0.3193
Average Reasoning Score: 3.00

### Job Hunting
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.80
Average Complexity: 0.3251
Average Reasoning Score: 3.10

### Journaling
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.80
Average Complexity: 0.3103
Average Reasoning Score: 3.00

### Journalism Non Political
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.90
Average Complexity: 0.3575
Average Reasoning Score: 3.60

### Junior Sports
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 38.50
Average Complexity: 0.3230
Average Reasoning Score: 3.10

### Kanban
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 31.80
Average Complexity: 0.4033
Average Reasoning Score: 4.00

### Kids Crafts
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 39.40
Average Complexity: 0.3165
Average Reasoning Score: 3.00

### Kids Robotics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.70
Average Complexity: 0.3339
Average Reasoning Score: 3.30

### Knitting And Crochet
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.80
Average Complexity: 0.3223
Average Reasoning Score: 3.10

### Kotlin Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.50
Average Complexity: 0.4366
Average Reasoning Score: 4.30

### Kubernetes
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 28.50
Average Complexity: 0.4917
Average Reasoning Score: 5.00

### Labor Economics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.80
Average Complexity: 0.3289
Average Reasoning Score: 3.20

### Landscape Architecture
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.80
Average Complexity: 0.3414
Average Reasoning Score: 3.20

### Language Acquisition
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.90
Average Complexity: 0.3307
Average Reasoning Score: 3.30

### Language Learning For Kids
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 41.90
Average Complexity: 0.3433
Average Reasoning Score: 3.50

### Language Trivia
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.10
Average Complexity: 0.3273
Average Reasoning Score: 3.10

### Languages General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.40
Average Complexity: 0.3620
Average Reasoning Score: 3.60

### Latch Hooking
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 40.60
Average Complexity: 0.3256
Average Reasoning Score: 3.10

### Leadership
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.40
Average Complexity: 0.3358
Average Reasoning Score: 3.30

### Lean Startup
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3477
Average Reasoning Score: 3.40

### Learning Strategies
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.40
Average Complexity: 0.3671
Average Reasoning Score: 3.60

### Learning Theories
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.70
Average Complexity: 0.3371
Average Reasoning Score: 3.20

### Library Management
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.10
Average Complexity: 0.3515
Average Reasoning Score: 3.60

### Library Science
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.80
Average Complexity: 0.3365
Average Reasoning Score: 3.30

### Lifestyle Audits
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 36.50
Average Complexity: 0.3605
Average Reasoning Score: 3.70

### Lifestyle General
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.40
Average Complexity: 0.3316
Average Reasoning Score: 3.10

### Linguistic Anthropology
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.40
Average Complexity: 0.3216
Average Reasoning Score: 3.10

### Linguistics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 30.60
Average Complexity: 0.3446
Average Reasoning Score: 3.40

### Literary Agents
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 35.70
Average Complexity: 0.3692
Average Reasoning Score: 3.70

### Literary Criticism
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.40
Average Complexity: 0.3224
Average Reasoning Score: 3.00

### Literary Genres
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.80
Average Complexity: 0.3647
Average Reasoning Score: 3.70

### Literature General
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.30
Average Complexity: 0.3687
Average Reasoning Score: 3.70

### Livestreaming
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.30
Average Complexity: 0.3315
Average Reasoning Score: 3.20

### Logic
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.10
Average Complexity: 0.3237
Average Reasoning Score: 3.20

### Logic Puzzles
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.00
Average Complexity: 0.3550
Average Reasoning Score: 3.60

### Logistics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.30
Average Complexity: 0.3355
Average Reasoning Score: 3.20

### Lunar Exploration
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 39.20
Average Complexity: 0.3279
Average Reasoning Score: 3.20

### Machine Learning
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 34.40
Average Complexity: 0.4276
Average Reasoning Score: 4.40

### Machine Learning Advanced
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.50
Average Complexity: 0.4280
Average Reasoning Score: 4.30

### Macram
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 30.80
Average Complexity: 0.3162
Average Reasoning Score: 3.00

### Macroeconomics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.10
Average Complexity: 0.3208
Average Reasoning Score: 3.10

### Magic Shows
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.90
Average Complexity: 0.3782
Average Reasoning Score: 4.00

### Magical Realism
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.30
Average Complexity: 0.3510
Average Reasoning Score: 3.40

### Malware Analysis
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.30
Average Complexity: 0.3915
Average Reasoning Score: 3.90

### Mammalogy
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.50
Average Complexity: 0.3186
Average Reasoning Score: 3.00

### Mandarin Chinese
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.80
Average Complexity: 0.3281
Average Reasoning Score: 3.10

### Manga Art Style
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.60
Average Complexity: 0.3745
Average Reasoning Score: 3.90

### Marine Biology
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.90
Average Complexity: 0.3548
Average Reasoning Score: 3.50

### Marine Engineering
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.40
Average Complexity: 0.3668
Average Reasoning Score: 3.60

### Marine Transport
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.60
Average Complexity: 0.3370
Average Reasoning Score: 3.30

### Maritime History
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.00
Average Complexity: 0.3311
Average Reasoning Score: 3.20

### Market Research
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.40
Average Complexity: 0.3334
Average Reasoning Score: 3.20

### Marketing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.30
Average Complexity: 0.3313
Average Reasoning Score: 3.20

### Mars Missions
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.70
Average Complexity: 0.3276
Average Reasoning Score: 3.10

### Massage Therapy
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.40
Average Complexity: 0.3293
Average Reasoning Score: 3.20

### Mechanical Engineering
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.20
Average Complexity: 0.3410
Average Reasoning Score: 3.30

### Media Communication
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.10
Average Complexity: 0.3266
Average Reasoning Score: 3.20

### Media Ethics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.50
Average Complexity: 0.3410
Average Reasoning Score: 3.40

### Media Ethics Non Religious
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.30
Average Complexity: 0.3306
Average Reasoning Score: 3.20

### Media General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.20
Average Complexity: 0.3302
Average Reasoning Score: 3.20

### Medieval Studies
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3262
Average Reasoning Score: 3.10

### Meditation Non Religious
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.30
Average Complexity: 0.3194
Average Reasoning Score: 3.00

### Meditation Retreats Non Religious
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.30
Average Complexity: 0.3268
Average Reasoning Score: 3.10

### Mediterranean Cuisine
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.30
Average Complexity: 0.3191
Average Reasoning Score: 3.00

### Meetup Organization
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.10
Average Complexity: 0.3388
Average Reasoning Score: 3.30

### Memoirs
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.00
Average Complexity: 0.3747
Average Reasoning Score: 3.80

### Memory Palaces
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.40
Average Complexity: 0.3277
Average Reasoning Score: 3.00

### Memory Techniques
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.90
Average Complexity: 0.3369
Average Reasoning Score: 3.30

### Memory Training
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.40
Average Complexity: 0.3323
Average Reasoning Score: 3.30

### Mental Health
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.30
Average Complexity: 0.3730
Average Reasoning Score: 3.70

### Mentoring
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.70
Average Complexity: 0.3377
Average Reasoning Score: 3.20

### Metacognition
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.70
Average Complexity: 0.3701
Average Reasoning Score: 3.70

### Metadata Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.10
Average Complexity: 0.3523
Average Reasoning Score: 3.50

### Metalworking
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.20
Average Complexity: 0.3319
Average Reasoning Score: 3.20

### Metaphysics Non Religious
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 31.10
Average Complexity: 0.3407
Average Reasoning Score: 3.40

### Mexican Cuisine
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.20
Average Complexity: 0.3184
Average Reasoning Score: 3.00

### Microeconomics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.10
Average Complexity: 0.3164
Average Reasoning Score: 3.00

### Microservices
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 31.90
Average Complexity: 0.5276
Average Reasoning Score: 5.30

### Middle Eastern Cuisine Non Religious
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.90
Average Complexity: 0.3226
Average Reasoning Score: 3.10

### Mime
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.70
Average Complexity: 0.4042
Average Reasoning Score: 3.90

### Mind Body Connection
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.30
Average Complexity: 0.3436
Average Reasoning Score: 3.40

### Mind Mapping
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 33.90
Average Complexity: 0.3851
Average Reasoning Score: 3.80

### Mindful Eating
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.60
Average Complexity: 0.3140
Average Reasoning Score: 3.00

### Mindfulness
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.00
Average Complexity: 0.3156
Average Reasoning Score: 3.00

### Mineralogy
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.50
Average Complexity: 0.3453
Average Reasoning Score: 3.40

### Minimalism
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.70
Average Complexity: 0.3261
Average Reasoning Score: 3.10

### Minimalist Wardrobe
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.50
Average Complexity: 0.3211
Average Reasoning Score: 3.00

### Ml Ethics Non Political
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 38.60
Average Complexity: 0.4095
Average Reasoning Score: 3.90

### Mnemonic Devices
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3349
Average Reasoning Score: 3.30

### Mobile App Development
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 36.40
Average Complexity: 0.3918
Average Reasoning Score: 3.90

### Model Building
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.80
Average Complexity: 0.3234
Average Reasoning Score: 3.20

### Modern History
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.90
Average Complexity: 0.3288
Average Reasoning Score: 3.20

### Modernism
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.30
Average Complexity: 0.3437
Average Reasoning Score: 3.40

### Moral Psychology
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.90
Average Complexity: 0.3559
Average Reasoning Score: 3.50

### Motion Graphics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.60
Average Complexity: 0.3382
Average Reasoning Score: 3.40

### Motivation
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.50
Average Complexity: 0.3221
Average Reasoning Score: 3.00

### Museology General
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.70
Average Complexity: 0.3313
Average Reasoning Score: 3.20

### Museum Education
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.80
Average Complexity: 0.3351
Average Reasoning Score: 3.20

### Museum Marketing
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.70
Average Complexity: 0.3397
Average Reasoning Score: 3.30

### Museum Studies
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.50
Average Complexity: 0.3368
Average Reasoning Score: 3.30

### Museum Technology
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.60
Average Complexity: 0.3339
Average Reasoning Score: 3.30

### Music General
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.60
Average Complexity: 0.3278
Average Reasoning Score: 3.20

### Music Production
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.90
Average Complexity: 0.3381
Average Reasoning Score: 3.30

### Music Theory
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.90
Average Complexity: 0.3220
Average Reasoning Score: 3.10

### Musical Theatre
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.30
Average Complexity: 0.3464
Average Reasoning Score: 3.40

### Mystery
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.40
Average Complexity: 0.3756
Average Reasoning Score: 3.70

### Mythology General Non Religious
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.90
Average Complexity: 0.3226
Average Reasoning Score: 3.10

### Nanotechnology
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 31.40
Average Complexity: 0.3577
Average Reasoning Score: 3.30

### Natural Language Processing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.50
Average Complexity: 0.4145
Average Reasoning Score: 4.20

### Nature Conservation Efforts
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 38.30
Average Complexity: 0.3324
Average Reasoning Score: 3.20

### Nature General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.10
Average Complexity: 0.3274
Average Reasoning Score: 3.10

### Nature Photography
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.30
Average Complexity: 0.3534
Average Reasoning Score: 3.60

### Network Forensics
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.20
Average Complexity: 0.4324
Average Reasoning Score: 4.50

### Network Security
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.30
Average Complexity: 0.5296
Average Reasoning Score: 5.30

### Networking
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.10
Average Complexity: 0.4717
Average Reasoning Score: 4.80

### Neural Interfaces
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.20
Average Complexity: 0.3649
Average Reasoning Score: 3.80

### Neuroplasticity
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.20
Average Complexity: 0.3188
Average Reasoning Score: 3.10

### Neuropsychology
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.30
Average Complexity: 0.3601
Average Reasoning Score: 3.70

### Neuroscience
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 30.30
Average Complexity: 0.3440
Average Reasoning Score: 3.40

### Neutron Stars
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.80
Average Complexity: 0.3262
Average Reasoning Score: 3.20

### Nonprofit Leadership
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.70
Average Complexity: 0.3307
Average Reasoning Score: 3.30

### Nonprofit Management
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.60
Average Complexity: 0.3545
Average Reasoning Score: 3.60

### Nonverbal Communication
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.60
Average Complexity: 0.3352
Average Reasoning Score: 3.00

### Note Taking Systems
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.10
Average Complexity: 0.3464
Average Reasoning Score: 3.30

### Novels
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.00
Average Complexity: 0.4238
Average Reasoning Score: 4.20

### Nuclear Engineering
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.80
Average Complexity: 0.3583
Average Reasoning Score: 3.60

### Number Theory
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.70
Average Complexity: 0.3550
Average Reasoning Score: 3.50

### Nutrition
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 31.50
Average Complexity: 0.3320
Average Reasoning Score: 3.20

### Observability
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 31.80
Average Complexity: 0.5170
Average Reasoning Score: 5.00

### Onboarding
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.90
Average Complexity: 0.3431
Average Reasoning Score: 3.40

### Online Communities
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.50
Average Complexity: 0.3254
Average Reasoning Score: 3.10

### Online Etiquette
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.80
Average Complexity: 0.3421
Average Reasoning Score: 3.40

### Online Learning
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.70
Average Complexity: 0.3463
Average Reasoning Score: 3.30

### Online Mentorship
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3493
Average Reasoning Score: 3.30

### Open Access Publishing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.50
Average Complexity: 0.3558
Average Reasoning Score: 3.60

### Open Source Software
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 34.80
Average Complexity: 0.4054
Average Reasoning Score: 4.00

### Opera Non Religious
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.00
Average Complexity: 0.3561
Average Reasoning Score: 3.50

### Opera Production
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.70
Average Complexity: 0.3552
Average Reasoning Score: 3.50

### Operating Systems
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.30
Average Complexity: 0.4372
Average Reasoning Score: 4.40

### Operations Research
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.50
Average Complexity: 0.4370
Average Reasoning Score: 4.40

### Organic Farming
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.30
Average Complexity: 0.3249
Average Reasoning Score: 3.20

### Organization
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.80
Average Complexity: 0.3402
Average Reasoning Score: 3.30

### Organizational Behavior
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.00
Average Complexity: 0.3613
Average Reasoning Score: 3.50

### Organizational Culture
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.00
Average Complexity: 0.3251
Average Reasoning Score: 3.10

### Origami
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.10
Average Complexity: 0.3145
Average Reasoning Score: 3.00

### Ornithology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.70
Average Complexity: 0.3520
Average Reasoning Score: 3.40

### Outdoor Play
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.80
Average Complexity: 0.3275
Average Reasoning Score: 3.10

### Overcoming Procrastination
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 34.70
Average Complexity: 0.3370
Average Reasoning Score: 3.30

### Painting
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.20
Average Complexity: 0.3248
Average Reasoning Score: 3.10

### Pantomime
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 32.20
Average Complexity: 0.4087
Average Reasoning Score: 4.00

### Paper Crafts
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.20
Average Complexity: 0.3182
Average Reasoning Score: 3.00

### Parallel Computing
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.60
Average Complexity: 0.4646
Average Reasoning Score: 4.60

### Parenting General
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.40
Average Complexity: 0.3285
Average Reasoning Score: 3.10

### Parenting Tips
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 38.00
Average Complexity: 0.3258
Average Reasoning Score: 3.10

### Parkour
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 31.30
Average Complexity: 0.3386
Average Reasoning Score: 3.40

### Pastry Techniques
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.80
Average Complexity: 0.3183
Average Reasoning Score: 3.00

### Pedagogy
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.10
Average Complexity: 0.3461
Average Reasoning Score: 3.30

### Peer Learning
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.40
Average Complexity: 0.3518
Average Reasoning Score: 3.40

### Peer Relationships
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.30
Average Complexity: 0.3271
Average Reasoning Score: 3.10

### Peer To Peer Lending
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.30
Average Complexity: 0.3757
Average Reasoning Score: 3.80

### Penetration Testing
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 34.20
Average Complexity: 0.4876
Average Reasoning Score: 4.90

### Performance Appraisals
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.50
Average Complexity: 0.4345
Average Reasoning Score: 4.40

### Performance Theory
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.30
Average Complexity: 0.3839
Average Reasoning Score: 3.90

### Performing Arts
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.50
Average Complexity: 0.3730
Average Reasoning Score: 3.80

### Performing Arts Advanced
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.30
Average Complexity: 0.3764
Average Reasoning Score: 3.90

### Personal Branding
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.70
Average Complexity: 0.3526
Average Reasoning Score: 3.30

### Personal Development
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.30
Average Complexity: 0.3408
Average Reasoning Score: 3.20

### Personal Finance
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.80
Average Complexity: 0.3744
Average Reasoning Score: 3.70

### Personal Finance Advanced
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 38.00
Average Complexity: 0.4091
Average Reasoning Score: 4.20

### Personal Organization
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.50
Average Complexity: 0.3443
Average Reasoning Score: 3.20

### Pet Care
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.90
Average Complexity: 0.3444
Average Reasoning Score: 3.50

### Petrology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.30
Average Complexity: 0.3319
Average Reasoning Score: 3.20

### Philosophy General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 30.40
Average Complexity: 0.3241
Average Reasoning Score: 3.20

### Philosophy Of Language
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.60
Average Complexity: 0.3369
Average Reasoning Score: 3.50

### Philosophy Of Mind
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.10
Average Complexity: 0.3328
Average Reasoning Score: 3.20

### Philosophy Of Science
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.40
Average Complexity: 0.3348
Average Reasoning Score: 3.30

### Phonetics
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.80
Average Complexity: 0.3305
Average Reasoning Score: 3.10

### Photography
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 31.40
Average Complexity: 0.3699
Average Reasoning Score: 3.80

### Physical Fitness
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.90
Average Complexity: 0.3250
Average Reasoning Score: 3.10

### Physical Geography
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.20
Average Complexity: 0.3379
Average Reasoning Score: 3.20

### Physics
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.50
Average Complexity: 0.3249
Average Reasoning Score: 3.10

### Pickling
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.30
Average Complexity: 0.3289
Average Reasoning Score: 3.20

### Pilates
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.90
Average Complexity: 0.3268
Average Reasoning Score: 3.10

### Pilot Training
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.10
Average Complexity: 0.3295
Average Reasoning Score: 3.30

### Planetary Science
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.00
Average Complexity: 0.3494
Average Reasoning Score: 3.50

### Plant Based Desserts
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 37.00
Average Complexity: 0.3194
Average Reasoning Score: 3.00

### Plant Ecology
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.20
Average Complexity: 0.3436
Average Reasoning Score: 3.30

### Plant Pathology
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.70
Average Complexity: 0.3970
Average Reasoning Score: 4.00

### Plant Taxonomy
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 32.90
Average Complexity: 0.3179
Average Reasoning Score: 3.00

### Plate Tectonics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.00
Average Complexity: 0.3354
Average Reasoning Score: 3.30

### Poaching
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 33.40
Average Complexity: 0.3189
Average Reasoning Score: 3.00

### Podcasting
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.70
Average Complexity: 0.3307
Average Reasoning Score: 3.30

### Poetry
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 30.50
Average Complexity: 0.3782
Average Reasoning Score: 4.00

### Pop Art
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.60
Average Complexity: 0.3739
Average Reasoning Score: 3.70

### Portuguese Language
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.10
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Positive Discipline
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.90
Average Complexity: 0.3180
Average Reasoning Score: 3.10

### Positive Psychology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.50
Average Complexity: 0.3313
Average Reasoning Score: 3.10

### Post Production Editing
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.10
Average Complexity: 0.3176
Average Reasoning Score: 3.00

### Postmodernism
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 30.20
Average Complexity: 0.3634
Average Reasoning Score: 3.60

### Pragmatics
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 32.90
Average Complexity: 0.3389
Average Reasoning Score: 3.50

### Predictive Analytics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.30
Average Complexity: 0.3874
Average Reasoning Score: 3.80

### Prehistoric Archaeology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.80
Average Complexity: 0.3323
Average Reasoning Score: 3.20

### Preservation Of Materials
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 39.30
Average Complexity: 0.3648
Average Reasoning Score: 3.50

### Pressure Cooking
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.60
Average Complexity: 0.3264
Average Reasoning Score: 3.20

### Preventive Medicine
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.30
Average Complexity: 0.3645
Average Reasoning Score: 3.50

### Primatology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.80
Average Complexity: 0.3215
Average Reasoning Score: 3.20

### Printmaking
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.30
Average Complexity: 0.3381
Average Reasoning Score: 3.30

### Product Analytics
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.80
Average Complexity: 0.3473
Average Reasoning Score: 3.40

### Product Design
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.20
Average Complexity: 0.3196
Average Reasoning Score: 3.00

### Product Development
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.70
Average Complexity: 0.3361
Average Reasoning Score: 3.40

### Product Led Growth
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.30
Average Complexity: 0.3261
Average Reasoning Score: 3.10

### Product Management
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.30
Average Complexity: 0.3386
Average Reasoning Score: 3.40

### Productivity Apps
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.70
Average Complexity: 0.3415
Average Reasoning Score: 3.30

### Professional Ethics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.50
Average Complexity: 0.3681
Average Reasoning Score: 3.50

### Professional Etiquette
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.30
Average Complexity: 0.3265
Average Reasoning Score: 3.00

### Professional Image
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.10
Average Complexity: 0.3271
Average Reasoning Score: 3.00

### Program Evaluation
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.70
Average Complexity: 0.3407
Average Reasoning Score: 3.40

### Programming Languages
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 31.70
Average Complexity: 0.4177
Average Reasoning Score: 4.30

### Project Based Learning
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.60
Average Complexity: 0.3704
Average Reasoning Score: 3.50

### Project Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.40
Average Complexity: 0.3549
Average Reasoning Score: 3.50

### Project Planning
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.40
Average Complexity: 0.3952
Average Reasoning Score: 4.00

### Psycholinguistics
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 33.60
Average Complexity: 0.3624
Average Reasoning Score: 3.80

### Psychology General
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 35.60
Average Complexity: 0.3765
Average Reasoning Score: 3.70

### Pub Quizzes Non Political
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.10
Average Complexity: 0.3369
Average Reasoning Score: 3.30

### Public Relations
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.90
Average Complexity: 0.3360
Average Reasoning Score: 3.40

### Public Spaces Design
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.60
Average Complexity: 0.3249
Average Reasoning Score: 3.10

### Public Speaking
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.00
Average Complexity: 0.3426
Average Reasoning Score: 3.60

### Public Speaking Advanced
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.30
Average Complexity: 0.3409
Average Reasoning Score: 3.50

### Publishing
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 32.60
Average Complexity: 0.3542
Average Reasoning Score: 3.60

### Puppetry
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.30
Average Complexity: 0.3903
Average Reasoning Score: 4.00

### Puppetry For Kids
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.70
Average Complexity: 0.3899
Average Reasoning Score: 4.00

### Puzzle Games
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.80
Average Complexity: 0.3270
Average Reasoning Score: 3.20

### Python Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.00
Average Complexity: 0.3926
Average Reasoning Score: 4.20

### Quantum Mechanics
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.10
Average Complexity: 0.3937
Average Reasoning Score: 3.80

### Quizzing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 30.60
Average Complexity: 0.3159
Average Reasoning Score: 3.00

### Rail Systems
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 36.40
Average Complexity: 0.3424
Average Reasoning Score: 3.30

### Real Estate Investing
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 40.80
Average Complexity: 0.4127
Average Reasoning Score: 4.20

### Recipe Development
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.30
Average Complexity: 0.3180
Average Reasoning Score: 3.00

### Recommendation Systems
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.30
Average Complexity: 0.3762
Average Reasoning Score: 3.70

### Recreation
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.40
Average Complexity: 0.3338
Average Reasoning Score: 3.30

### Recruitment
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.60
Average Complexity: 0.3730
Average Reasoning Score: 3.70

### Recycling
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 34.00
Average Complexity: 0.3510
Average Reasoning Score: 3.50

### Reddit Management
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.50
Average Complexity: 0.3383
Average Reasoning Score: 3.30

### Reflexology
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 33.20
Average Complexity: 0.3136
Average Reasoning Score: 3.00

### Regional Cuisines
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.30
Average Complexity: 0.3189
Average Reasoning Score: 3.00

### Reinforcement Learning
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.20
Average Complexity: 0.5302
Average Reasoning Score: 5.30

### Relationships Non Erotic
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.30
Average Complexity: 0.3251
Average Reasoning Score: 3.10

### Remote Work
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.50
Average Complexity: 0.3567
Average Reasoning Score: 3.60

### Renaissance Art
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.20
Average Complexity: 0.3371
Average Reasoning Score: 3.20

### Renaissance Studies
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 34.40
Average Complexity: 0.3222
Average Reasoning Score: 3.10

### Renewable Energy
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.40
Average Complexity: 0.3603
Average Reasoning Score: 3.40

### Renewable Energy Careers
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.80
Average Complexity: 0.3838
Average Reasoning Score: 3.80

### Requirement Gathering
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.50
Average Complexity: 0.3442
Average Reasoning Score: 3.40

### Restaurant Management
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.40
Average Complexity: 0.3330
Average Reasoning Score: 3.10

### Resume Building
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.00
Average Complexity: 0.3377
Average Reasoning Score: 3.20

### Retirement Planning
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 38.40
Average Complexity: 0.4004
Average Reasoning Score: 4.00

### Riddles
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.60
Average Complexity: 0.3217
Average Reasoning Score: 3.10

### Risk Management
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 38.10
Average Complexity: 0.4302
Average Reasoning Score: 4.30

### Roadmapping
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 31.10
Average Complexity: 0.3538
Average Reasoning Score: 3.50

### Robotics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.40
Average Complexity: 0.3432
Average Reasoning Score: 3.40

### Rock Collecting
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.70
Average Complexity: 0.3168
Average Reasoning Score: 3.10

### Rock Music
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.20
Average Complexity: 0.3306
Average Reasoning Score: 3.20

### Rocket Propulsion
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.50
Average Complexity: 0.3578
Average Reasoning Score: 3.60

### Role Playing Games
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.30
Average Complexity: 0.3523
Average Reasoning Score: 3.40

### Romance Clean Content
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.10
Average Complexity: 0.3540
Average Reasoning Score: 3.50

### Rural Sociology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 40.60
Average Complexity: 0.3127
Average Reasoning Score: 3.10

### Russian Language
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.30
Average Complexity: 0.3171
Average Reasoning Score: 3.00

### Rust Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.60
Average Complexity: 0.4874
Average Reasoning Score: 4.90

### Salary Negotiation
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.60
Average Complexity: 0.3613
Average Reasoning Score: 3.60

### Science Facts
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 32.90
Average Complexity: 0.3376
Average Reasoning Score: 3.20

### Science Fiction
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.40
Average Complexity: 0.4153
Average Reasoning Score: 4.20

### Scrapbooking
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.80
Average Complexity: 0.3161
Average Reasoning Score: 3.00

### Screen Time Management
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 40.80
Average Complexity: 0.3268
Average Reasoning Score: 3.20

### Screenwriting
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.50
Average Complexity: 0.4029
Average Reasoning Score: 4.10

### Scrum Framework
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.90
Average Complexity: 0.3419
Average Reasoning Score: 3.30

### Sculpture
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 32.40
Average Complexity: 0.3230
Average Reasoning Score: 3.10

### Search Engine Marketing
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.00
Average Complexity: 0.3446
Average Reasoning Score: 3.40

### Search Engine Optimization
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.40
Average Complexity: 0.3423
Average Reasoning Score: 3.20

### Seismology
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.20
Average Complexity: 0.4050
Average Reasoning Score: 3.90

### Self Compassion
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.20
Average Complexity: 0.3272
Average Reasoning Score: 3.10

### Self Directed Learning
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.50
Average Complexity: 0.3556
Average Reasoning Score: 3.30

### Self Evaluation
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 33.90
Average Complexity: 0.3419
Average Reasoning Score: 3.40

### Self Help Topics
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.50
Average Complexity: 0.3278
Average Reasoning Score: 3.00

### Self Publishing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 32.30
Average Complexity: 0.3269
Average Reasoning Score: 3.20

### Self Reflection
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.60
Average Complexity: 0.3410
Average Reasoning Score: 3.20

### Semantics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.30
Average Complexity: 0.3600
Average Reasoning Score: 3.50

### Serverless Architecture
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 39.20
Average Complexity: 0.5298
Average Reasoning Score: 5.30

### Shell Collecting
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.00
Average Complexity: 0.3182
Average Reasoning Score: 3.00

### Short Stories
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.70
Average Complexity: 0.3743
Average Reasoning Score: 3.70

### Sibling Relationships
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.50
Average Complexity: 0.3269
Average Reasoning Score: 3.10

### Skateboarding
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 33.10
Average Complexity: 0.3173
Average Reasoning Score: 3.00

### Skiing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.80
Average Complexity: 0.3245
Average Reasoning Score: 3.10

### Sleep Science
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3188
Average Reasoning Score: 3.10

### Slow Cooking
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 37.00
Average Complexity: 0.3185
Average Reasoning Score: 3.00

### Small Business
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.60
Average Complexity: 0.3286
Average Reasoning Score: 3.20

### Smart Cities
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.50
Average Complexity: 0.3509
Average Reasoning Score: 3.50

### Smart Homes
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.50
Average Complexity: 0.3434
Average Reasoning Score: 3.40

### Smart Materials
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.70
Average Complexity: 0.3398
Average Reasoning Score: 3.30

### Snowboarding
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.80
Average Complexity: 0.3306
Average Reasoning Score: 3.20

### Social Enterprise
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 39.00
Average Complexity: 0.3381
Average Reasoning Score: 3.20

### Social Etiquette
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.60
Average Complexity: 0.3212
Average Reasoning Score: 3.00

### Social History
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.60
Average Complexity: 0.3292
Average Reasoning Score: 3.20

### Social Media Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.20
Average Complexity: 0.3580
Average Reasoning Score: 3.50

### Social Media Strategy
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.60
Average Complexity: 0.3440
Average Reasoning Score: 3.40

### Social Movements Non Political Focus
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 39.10
Average Complexity: 0.3336
Average Reasoning Score: 3.20

### Social Psychology
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.60
Average Complexity: 0.3249
Average Reasoning Score: 3.10

### Social Skills
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.40
Average Complexity: 0.3226
Average Reasoning Score: 3.00

### Social Stratification
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.10
Average Complexity: 0.3154
Average Reasoning Score: 3.10

### Sociology General
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.70
Average Complexity: 0.3308
Average Reasoning Score: 3.30

### Sociology Of Education
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 41.30
Average Complexity: 0.3239
Average Reasoning Score: 3.20

### Sociology Of Family
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 40.50
Average Complexity: 0.3265
Average Reasoning Score: 3.10

### Software Architecture
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 33.40
Average Complexity: 0.5099
Average Reasoning Score: 5.20

### Software Development
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 32.90
Average Complexity: 0.4209
Average Reasoning Score: 4.20

### Software Testing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.50
Average Complexity: 0.4150
Average Reasoning Score: 4.20

### Sound Baths
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.00
Average Complexity: 0.3220
Average Reasoning Score: 3.10

### Sound Design
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.90
Average Complexity: 0.3222
Average Reasoning Score: 3.10

### Sous Vide
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.50
Average Complexity: 0.3185
Average Reasoning Score: 3.00

### Space Colonization Concepts
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.40
Average Complexity: 0.3287
Average Reasoning Score: 3.10

### Space Exploration
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.90
Average Complexity: 0.3181
Average Reasoning Score: 3.10

### Space Robotics
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.80
Average Complexity: 0.3483
Average Reasoning Score: 3.40

### Space Travel Non Political
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.40
Average Complexity: 0.3296
Average Reasoning Score: 3.20

### Spacecraft Design
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 38.90
Average Complexity: 0.3645
Average Reasoning Score: 3.70

### Spanish Language
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.70
Average Complexity: 0.3206
Average Reasoning Score: 3.10

### Special Education
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.80
Average Complexity: 0.3581
Average Reasoning Score: 3.50

### Special Needs Parenting
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 41.30
Average Complexity: 0.3196
Average Reasoning Score: 3.00

### Speech Recognition
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 36.80
Average Complexity: 0.4013
Average Reasoning Score: 4.10

### Speed Memorization
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.00
Average Complexity: 0.3228
Average Reasoning Score: 3.10

### Speed Reading
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.50
Average Complexity: 0.3418
Average Reasoning Score: 3.40

### Speedrunning
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 33.50
Average Complexity: 0.3274
Average Reasoning Score: 3.10

### Sports Analytics
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.80
Average Complexity: 0.3698
Average Reasoning Score: 3.80

### Sports Coaching
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.00
Average Complexity: 0.3646
Average Reasoning Score: 3.70

### Sports General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.50
Average Complexity: 0.3369
Average Reasoning Score: 3.30

### Sports Medicine
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 35.10
Average Complexity: 0.3608
Average Reasoning Score: 3.60

### Sports Nutrition
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.00
Average Complexity: 0.3518
Average Reasoning Score: 3.50

### Sports Psychology
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.60
Average Complexity: 0.3708
Average Reasoning Score: 3.80

### Stage Acting
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 34.80
Average Complexity: 0.3615
Average Reasoning Score: 3.50

### Stage Directing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.30
Average Complexity: 0.3603
Average Reasoning Score: 3.60

### Stage Lighting
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.80
Average Complexity: 0.3315
Average Reasoning Score: 3.20

### Stage Makeup
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.00
Average Complexity: 0.3516
Average Reasoning Score: 3.40

### Stage Management
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.80
Average Complexity: 0.3334
Average Reasoning Score: 3.30

### Stakeholder Engagement
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.90
Average Complexity: 0.3898
Average Reasoning Score: 3.70

### Stamp Collecting
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.90
Average Complexity: 0.3142
Average Reasoning Score: 3.10

### Stand Up Comedy
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.90
Average Complexity: 0.3581
Average Reasoning Score: 3.60

### Startup Culture
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.20
Average Complexity: 0.3186
Average Reasoning Score: 3.00

### Statistical Modeling
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 35.40
Average Complexity: 0.3947
Average Reasoning Score: 4.20

### Statistics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 33.60
Average Complexity: 0.3656
Average Reasoning Score: 3.70

### Stellar Evolution
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.10
Average Complexity: 0.3192
Average Reasoning Score: 3.00

### Stem Education
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.90
Average Complexity: 0.3380
Average Reasoning Score: 3.40

### Stir Frying
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 32.20
Average Complexity: 0.3182
Average Reasoning Score: 3.00

### Stock Market
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.10
Average Complexity: 0.3857
Average Reasoning Score: 3.80

### Stop Motion
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.10
Average Complexity: 0.3376
Average Reasoning Score: 3.30

### Storyboarding
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.80
Average Complexity: 0.3965
Average Reasoning Score: 4.10

### Storytelling For Children
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 35.70
Average Complexity: 0.3896
Average Reasoning Score: 4.10

### Strategic Planning
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.40
Average Complexity: 0.3841
Average Reasoning Score: 3.80

### Street Art
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.70
Average Complexity: 0.3354
Average Reasoning Score: 3.30

### Street Art History
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 39.80
Average Complexity: 0.3228
Average Reasoning Score: 3.10

### Stress Management
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.80
Average Complexity: 0.3174
Average Reasoning Score: 3.00

### Stress Physiology
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 35.90
Average Complexity: 0.3492
Average Reasoning Score: 3.50

### Stress Relief Activities
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.00
Average Complexity: 0.3285
Average Reasoning Score: 3.20

### Structural Design
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 35.60
Average Complexity: 0.3447
Average Reasoning Score: 3.30

### Succession Planning
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.60
Average Complexity: 0.3834
Average Reasoning Score: 3.70

### Sugar Sculpting
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 35.90
Average Complexity: 0.3186
Average Reasoning Score: 3.00

### Supply Chain Management
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 40.40
Average Complexity: 0.3893
Average Reasoning Score: 3.80

### Surfing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.00
Average Complexity: 0.3250
Average Reasoning Score: 3.10

### Surrealism
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 32.20
Average Complexity: 0.3787
Average Reasoning Score: 3.80

### Sustainability General
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.10
Average Complexity: 0.3400
Average Reasoning Score: 3.30

### Sustainability Metrics
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 40.00
Average Complexity: 0.3816
Average Reasoning Score: 3.70

### Sustainable Architecture
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.40
Average Complexity: 0.3439
Average Reasoning Score: 3.20

### Sustainable Design
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.10
Average Complexity: 0.3340
Average Reasoning Score: 3.20

### Sustainable Fashion
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.10
Average Complexity: 0.3364
Average Reasoning Score: 3.20

### Sustainable Home Design
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 40.60
Average Complexity: 0.3322
Average Reasoning Score: 3.10

### Sustainable Tourism
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.70
Average Complexity: 0.3236
Average Reasoning Score: 3.20

### Swarm Robotics
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 37.30
Average Complexity: 0.4325
Average Reasoning Score: 4.30

### Swift Programming
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.20
Average Complexity: 0.4144
Average Reasoning Score: 4.00

### Synthetic Biology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.50
Average Complexity: 0.3752
Average Reasoning Score: 3.70

### Systems Administration
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.80
Average Complexity: 0.5120
Average Reasoning Score: 5.30

### Talent Management
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.90
Average Complexity: 0.3719
Average Reasoning Score: 3.60

### Talent Shows
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 37.30
Average Complexity: 0.3619
Average Reasoning Score: 3.80

### Tattoo Art Non Erotic
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 37.30
Average Complexity: 0.3256
Average Reasoning Score: 3.10

### Taxation General
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.50
Average Complexity: 0.3297
Average Reasoning Score: 3.30

### Team Building
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.40
Average Complexity: 0.3490
Average Reasoning Score: 3.40

### Team Sports Non Political
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 34.70
Average Complexity: 0.3247
Average Reasoning Score: 3.10

### Technical Writing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.00
Average Complexity: 0.3499
Average Reasoning Score: 3.40

### Technology Trends
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.70
Average Complexity: 0.4001
Average Reasoning Score: 4.10

### Television Production
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3363
Average Reasoning Score: 3.30

### Textile Science
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.80
Average Complexity: 0.3354
Average Reasoning Score: 3.30

### Theater Arts
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.00
Average Complexity: 0.3627
Average Reasoning Score: 3.70

### Threat Intelligence
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.20
Average Complexity: 0.4406
Average Reasoning Score: 4.40

### Thriller
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 31.20
Average Complexity: 0.3615
Average Reasoning Score: 3.60

### Time Blocking
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.20
Average Complexity: 0.3368
Average Reasoning Score: 3.30

### Time Management
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.80
Average Complexity: 0.3321
Average Reasoning Score: 3.20

### Tiny Home Living
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 41.50
Average Complexity: 0.3299
Average Reasoning Score: 3.10

### Topology
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 31.60
Average Complexity: 0.3211
Average Reasoning Score: 3.00

### Tour Guiding
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.40
Average Complexity: 0.3220
Average Reasoning Score: 3.20

### Tourism
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 33.10
Average Complexity: 0.3352
Average Reasoning Score: 3.40

### Trade Shows
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.50
Average Complexity: 0.3240
Average Reasoning Score: 3.10

### Trading Card Games
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 38.30
Average Complexity: 0.3203
Average Reasoning Score: 3.10

### Trading Cards Sports
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.40
Average Complexity: 0.3219
Average Reasoning Score: 3.20

### Traditional Publishing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.90
Average Complexity: 0.3584
Average Reasoning Score: 3.60

### Transit Oriented Development
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.80
Average Complexity: 0.3352
Average Reasoning Score: 3.20

### Transportation General
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.00
Average Complexity: 0.3426
Average Reasoning Score: 3.40

### Travel Planning
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.10
Average Complexity: 0.3421
Average Reasoning Score: 3.40

### Travel Writing
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.00
Average Complexity: 0.3340
Average Reasoning Score: 3.30

### Trend Forecasting
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.70
Average Complexity: 0.3471
Average Reasoning Score: 3.40

### Trivia Nights
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.30
Average Complexity: 0.3272
Average Reasoning Score: 3.20

### Uav Drone Operation
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 39.30
Average Complexity: 0.3744
Average Reasoning Score: 3.60

### Ui Ux Design
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 33.80
Average Complexity: 0.3490
Average Reasoning Score: 3.40

### Underwater Archaeology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.70
Average Complexity: 0.3302
Average Reasoning Score: 3.30

### Upwork Strategies
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 38.00
Average Complexity: 0.3381
Average Reasoning Score: 3.20

### Urban Development
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.50
Average Complexity: 0.3331
Average Reasoning Score: 3.20

### Urban Economics Non Political
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 39.60
Average Complexity: 0.3197
Average Reasoning Score: 3.10

### Urban Gardening
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 36.20
Average Complexity: 0.3227
Average Reasoning Score: 3.10

### Urban Geography
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 38.00
Average Complexity: 0.3456
Average Reasoning Score: 3.30

### Urban Mobility
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.30
Average Complexity: 0.3387
Average Reasoning Score: 3.30

### Urban Planning
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.80
Average Complexity: 0.3357
Average Reasoning Score: 3.20

### Urban Studies
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 38.10
Average Complexity: 0.3263
Average Reasoning Score: 3.20

### Usability Testing
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.10
Average Complexity: 0.3686
Average Reasoning Score: 3.60

### User Interface Design
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.20
Average Complexity: 0.3492
Average Reasoning Score: 3.30

### User Research
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 36.40
Average Complexity: 0.3583
Average Reasoning Score: 3.60

### User Story Mapping
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 34.30
Average Complexity: 0.3306
Average Reasoning Score: 3.20

### Van Life
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 36.90
Average Complexity: 0.3229
Average Reasoning Score: 3.10

### Variety Shows
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 35.80
Average Complexity: 0.3520
Average Reasoning Score: 3.60

### Vegan Cooking
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.70
Average Complexity: 0.3226
Average Reasoning Score: 3.10

### Vegetarian Cooking
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 36.70
Average Complexity: 0.3193
Average Reasoning Score: 3.00

### Vendor Coordination
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.70
Average Complexity: 0.3677
Average Reasoning Score: 3.70

### Venture Capital
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 37.00
Average Complexity: 0.3860
Average Reasoning Score: 4.00

### Verbal Communication
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.60
Average Complexity: 0.3423
Average Reasoning Score: 3.50

### Veterinary Basics
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 38.80
Average Complexity: 0.3489
Average Reasoning Score: 3.40

### Video Conferencing
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 35.60
Average Complexity: 0.3254
Average Reasoning Score: 3.10

### Video Editing
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 37.10
Average Complexity: 0.3436
Average Reasoning Score: 3.30

### Video Games
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.00
Average Complexity: 0.3320
Average Reasoning Score: 3.10

### Vintage Clothing
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.10
Average Complexity: 0.3277
Average Reasoning Score: 3.10

### Virtual Assistance
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.10
Average Complexity: 0.3302
Average Reasoning Score: 3.20

### Virtual Events
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.00
Average Complexity: 0.3294
Average Reasoning Score: 3.20

### Virtual Events Advanced
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 38.00
Average Complexity: 0.3254
Average Reasoning Score: 3.10

### Virtual Reality
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 35.40
Average Complexity: 0.3437
Average Reasoning Score: 3.30

### Visitor Experience
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.00
Average Complexity: 0.3451
Average Reasoning Score: 3.50

### Visual Anthropology
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.30
Average Complexity: 0.3332
Average Reasoning Score: 3.40

### Visual Effects
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 33.50
Average Complexity: 0.3304
Average Reasoning Score: 3.20

### Voice Acting
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.40
Average Complexity: 0.3485
Average Reasoning Score: 3.40

### Voice Acting Advanced
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 38.00
Average Complexity: 0.3459
Average Reasoning Score: 3.40

### Volcanology
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 35.80
Average Complexity: 0.3669
Average Reasoning Score: 3.50

### Volunteer Management
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.50
Average Complexity: 0.3297
Average Reasoning Score: 3.10

### Volunteer Tourism
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.60
Average Complexity: 0.3403
Average Reasoning Score: 3.40

### Vulnerability Assessment
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 34.20
Average Complexity: 0.6326
Average Reasoning Score: 6.40

### Walking Meditation
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 34.40
Average Complexity: 0.3194
Average Reasoning Score: 3.00

### Wealth Management
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.60
Average Complexity: 0.3887
Average Reasoning Score: 3.90

### Wearable Tech
Total: 10
LOCAL: 4
REMOTE: 6
REMOTE Rate: 60.00%
Average Length: 37.20
Average Complexity: 0.3508
Average Reasoning Score: 3.60

### Web Comics
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 35.90
Average Complexity: 0.3842
Average Reasoning Score: 3.90

### Web Development
Total: 10
LOCAL: 1
REMOTE: 9
REMOTE Rate: 90.00%
Average Length: 33.30
Average Complexity: 0.4146
Average Reasoning Score: 4.20

### Web Development Freelancing
Total: 10
LOCAL: 2
REMOTE: 8
REMOTE Rate: 80.00%
Average Length: 34.00
Average Complexity: 0.3809
Average Reasoning Score: 3.90

### Weddings Non Religious Ceremony
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 37.40
Average Complexity: 0.3249
Average Reasoning Score: 3.00

### Wellness Advanced
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 34.70
Average Complexity: 0.3327
Average Reasoning Score: 3.20

### Wellness Practices
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 33.80
Average Complexity: 0.3240
Average Reasoning Score: 3.20

### Wildlife Conservation
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 37.20
Average Complexity: 0.3617
Average Reasoning Score: 3.60

### Wildlife Rehabilitation
Total: 10
LOCAL: 5
REMOTE: 5
REMOTE Rate: 50.00%
Average Length: 37.90
Average Complexity: 0.3437
Average Reasoning Score: 3.40

### Wine Collecting
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 37.40
Average Complexity: 0.3335
Average Reasoning Score: 3.20

### Woodworking
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 31.80
Average Complexity: 0.3215
Average Reasoning Score: 3.10

### Work Life Balance
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 36.60
Average Complexity: 0.3308
Average Reasoning Score: 3.20

### Workspace Organization
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 37.40
Average Complexity: 0.3294
Average Reasoning Score: 3.20

### World Cultures General
Total: 10
LOCAL: 8
REMOTE: 2
REMOTE Rate: 20.00%
Average Length: 36.30
Average Complexity: 0.3246
Average Reasoning Score: 3.10

### World Languages
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 29.70
Average Complexity: 0.3179
Average Reasoning Score: 3.00

### Written Communication
Total: 10
LOCAL: 6
REMOTE: 4
REMOTE Rate: 40.00%
Average Length: 35.50
Average Complexity: 0.3533
Average Reasoning Score: 3.50

### Yoga
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 31.50
Average Complexity: 0.3205
Average Reasoning Score: 3.00

### Young Adult Literature
Total: 10
LOCAL: 3
REMOTE: 7
REMOTE Rate: 70.00%
Average Length: 36.00
Average Complexity: 0.3649
Average Reasoning Score: 3.70

### Youtube Content Creation
Total: 10
LOCAL: 7
REMOTE: 3
REMOTE Rate: 30.00%
Average Length: 39.80
Average Complexity: 0.3405
Average Reasoning Score: 3.30

### Zero Trust Security
Total: 10
LOCAL: 0
REMOTE: 10
REMOTE Rate: 100.00%
Average Length: 36.40
Average Complexity: 0.4875
Average Reasoning Score: 5.00

### Zero Waste Lifestyle
Total: 10
LOCAL: 10
REMOTE: 0
REMOTE Rate: 0.00%
Average Length: 38.80
Average Complexity: 0.3210
Average Reasoning Score: 3.00

### Zoology
Total: 10
LOCAL: 9
REMOTE: 1
REMOTE Rate: 10.00%
Average Length: 29.70
Average Complexity: 0.3218
Average Reasoning Score: 3.10



---

## 2. Parent Category Analysis

The imported dataset (`10k_chatbot_prompts`) contains `parent_category` and `subcategory` metadata. We analyzed these 9,970 samples separately to isolate their routing patterns.

### Overrepresentation Analysis
The imported dataset represents **66.47%** of the entire merged dataset. In terms of category frequency, there is a highly artificial flat distribution:
- Out of **998 unique parent categories**, 996 have exactly **10 samples** each, and 2 have **20 samples** each.
- Niche, obscure academic or hobby categories (e.g., *Dollmaking*, *Origami*, *Braising*, *Archaeozoology*) are collectively heavily overrepresented compared to real-world production distributions, which are typically dominated by coding, general conversation, planning, and basic Q&A.

### Mostly LOCAL Parent Categories
The following parent categories have the lowest REMOTE rates (mostly routed to LOCAL):

| Parent Category | Total Samples | LOCAL | REMOTE | REMOTE % | Avg Reasoning |
|---|---:|---:|---:|---:|---:|
| Adventure Tourism | 10 | 10 | 0 | 0.0% | 3.00 |
| Alumni Networks | 10 | 10 | 0 | 0.0% | 3.00 |
| Animal Behavior | 10 | 9 | 1 | 10.0% | 3.10 |
| Animals (General) | 10 | 9 | 1 | 10.0% | 3.30 |
| Archaeobotany | 10 | 10 | 0 | 0.0% | 3.00 |
| Archaeology (Methods) | 10 | 9 | 1 | 10.0% | 3.10 |
| Archaeozoology | 10 | 10 | 0 | 0.0% | 3.00 |
| Aromatherapy | 10 | 10 | 0 | 0.0% | 3.00 |
| Art History (General) | 10 | 9 | 1 | 10.0% | 3.20 |
| Astronomy | 10 | 9 | 1 | 10.0% | 3.00 |
| Bartending (Mixology) | 10 | 10 | 0 | 0.0% | 3.00 |
| Beverage Crafting | 10 | 10 | 0 | 0.0% | 3.00 |
| Bilingualism | 10 | 9 | 1 | 10.0% | 3.10 |
| Bioarchaeology | 10 | 9 | 1 | 10.0% | 3.10 |
| Birdwatching | 10 | 9 | 1 | 10.0% | 3.10 |
| Book Marketing | 10 | 9 | 1 | 10.0% | 3.00 |
| CAD Design | 10 | 9 | 1 | 10.0% | 3.00 |
| Cake Sculpting | 10 | 9 | 1 | 10.0% | 3.00 |
| Calligraphy | 10 | 10 | 0 | 0.0% | 3.00 |
| Calligraphy (Advanced) | 10 | 10 | 0 | 0.0% | 3.00 |


*Interpretation*: Mostly LOCAL categories represent basic arts, crafts, simple language learning, culinary techniques, and daily life hobbies. These prompts have low technical complexity and average reasoning scores around 3.0, placing their routing scores below the threshold of 23.

### Mostly REMOTE Parent Categories
The following parent categories have the highest REMOTE rates (mostly routed to REMOTE):

| Parent Category | Total Samples | LOCAL | REMOTE | REMOTE % | Avg Reasoning |
|---|---:|---:|---:|---:|---:|
| 5G Communication | 10 | 1 | 9 | 90.0% | 4.20 |
| API Design | 10 | 0 | 10 | 100.0% | 5.10 |
| Adaptation Strategies | 10 | 0 | 10 | 100.0% | 3.60 |
| Adoption (General) | 10 | 1 | 9 | 90.0% | 3.90 |
| Agriculture | 10 | 1 | 9 | 90.0% | 3.60 |
| Algebra | 10 | 0 | 10 | 100.0% | 3.00 |
| Algorithms | 10 | 0 | 10 | 100.0% | 5.50 |
| Antique Furniture | 10 | 1 | 9 | 90.0% | 3.20 |
| Architecture (General) | 10 | 0 | 10 | 100.0% | 3.10 |
| Artificial Intelligence | 10 | 1 | 9 | 90.0% | 4.60 |
| Assessment Methods | 10 | 1 | 9 | 90.0% | 3.90 |
| AutoML | 10 | 0 | 10 | 100.0% | 4.60 |
| Ballet | 10 | 1 | 9 | 90.0% | 3.80 |
| Business Continuity | 10 | 1 | 9 | 90.0% | 4.90 |
| C Programming | 10 | 1 | 9 | 90.0% | 5.10 |
| C++ Programming | 10 | 0 | 10 | 100.0% | 4.50 |
| Calculus | 10 | 0 | 10 | 100.0% | 3.10 |
| Change Management | 10 | 1 | 9 | 90.0% | 3.90 |
| Circular Economy | 10 | 0 | 10 | 100.0% | 4.10 |
| Circus Performances | 10 | 0 | 10 | 100.0% | 3.90 |


*Interpretation*: Mostly REMOTE categories represent advanced computer science, cloud computing, mathematics, and software architecture. These prompts naturally contain dense technical terminology (e.g., *Kubernetes*, *API*, *Transformer*) and complex reasoning instructions, easily pushing them past the routing threshold of 23.

---

## 3. Prompt Length Analysis

Prompts were partitioned into the following length buckets:
- **0–100 words**
- **100–250 words**
- **250–500 words**
- **500+ words**

### Length Bucket Distribution
| Bucket | Total Samples | LOCAL Samples | REMOTE Samples | REMOTE % | Average Routing Score |
|---|---:|---:|---:|---:|---:|
| **0–100 words** | 15,000 | 5,850 | 9,150 | 61.0% | 30.99 |
| **100–250 words** | 0 | 0 | 0 | 0.0% | N/A |
| **250–500 words** | 0 | 0 | 0 | 0.0% | N/A |
| **500+ words** | 0 | 0 | 0 | 0.0% | N/A |

### Length Bias Evaluation
> [!IMPORTANT]
> **Key Finding**: Prompt length is **NOT** dominating routing decisions, but it represents a **fatal data coverage gap**.
> 
> 1. **Zero Length Variance**: Every single one of the 15,000 prompts lies between **24 and 80 words** (mean = 41.80 words). There is zero representation of medium (100–250), long (250–500), or extra-long (500+) prompts.
> 2. **Correlation within Bucket**: Within the 0–100 range, `word_count` has a correlation of `+0.37` with the REMOTE label, and `estimated_input_tokens` has `+0.27`.
> 3. **Implication**: A router trained on this dataset will have no concept of prompts longer than 80 words. If deployed in production, it will likely exhibit extreme out-of-distribution instability when faced with typical long-context user queries (e.g., code repositories, documents to summarize).

---

## 4. Routing Decision Analysis

To identify what features drive the routing decisions (LOCAL vs. REMOTE) under the deterministic labeling policy, we trained a Random Forest classifier and calculated Pearson correlations.

### Feature Importance and Correlation Rankings
| Rank | Feature | Random Forest Importance | Correlation with REMOTE | Description |
|---|---|---|---|---|
| 1 | `complexity_score` | 0.2505 | +0.5273 | Overall complexity score in [0, 1] mapped from semantic features. |
| 2 | `reasoning_score` | 0.2073 | +0.5903 | Integer reasoning score in [0, 10] (dominant routing signal). |
| 3 | `technical_complexity` | 0.1107 | +0.6126 | Heuristic technical complexity in [0, 1] based on domain keywords. |
| 4 | `api_keywords` | 0.1068 | +0.4994 | Count/presence of API-related engineering terms. |
| 5 | `task_complexity` | 0.0711 | +0.4149 | Cognitive task complexity score based on verb groupings. |
| 6 | `contains_code` | 0.0308 | +0.2866 | Boolean flag indicating code syntax or keywords. |
| 7 | `code_indicators` | 0.0275 | +0.2952 | Counts of specific coding keyword matches. |
| 8 | `constraint_density` | 0.0254 | +0.3590 | Ratio of constraints to prompt length. |
| 9 | `reasoning_depth` | 0.0249 | -0.0715 | Depth score of logical/analytical reasoning patterns. |
| 10 | `system_design_keywords` | 0.0213 | +0.1689 | Count of system architecture and design terms. |
| 11 | `algorithmic_complexity` | 0.0209 | +0.1449 | Presence of algorithm/complexity terms (e.g. Big-O). |
| 12 | `math_indicators` | 0.0205 | +0.2432 | Count of math keywords or LaTeX symbols. |
| 13 | `contains_math` | 0.0197 | +0.1798 | Boolean flag indicating mathematical symbols/keywords. |
| 14 | `context_complexity` | 0.0176 | +0.2727 | Lexical context load and concept density score. |
| 15 | `constraint_complexity` | 0.0161 | +0.3941 | Aggregate score for formatting/style constraints. |
| 16 | `word_count` | 0.0094 | +0.3713 | Number of words in the prompt. |
| 17 | `prompt_length` | 0.0070 | +0.2770 | Character count of the prompt. |
| 18 | `contains_question` | 0.0057 | -0.3390 | Presence of interrogative keywords or question marks. |
| 19 | `estimated_input_tokens` | 0.0031 | +0.2770 | Estimated token length based on char count. |
| 20 | `sql_indicators` | 0.0020 | -0.2772 | Presence of SQL query statements or keywords. |
| 21 | `contains_numbers` | 0.0009 | +0.1625 | Presence of numeric digits. |
| 22 | `dependency_between_subtasks` | 0.0007 | +0.2674 | Score for logical subtask dependencies. |
| 23 | `multi_turn_context` | 0.0001 | -0.0051 | Flag for conversational turn history. |
| 24 | `contains_json` | 0.0000 | +nan | Boolean flag indicating JSON brackets or keys. |
| 25 | `contains_markdown` | 0.0000 | +0.0065 | Presence of Markdown styling tags. |
| 26 | `presence_of_tables` | 0.0000 | +nan | Presence of Markdown or CSV table patterns. |


### Key Predictors Analysis
1. **`complexity_score` and `reasoning_score` (Dominant)**: Together, these account for over **45%** of the feature importance. This is by design: `reasoning_score` contributes `5` points per level to the routing score. A reasoning score of 5 adds 25 points, immediately crossing the threshold of 23.
2. **`technical_complexity` and `api_keywords`**: Technical domain signals are heavily weighted (+12 points for technical complexity, +3 for API keywords).
3. **`word_count` and `estimated_input_tokens`**: Word count and token pressure are weak predictors (ranked 16th and 17th) because the length of prompts is constrained between 24 and 80 words.

---

## 5. Category Imbalance Detection

We detected severe routing anomalies across major categories, indicating unrealistic routing behavior:

### Anomalous Categories Table
| Category | Total Samples | LOCAL | REMOTE | REMOTE % | Avg Reasoning | Status |
|---|---:|---:|---:|---:|---:|---|
| **Reasoning** | 697 | 0 | 697 | 100.0% | 4.76 | 🚨 Unrealistic |
| **Planning** | 698 | 7 | 691 | 99.0% | 5.57 | 🚨 Unrealistic |
| **Translation** | 698 | 2 | 696 | 99.7% | 4.22 | 🚨 Unrealistic |
| **Mathematics** | 708 | 2 | 706 | 99.7% | 4.71 | 🚨 Unrealistic |
| **General** | 135 | 6 | 129 | 95.6% | 4.21 | 🚨 Unrealistic |
| **Coding** | 698 | 132 | 566 | 81.1% | 3.77 | 🚨 Unrealistic |

### Why these routing ratios are unrealistic:
- **Translation (99.7% REMOTE)**: Translation is a straightforward task that local models excel at. Routing 99.7% of translation queries to a remote model is economically and operationally absurd. It occurs because the translation prompts in the existing project data are synthetically stuffed with technical and clinical terminology (e.g., *"Translate this operational notice about incorrect dosage guidance into French..."*), which artificially inflates the technical complexity score.
- **Reasoning (100% REMOTE) and Planning (99% REMOTE)**: Simple planning or reasoning queries (e.g., making a basic shopping checklist or simple logical comparisons) can easily run locally. They are sent remote because of **instruction inflation** in the templates (e.g., forcing *"Discuss edge cases and failure modes"* or *"Make assumptions explicit for auditability"*), which adds heavy analytical reasoning scores.
- **Mathematics (99.7% REMOTE)**: As analyzed, the threshold is 23. A math prompt gets a +4 task type nudge and a +4 `contains_math` bonus. If its reasoning score is 3 or more (which is the case for 100% of the mathematics prompts in this dataset), its routing score is at least `(3 * 5) + 4 + 4 = 23`. Thus, **every math prompt with reasoning score >= 3 is routed REMOTE**, regardless of how simple the math is (e.g., simple algebra or geometry).

---

## 6. Dataset Quality Audit

A thorough audit of the dataset's quality reveals the following statistics:

- **Duplicate Rate**: **0.0000%** (0 exact duplicates). The initial deduplication step successfully removed exact string duplicates.
- **Near-Duplicate Rate (Global)**: **0.1733%** (26 pairs found). 
  *Note*: The original dataset preparation script used a local window of 1,500 elements for Jaccard deduplication. Our global audit revealed **26 near-duplicate pairs** across the entire 15,000 dataset that slipped past the local deduplication window.
- **Empty Prompts**: **0** (no null, empty, or whitespace-only prompts).
- **Missing Metadata**: **0** (no missing cells or NaNs in the feature columns).
- **Average Prompt Diversity (TTR)**: **0.0225** (Type-Token Ratio). This TTR is exceptionally low, indicating extreme vocabulary homogeneity. This is a direct consequence of prompts being synthetically generated via repetitive templates.

### Class Overlap & Label Inconsistency Case Study
Out of the 26 near-duplicate pairs, we found **1 pair** with conflicting routing labels (one LOCAL, one REMOTE).

#### Label Inconsistency Example:
- **Prompt 1 (LOCAL, Routing Score: 19)**: 
  `"Compose a reflective story in which district becomes a symbol for changing processing time. Include practical examples, important trade-offs, and implementation guidance where relevant."`
- **Prompt 2 (REMOTE, Routing Score: 24)**: 
  `"Compose a reflective story in which public dataset becomes a symbol for changing processing time. Include practical examples, important trade-offs, and implementation guidance where relevant."`

#### Root Cause Analysis:
The only lexical difference is the substitution of *"district"* with *"public dataset"*. 
In the feature extractor, the word **`public`** matches Java/C++ access modifier syntax and triggers the `contains_code` flag. This adds **5 points** to the routing score (pushing P2 from 19 to 24), crossing the threshold of 23 and routing P2 to REMOTE. 
This demonstrates that the rule-based labeling pipeline is fragile and introduces labels that are inconsistent and noisy.

---

## 7. Training Readiness

### Is this dataset suitable for production training?
> [!CAUTION]
> **NO**

### Key Reasons for Unsuitability:
1. **Severe Label Bias**: 90.6% of the project-specific data is labeled REMOTE. Training a model on this will result in a model that routes almost everything to REMOTE, destroying the latency/cost savings of a hybrid router.
2. **Artificial Suffix Inflation**: 100% of the imported dataset (representing 66% of the merged dataset) contains the exact same template suffix: `". Include practical examples, important trade-offs, and implementation guidance where relevant."` This artificially adds +15 points to the routing score of all 10,000 chatbot prompts.
3. **No Length Diversity**: The dataset contains zero prompts longer than 80 words. The model will fail to generalize to typical long production inputs.
4. **Keyword Fragility**: The deterministic labeling policy has built-in fragility (e.g., the word "public" triggering a code flag), causing near-identical prompts to get conflicting labels.

---

## 8. Recommendations

To make this dataset suitable for production training, we recommend the following minimal corrections:

1. **Remove Template Suffixes**: Strip the synthetic trailing suffix (*"Include practical examples..."*) from the imported dataset before feature extraction. This will restore the true, uninflated complexity scores of the baseline prompts.
2. **Introduce Long-Context Prompts**: Inject a small, curated set (e.g., 2,000–3,000 prompts) of actual long-context user queries (summarizations, documents, codebases) ranging from 100 to 2,000+ words to ensure length coverage.
3. **Adjust Math & Translation Routing Thresholds**: Mutate the task-type weights or threshold in `RouterConfig` so that simple math (Algebra/Geometry) and standard translation tasks do not unconditionally trigger REMOTE routing.
4. **Refine Code Detection Rules**: Update `_RE_CODE_PATTERN` in the feature extractor to require structural context for keywords like `public` (e.g., `public class` or `public static`) to prevent common English nouns from falsely triggering the code flag.
5. **Run Global Deduplication**: Perform Jaccard deduplication globally over the entire merged dataset, rather than in a sliding window of 1,500, to remove the remaining 26 near-duplicates.

---

### Conclusion

"The following issues should be corrected before training."
